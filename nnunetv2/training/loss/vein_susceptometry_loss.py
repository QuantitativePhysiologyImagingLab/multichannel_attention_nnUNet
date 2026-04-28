import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import torch.nn.functional as F
import numpy as np

class PhysicsFieldLoss(nn.Module):
    """
    Physics-informed loss for vein segmentation with full-field dipole matching.

    Expected inputs:
      - net_output: (B, C, X, Y, Z) logits   (softmax over C; 'vein_channel' gives p)
      - target:     (B, 2, X, Y, Z)
            target[:,0] = chi_qsm_ppm       (tissue susceptibility, ppm)
            target[:,1] = localfield_ppm    (measured local field after LBV/RESHARP, ppm)
      - Optional kwargs:
            brain_mask:   (B,1,X,Y,Z) 0/1   (defaults to target[:,0]!=0)
            vein_eval:    (B,1,X,Y,Z) 0/1   (region to compute masked MAE/sign/topK; defaults to predicted mask)
            b0_dir:       (3,) or (B,3)     unit vector(s) in image axes (x,y,z)
            voxel_size:   (sx,sy,sz) in mm  spacing (image axes)
            chi_blood_ppm: scalar           blood susceptibility used inside veins
            topk_frac:    float in (0,1)    fraction for tail loss (default 0.10)
            lambdas: dict with keys {'phys','mae','tail','sign'}

    Returns:
      total_loss, metrics_dict
    """
    def __init__(self,
                 vein_channel=1,
                 chi_blood_ppm=0.1,
                 lambdas=None,
                 topk_frac=0.10,
                 # NEW defaults
                 default_voxel_size=(0.6,0.6,0.6),              # e.g. (0.6,0.6,0.6)
                 expects_b0_from_forward=True,         # require b0_dir in forward by default
                 eval_mask_mode='predicted_hard',      # or 'provided'
                 learnable_chi=False,
                 chi_blood_bounds=(0.15, 0.45),
                 b_meas_ref=0.007,                     # mean(|B_meas|) for the reference method (TGV)
                 debug=True):
        super().__init__()

        self.debug = debug

        self.vein_channel = vein_channel
        self.topk_frac = float(topk_frac)
        self.lambdas = dict(phys=0.55, mae=0.25, tail=0.15, sign=0.05) if lambdas is None else dict(lambdas)

        # store defaults
        self.default_voxel_size = default_voxel_size
        self.expects_b0_from_forward = expects_b0_from_forward
        self.eval_mask_mode = eval_mask_mode
        self.b_meas_ref = float(b_meas_ref)

        # chi_blood as (optionally) learnable, with clamping in forward
        # self._chi_blood = nn.Parameter(torch.tensor(float(chi_blood_ppm)), requires_grad=learnable_chi)
        self._chi_blood = float(chi_blood_ppm)
        self._chi_bounds = tuple(map(float, chi_blood_bounds))
        
    @staticmethod
    def _dipole_field_from_chi(chi_xyz, voxel_size, b0_dir):
        """
        chi_xyz: (B,1,X,Y,Z) ppm
        voxel_size: (sx,sy,sz) mm
        b0_dir: (B,3) or (3,) unit vectors in image axes (x,y,z)
        returns: (B,1,X,Y,Z) ppm
        """
        B, _, X, Y, Z = chi_xyz.shape
        dtype = chi_xyz.dtype
        device = chi_xyz.device

        sx, sy, sz = map(float, voxel_size)  # spacing in image axes (x,y,z)

        kx = torch.fft.fftfreq(X, d=sx, device=device, dtype=dtype)
        ky = torch.fft.fftfreq(Y, d=sy, device=device, dtype=dtype)
        kz = torch.fft.fftfreq(Z, d=sz, device=device, dtype=dtype)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')  # (X,Y,Z)

        if b0_dir.ndim == 1:
            b0 = b0_dir.to(device=device, dtype=dtype) / (b0_dir.norm()+1e-30)
            bx, by, bz = b0[0], b0[1], b0[2]
            kb = bx*KX + by*KY + bz*KZ
            k2 = KX*KX + KY*KY + KZ*KZ
            D = (1.0/3.0) - (kb*kb) / (k2 + torch.finfo(dtype).eps)
            D[0,0,0] = 0.0
            D = D[None,None]  # (1,1,X,Y,Z) broadcast over batch
        else:
            # per-sample b0 (B,3)
            b0 = b0_dir.to(device=device, dtype=dtype)
            b0 = b0 / (b0.norm(dim=1, keepdim=True)+1e-30)
            bx, by, bz = b0[:,0:1,None,None,None], b0[:,1:2,None,None,None], b0[:,2:3,None,None,None]
            KXb = KX[None,None]; KYb = KY[None,None]; KZb = KZ[None,None]
            kb = bx*KXb + by*KYb + bz*KZb
            k2 = KXb*KXb + KYb*KYb + KZb*KZb
            D = (1.0/3.0) - (kb*kb) / (k2 + torch.finfo(dtype).eps)
            D[:,:,0,0,0] = 0.0  # DC

        Chi_k = torch.fft.fftn(chi_xyz, dim=(-3,-2,-1))
        field_k = Chi_k * D
        field = torch.fft.ifftn(field_k, dim=(-3,-2,-1)).real
        return field

    def _resize_like(self, x, ref, is_mask=False):
        if x.shape[2:] == ref.shape[2:]:
            return x
        mode = 'nearest' if is_mask else 'trilinear'
        return F.interpolate(x, size=ref.shape[2:], mode=mode, align_corners=False if mode=='trilinear' else None)

    def forward(self, net_output, data,
                b0_dir,
                target=None,
                brain_mask=None, vein_eval=None,
                chi_blood_ppm=None):
        """
        net_output: (B,C,X,Y,Z)
        target:     (B,2,X,Y,Z)  [0]=chi_qsm_ppm, [1]=localfield_ppm
        b0_dir:     (3,) or (B,3)
        voxel_size: (sx,sy,sz) in mm
        """
        # Cast priors to the net dtype/device
        chi_qsm = data[:, 0:1].to(device=net_output.device, dtype=net_output.dtype)   # ppm
        B_meas  = data[:, 1:2].to(device=net_output.device, dtype=net_output.dtype)   # ppm

        probs = torch.softmax(net_output, dim=1)  # (B,C,X,Y,Z)
        vein_p = probs[:, self.vein_channel:self.vein_channel+1]  # (B,1,X,Y,Z)

        if brain_mask is None:
            brain_mask = (chi_qsm != 0).to(net_output.dtype)
        else:
            brain_mask = brain_mask.to(net_output.dtype)

        # choose eval mask: provided or predicted (hard)
        if vein_eval is None:
            vein_eval = (vein_p >= 0.5).to(net_output.dtype)
        else:
            vein_eval = vein_eval.to(net_output.dtype)

        chi_qsm    = self._resize_like(chi_qsm,    net_output, is_mask=False)
        B_meas     = self._resize_like(B_meas,     net_output, is_mask=False)
        brain_mask = self._resize_like(brain_mask, net_output, is_mask=True)

        # Always compute gt_vein_mask when target is available — needed for both
        # chi_b estimation and the gradient-aware chi_total formulation below.
        gt_vein_mask = None
        if target is not None:
            gt_vein_mask = (target[:, 0:1] == self.vein_channel).to(device=net_output.device)
            gt_vein_mask = self._resize_like(gt_vein_mask.float(), chi_qsm, is_mask=True)

        # chi_blood (ppm): derive from GT vein mask if target is provided, else fall back
        if chi_blood_ppm is not None:
            chi_b = torch.tensor(chi_blood_ppm, device=net_output.device, dtype=net_output.dtype)
        elif gt_vein_mask is not None:
            gt_vein_bool = (gt_vein_mask > 0) & (brain_mask > 0)
            if gt_vein_bool.any():
                vein_vals = chi_qsm[gt_vein_bool].float()
                p5,  p95  = torch.quantile(vein_vals, torch.tensor([0.05, 0.95], device=vein_vals.device))
                vein_vals = vein_vals.clamp(p5, p95)
                p20       = torch.quantile(vein_vals, 0.20)
                top80     = vein_vals[vein_vals >= p20]
                chi_b     = top80.mean()
                print(f"[PHYSICS] chi_b={float(chi_b):.4f} ppm "
                      f"(top-80% of {gt_vein_bool.sum().item()} GT vein voxels)", flush=True)
            else:
                chi_b = torch.tensor(float(self._chi_blood), device=net_output.device, dtype=net_output.dtype)
        else:
            if isinstance(self._chi_blood, torch.Tensor):
                chi_b = self._chi_blood.to(device=net_output.device, dtype=net_output.dtype)
            else:
                chi_b = torch.tensor(float(self._chi_blood), device=net_output.device, dtype=net_output.dtype)

        chi_b = chi_b.view(1, 1, 1, 1, 1)

        # ── Residual-field physics loss ──────────────────────────────────────
        #
        # Old approach: compare dipole_field(chi_total) vs B_meas globally.
        #   Problem: veins are 1-2% of brain volume, so their field contribution
        #   is tiny relative to background tissue → gradient ≈ 0 in practice.
        #   Test showed only 1.07-1.23x separation between GT and bad segmentations.
        #
        # New approach (residual field):
        #   1. Subtract the background field (from non-vein tissue) out of B_meas.
        #      B_residual = B_meas − dipole_field(chi_qsm * (1 − gt_vein_mask))
        #      This isolates the measured field signal attributable to veins.
        #   2. Predict only the vein field contribution:
        #      B_vein_pred = dipole_field(vein_p * chi_b)
        #   3. Loss = MAE(B_vein_pred, B_residual) in brain.
        #
        # Benefits:
        #   - Gradient ∝ chi_b (always ~0.1 ppm, never near-zero).
        #   - Loss is sensitive to vein *location* not just total field accuracy.
        #   - Test showed 1.25x separation for mid-quality segmentations (vs 1.07x old).
        #   - B_residual is computed once with no_grad — only vein_p backprops.

        if gt_vein_mask is not None:
            chi_bg = chi_qsm.detach() * (1.0 - gt_vein_mask.detach())
        else:
            chi_bg = chi_qsm.detach()

        with torch.no_grad():
            B_background = self._dipole_field_from_chi(chi_bg, self.default_voxel_size, b0_dir)
            B_residual   = B_meas - B_background   # vein-attributed field

        B_vein_pred = self._dipole_field_from_chi(
            vein_p * chi_b.view(1, 1, 1, 1, 1), self.default_voxel_size, b0_dir
        )

        # MAE between predicted vein field and residual field, inside brain
        loss_phys = ((B_vein_pred - B_residual).abs() * brain_mask).sum() / \
                    brain_mask.sum().clamp_min(1.0)

        # Top-10% tail loss on the same residual (catches localised errors)
        resid_vals = (B_vein_pred - B_residual).abs()[brain_mask > 0].flatten()
        k = max(1, int(self.topk_frac * resid_vals.numel()))
        top10 = torch.topk(resid_vals, k).values.mean()

        # Sign consistency: vein field and residual field should agree in sign
        sign_hinge = F.relu(-(B_vein_pred * B_residual.detach())[brain_mask > 0]).mean()

        # Simple weighted combination (drop mae_masked — redundant with loss_phys now)
        L = loss_phys + 0.15 * top10 + 0.05 * sign_hinge

        # Scale DOWN for noisier QSMs whose chi_bg is less accurate, producing a
        # noisier B_residual that would inflate the loss unfairly.
        b_scale = B_meas[brain_mask > 0].float().abs().mean().clamp_min(1e-8).detach()
        scale_factor = min(float(self.b_meas_ref / b_scale), 1.0)
        L_unscaled = L.detach()
        L = L * scale_factor

        # alias for debug/metrics compatibility
        mae_masked = loss_phys

        # ---------- DEBUG: check components & total ----------
        if self.debug:
            lp = float(loss_phys); lm = float(mae_masked)
            lt = float(top10);     ls = float(sign_hinge)
            L_val = float(L)

            # Always print a one-liner so we can monitor amplitude from iteration 1
            print(
                f"[PHYSICS] phys={lp:.5f} mae={lm:.5f} top10={lt:.5f} sign={ls:.5f} "
                f"L_raw={float(L_unscaled):.5f} b_scale={float(b_scale):.5f} "
                f"scale_factor={scale_factor:.3f} L_final={L_val:.5f}",
                flush=True
            )

            if (not torch.isfinite(loss_phys)) or (not torch.isfinite(mae_masked)) \
               or (not torch.isfinite(top10))   or (not torch.isfinite(sign_hinge)) \
               or (not torch.isfinite(L)):
                print("[PHYSICS NAN] non-finite component detected — see values above", flush=True)
        
        # print(L)

        metrics = {
            'loss_total': L.detach(),
            'loss_phys': loss_phys.detach(),
            'loss_mae': mae_masked.detach(),
            'loss_top10': top10.detach(),
            'loss_sign': sign_hinge.detach(),
            'chi_blood_ppm': chi_b.detach() if isinstance(chi_b, torch.Tensor) else torch.tensor(chi_b),
        }
        return L, metrics