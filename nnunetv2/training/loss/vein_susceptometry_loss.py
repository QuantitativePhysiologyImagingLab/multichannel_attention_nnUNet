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
                 chi_blood_bounds=(0.15, 0.45)):
        super().__init__()
        self.vein_channel = vein_channel
        self.topk_frac = float(topk_frac)
        self.lambdas = dict(phys=0.55, mae=0.25, tail=0.15, sign=0.05) if lambdas is None else dict(lambdas)

        # store defaults
        self.default_voxel_size = default_voxel_size
        self.expects_b0_from_forward = expects_b0_from_forward
        self.eval_mask_mode = eval_mask_mode

        # chi_blood as (optionally) learnable, with clamping in forward
        self._chi_blood = nn.Parameter(torch.tensor(float(chi_blood_ppm)), requires_grad=learnable_chi)
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

    def forward(self, net_output, target, *,
                b0_dir,
                brain_mask=None, vein_eval=None,
                chi_blood_ppm=None):
        """
        net_output: (B,C,X,Y,Z)
        target:     (B,2,X,Y,Z)  [0]=chi_qsm_ppm, [1]=localfield_ppm
        b0_dir:     (3,) or (B,3)
        voxel_size: (sx,sy,sz) in mm
        """
        probs = torch.softmax(net_output, dim=1)  # (B,C,X,Y,Z)
        vein_p = probs[:, self.vein_channel:self.vein_channel+1]  # (B,1,X,Y,Z)

        chi_qsm = target[:, 0:1]                      # (B,1,X,Y,Z), ppm
        B_meas  = target[:, 1:2]                    # (B,1,X,Y,Z), ppm

        if brain_mask is None:
            brain_mask = (chi_qsm != 0).float()
        else:
            brain_mask = brain_mask.float()

        # choose eval mask: provided or predicted (hard)
        if vein_eval is None:
            vein_eval = (vein_p >= 0.5).float()
        else:
            vein_eval = vein_eval.float()

        # chi_blood (ppm)
        chi_b = self._chi_blood if chi_blood_ppm is None else torch.tensor(float(chi_blood_ppm),
                                                                           device=net_output.device,
                                                                           dtype=net_output.dtype)

        # Composite susceptibility (detach chi_qsm so we don't backprop into it)
        chi_total = (1.0 - vein_p.detach()) * chi_qsm.detach() + vein_p * chi_b.view(1,1,1,1,1)

        # Full-volume dipole forward model
        B_pred = self._dipole_field_from_chi(chi_total, self.default_voxel_size, b0_dir)  # (B,1,X,Y,Z)

        # ----- physics terms (mirroring your previous script) -----

        # 1) Weighted whole-brain MAE (evidence = |B_meas|, normalized & capped)
        w = B_meas.abs()
        w = (w / (w.mean() + 1e-8)).clamp(max=5.).detach()
        num = (w * (B_pred - B_meas).abs() * brain_mask).sum()
        den = (w * brain_mask).sum().clamp_min(1.0)
        loss_phys = num / den

        # 2) Masked MAE on evaluation region (typically veins / dilated veins)
        valid = (vein_eval > 0) & (brain_mask > 0)
        if valid.any():
            mae_masked = ( (B_pred - B_meas).abs()[valid] ).mean()
            # 3) Sign hinge (penalize wrong sign)
            sign_hinge = F.relu(0.0 - (B_pred[valid] * B_meas[valid])).mean()
            # 4) Top-10% mean error in eval region
            vals = (B_pred - B_meas).abs()[valid].flatten()
            k = max(1, int(self.topk_frac * vals.numel()))
            top10 = torch.topk(vals, k).values.mean()
        else:
            mae_masked = torch.zeros((), device=net_output.device, dtype=net_output.dtype)
            sign_hinge = torch.zeros_like(mae_masked)
            top10 = torch.zeros_like(mae_masked)

        # ----- combine with scale-aware λ’s (fixed shares by default) -----
        # If you prefer adaptive λ’s, swap this block with the EMA weighting we discussed.
        L = (self.lambdas['phys'] * loss_phys +
             self.lambdas['mae']  * mae_masked +
             self.lambdas['tail'] * top10 +
             self.lambdas['sign'] * sign_hinge)

        metrics = {
            'loss_total': L.detach(),
            'loss_phys': loss_phys.detach(),
            'loss_mae': mae_masked.detach(),
            'loss_top10': top10.detach(),
            'loss_sign': sign_hinge.detach(),
            'chi_blood_ppm': chi_b.detach() if isinstance(chi_b, torch.Tensor) else torch.tensor(chi_b),
        }
        return L, metrics