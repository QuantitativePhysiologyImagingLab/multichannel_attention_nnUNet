"""
Detailed breakdown of why physics loss differs across subjects.
Runs only the physics forward pass and prints every intermediate quantity.

Usage:
    conda run -n manskelab python physics_breakdown.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import glob
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib

from nnunetv2.training.loss.vein_susceptometry_loss import PhysicsFieldLoss

IMAGES_DIR    = "/Users/yousifalkhoury/Downloads/medi_vein_seg/imagesTr"
LABELS_DIR    = "/Users/yousifalkhoury/Downloads/medi_vein_seg/labels"
B0_DIR        = torch.tensor([0.0, 0.0, 1.0])

# Background-removed unwrapped phase (Hz at 7T) to substitute as B_meas for 122-125
PHASE_PATH    = "/Users/yousifalkhoury/Downloads/medi_vein_seg/unwrapped_phase1_bgrm.nii.nii.gz"
HZ_TO_PPM     = 1.0 / (42.5774 * 7.0)   # 7T: 1 ppm = 298.04 Hz
USE_PHASE_FOR = {"VEIN_122", "VEIN_123", "VEIN_124", "VEIN_125"}

physics_fn = PhysicsFieldLoss(vein_channel=1, chi_blood_ppm=0.1, debug=False)
LAMBDAS = dict(phys=0.55, mae=0.25, tail=0.15, sign=0.05)


def stat(t, mask=None, name=""):
    v = t[mask].float() if mask is not None else t.float()
    v = v[torch.isfinite(v)]
    print(f"    {name:30s}  n={v.numel():>9,}  "
          f"mean={float(v.mean()):+.4f}  std={float(v.std()):+.4f}  "
          f"min={float(v.min()):+.4f}  max={float(v.max()):+.4f}")


def load_subject(sid):
    ch0 = nib.load(os.path.join(IMAGES_DIR, f"{sid}_0000.nii.gz")).get_fdata(dtype=np.float32)
    ch2 = nib.load(os.path.join(IMAGES_DIR, f"{sid}_0002.nii.gz")).get_fdata(dtype=np.float32)

    if sid in USE_PHASE_FOR:
        ch1 = nib.load(PHASE_PATH).get_fdata(dtype=np.float32) * HZ_TO_PPM
        print(f"  [B_meas source] unwrapped_phase1_bgrm (×{HZ_TO_PPM:.5f} Hz→ppm)  "
              f"std={ch1[ch1!=0].std():.4f} ppm")
    else:
        ch1 = nib.load(os.path.join(IMAGES_DIR, f"{sid}_0001.nii.gz")).get_fdata(dtype=np.float32)

    data  = torch.from_numpy(np.stack([ch0, ch1, ch2], 0)).unsqueeze(0)  # (1,3,D,H,W)
    label = torch.from_numpy(
                nib.load(os.path.join(LABELS_DIR, f"{sid}.nii.gz")).get_fdata(dtype=np.float32)
            ).long().unsqueeze(0).unsqueeze(0)                            # (1,1,D,H,W)
    return data, label


def make_net_output(label):
    vein   = (label[:, 0] == 1).float()
    logits = torch.stack([2.0*(1-vein)-1.0, 2.0*vein-1.0], dim=1).float()
    return logits + 0.3 * torch.randn_like(logits)


def breakdown(sid):
    print(f"\n{'='*70}")
    print(f"  {sid}")
    print(f"{'='*70}")

    data, label = load_subject(sid)
    net_output  = make_net_output(label)
    b0          = B0_DIR.view(1, 3).float()

    with torch.no_grad():
        # --- extract fields ---
        chi_qsm = data[:, 0:1].float()   # (1,1,D,H,W)
        B_meas  = data[:, 1:2].float()

        brain_mask = (chi_qsm != 0).float()
        gt_vein_mask = (label[:, 0:1] == 1).float()
        n_vein = int(gt_vein_mask.sum())
        n_brain = int(brain_mask.sum())

        # --- input data stats ---
        print(f"\n  Input statistics  (brain voxels={n_brain:,}, vein voxels={n_vein:,})")
        stat(chi_qsm, brain_mask.bool(), "chi_qsm [brain] (ppm)")
        stat(chi_qsm, gt_vein_mask.bool(), "chi_qsm [GT vein] (ppm)")
        stat(B_meas,  brain_mask.bool(), "B_meas  [brain] (ppm)")
        stat(B_meas,  gt_vein_mask.bool(), "B_meas  [GT vein] (ppm)")

        # --- chi_b estimate ---
        vein_vals = chi_qsm[gt_vein_mask.bool()].float()
        p20  = torch.quantile(vein_vals, 0.20)
        chi_b = vein_vals[vein_vals >= p20].mean()
        print(f"\n  chi_b (top-80% GT vein chi_qsm) = {float(chi_b):.4f} ppm")

        # --- forward model ---
        vein_p    = torch.softmax(net_output, dim=1)[:, 1:2]
        vein_eval = (vein_p >= 0.5).float()
        chi_b_t   = chi_b.view(1,1,1,1,1)
        chi_total = (1.0 - vein_eval) * chi_qsm + vein_eval * vein_p * chi_b_t
        B_pred    = physics_fn._dipole_field_from_chi(chi_total, (0.6, 0.6, 0.6), b0)

        print(f"\n  Forward model")
        stat(B_pred, brain_mask.bool(), "B_pred [brain] (ppm)")
        stat(B_pred, gt_vein_mask.bool(), "B_pred [GT vein] (ppm)")

        # --- residuals ---
        resid = (B_pred - B_meas).abs()
        print(f"\n  Residuals  |B_pred - B_meas|")
        stat(resid, brain_mask.bool(), "residual [brain]")
        stat(resid, gt_vein_mask.bool(), "residual [GT vein]")

        # --- per-lambda contributions ---
        w = B_meas.abs()
        w = (w / (w.mean() + 1e-8)).clamp(max=5.)
        loss_phys  = ((w * resid * brain_mask).sum()
                      / (w * brain_mask).sum().clamp_min(1.0))

        valid = (vein_eval > 0) & (brain_mask > 0)
        mae_masked = resid[valid].mean() if valid.any() else torch.zeros(1)
        sign_hinge = F.relu(-(B_pred[valid] * B_meas[valid])).mean() if valid.any() else torch.zeros(1)
        vals = resid[valid].flatten()
        k    = max(1, int(0.10 * vals.numel()))
        top10 = torch.topk(vals, k).values.mean() if vals.numel() > 0 else torch.zeros(1)

        L = (LAMBDAS['phys'] * loss_phys + LAMBDAS['mae'] * mae_masked
             + LAMBDAS['tail'] * top10 + LAMBDAS['sign'] * sign_hinge)

        print(f"\n  Physics loss breakdown")
        print(f"    {'term':<20}  {'raw value':>12}  {'lambda':>8}  {'contribution':>14}")
        print(f"    {'-'*60}")
        for name, raw, lam in [
            ("loss_phys (wt-MAE)", loss_phys, LAMBDAS['phys']),
            ("mae_masked (vein)", mae_masked, LAMBDAS['mae']),
            ("top10 tail",        top10,      LAMBDAS['tail']),
            ("sign_hinge",        sign_hinge, LAMBDAS['sign']),
        ]:
            print(f"    {name:<20}  {float(raw):>12.6f}  {lam:>8.2f}  {lam*float(raw):>14.6f}")
        print(f"    {'TOTAL':<20}  {'':>12}  {'':>8}  {float(L):>14.6f}")

        # --- weight field w stats ---
        print(f"\n  Weight field w (|B_meas| normalized, capped at 5)")
        stat(w, brain_mask.bool(), "w [brain]")


if __name__ == "__main__":
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "VEIN_*.nii.gz")))
    sids = [os.path.basename(f).replace(".nii.gz", "") for f in label_files]
    for sid in sids:
        breakdown(sid)
