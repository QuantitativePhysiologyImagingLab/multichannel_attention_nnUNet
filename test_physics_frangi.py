"""
Test whether physics and Frangi losses discriminate segmentation quality.

Data (all same subject):
  medi_qsm.nii.gz              - MEDI chi map (ppm)
  unwrapped_phase1_bgrm.nii.gz - local field after BGRM (Hz, 7T)
  gt.nii.gz                    - ground truth binary mask
  model_segm.nii.gz            - our model segmentation
  msvf_segm.nii.gz             - MSVF segmentation
  vesselFM_segm.nii.gz         - vesselFM (expected worst)

Expected ranking (lower physics loss = better):
  gt < model < msvf < vesselFM

Usage:
  conda run -n manskelab python test_physics_frangi.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib

from nnunetv2.training.loss.vein_susceptometry_loss import PhysicsFieldLoss
from nnunetv2.training.loss.frangi_loss import frangi_3d

TEST_DIR   = "/Users/yousifalkhoury/Downloads/testfrangiv3/test"
B0_DIR     = torch.tensor([0.0, 0.0, 1.0])
VOXEL_SIZE = (0.6, 0.6, 0.6)
HZ_TO_PPM  = 1.0 / (42.5774 * 7.0)   # 7T

physics_fn = PhysicsFieldLoss(vein_channel=1, chi_blood_ppm=0.1, debug=False,
                               default_voxel_size=VOXEL_SIZE)

SEGMENTATIONS = {
    "gt":        "gt.nii.gz",
    "model":     "model_segm.nii.gz",
    "msvf":      "msvf_segm.nii.gz",
    "vesselFM":  "vesselFM_segm.nii.gz",
}


def load_vol(fname):
    return nib.load(os.path.join(TEST_DIR, fname)).get_fdata(dtype=np.float32)


def to_tensor(arr):
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,X,Y,Z)


def segm_to_logits(mask):
    """Binary mask (0/1) → 2-class logits with hard assignment."""
    vein = torch.from_numpy(mask).float().unsqueeze(0)          # (1,X,Y,Z)
    bg   = 1.0 - vein
    return torch.stack([bg * 10 - 5, vein * 10 - 5], dim=1)    # (1,2,X,Y,Z)


def dice(pred_mask, gt_mask):
    inter = (pred_mask * gt_mask).sum()
    return (2 * inter / (pred_mask.sum() + gt_mask.sum() + 1e-6)).item()


def physics_loss_both_formulations(vein_p, chi_qsm, B_meas, brain_mask, gt_vein_mask, chi_b, b0):
    """Compute physics loss with OLD and NEW chi_total formulations."""
    chi_b_t = chi_b.view(1, 1, 1, 1, 1)

    results = {}
    for name, chi_total in [
        ("OLD (blend)",   (1.0 - vein_p) * chi_qsm + vein_p * chi_b_t),
        ("NEW (zero-bg)", chi_qsm * (1.0 - gt_vein_mask) + vein_p * chi_b_t),
    ]:
        B_pred = physics_fn._dipole_field_from_chi(chi_total, VOXEL_SIZE, b0)

        w = B_meas.abs()
        w = (w / (w.mean() + 1e-8)).clamp(max=5.0)
        num = (w * (B_pred - B_meas).abs() * brain_mask).sum()
        den = (w * brain_mask).sum().clamp_min(1.0)
        loss_phys = (num / den).item()

        # Pearson correlation inside brain (unit-independent check)
        bp = B_pred[brain_mask > 0].float()
        bm = B_meas[brain_mask > 0].float()
        bp_z = (bp - bp.mean()) / (bp.std() + 1e-8)
        bm_z = (bm - bm.mean()) / (bm.std() + 1e-8)
        corr = (bp_z * bm_z).mean().item()

        results[name] = {"loss": loss_phys, "corr": corr}
    return results


def frangi_loss_for_segm(vein_p, V_I, brain_mask):
    """Completion hinge: penalise vein_p < V_I in high-Frangi regions (top 10%)."""
    with torch.no_grad():
        flat = V_I[brain_mask > 0]
        if flat.numel() > 500_000:
            flat = flat[torch.randperm(flat.numel())[:500_000]]
        v99 = torch.quantile(flat, 0.99).clamp_min(1e-6)
        V_norm = (V_I / v99).clamp(0, 1)

        alpha, tau = 6.0, 1e-5
        V_gate = torch.sigmoid(alpha * (V_norm - tau))
        V_gate = F.max_pool3d(V_gate, kernel_size=3, stride=1, padding=1)

    valid = (V_norm > 0.9) & (brain_mask > 0)
    if valid.any():
        vi_w  = V_norm[valid].detach()
        hinge = (vi_w * F.relu(V_gate[valid].detach() - vein_p[valid])).mean().item()
    else:
        hinge = float("nan")
    return hinge


def main():
    print("Loading data...")
    chi_qsm   = to_tensor(load_vol("medi_qsm.nii.gz")).float()
    B_meas_hz = to_tensor(load_vol("unwrapped_phase1_bgrm.nii.nii.gz")).float()
    B_meas    = B_meas_hz * HZ_TO_PPM
    gt_mask   = to_tensor(load_vol("gt.nii.gz")).float()

    brain_mask    = (chi_qsm != 0).float()
    gt_vein_mask  = (gt_mask > 0.5).float()
    b0 = B0_DIR.view(1, 3).float()

    # chi_b from GT: top-80% of GT vein chi values
    vein_vals = chi_qsm[gt_vein_mask.bool()].float()
    p5, p95   = torch.quantile(vein_vals, torch.tensor([0.05, 0.95]))
    vein_vals = vein_vals.clamp(p5, p95)
    p20       = torch.quantile(vein_vals, 0.20)
    chi_b     = vein_vals[vein_vals >= p20].mean()
    print(f"chi_b = {chi_b:.4f} ppm  (from GT vein, {int(gt_vein_mask.sum())} voxels)")
    print(f"B_meas range after Hz→ppm: [{B_meas[brain_mask>0].min():.4f}, {B_meas[brain_mask>0].max():.4f}]")

    # Compute Frangi map from MEDI QSM (once, shared across all segmentations)
    print("\nComputing Frangi map from MEDI QSM (this may take ~1-2 min)...")
    # z-score QSM inside brain before Frangi
    bv = chi_qsm[brain_mask > 0]
    chi_z = (chi_qsm - bv.mean()) / (bv.std() + 1e-8)
    chi_z = chi_z * brain_mask
    with torch.no_grad():
        V_I, _ = frangi_3d(chi_z, sigmas=(0.6, 1.2, 1.8), alpha=0.5, beta=0.5,
                            c=15.0, bright_vessels=True)
    print(f"Frangi map: max={V_I.max():.4f}  nonzero={int((V_I>0).sum())}")

    # Header
    print("\n" + "="*90)
    print(f"{'Segm':<12}  {'Dice vs GT':>10}  "
          f"{'Phys OLD':>10}  {'Corr OLD':>9}  "
          f"{'Phys NEW':>10}  {'Corr NEW':>9}  "
          f"{'Frangi':>9}")
    print("-"*90)

    with torch.no_grad():
        for name, fname in SEGMENTATIONS.items():
            seg_mask = to_tensor(load_vol(fname)).float()
            vein_p   = (seg_mask > 0.5).float()

            dc = dice(vein_p.squeeze(), gt_vein_mask.squeeze())
            phys = physics_loss_both_formulations(
                vein_p, chi_qsm, B_meas, brain_mask, gt_vein_mask, chi_b, b0)
            fr = frangi_loss_for_segm(vein_p, V_I, brain_mask)

            print(f"{name:<12}  {dc:>10.4f}  "
                  f"{phys['OLD (blend)']['loss']:>10.5f}  {phys['OLD (blend)']['corr']:>9.4f}  "
                  f"{phys['NEW (zero-bg)']['loss']:>10.5f}  {phys['NEW (zero-bg)']['corr']:>9.4f}  "
                  f"{fr:>9.5f}")

    print("="*90)
    print("\nExpected: Dice and Corr should rank gt > model > msvf > vesselFM")
    print("          Phys and Frangi losses should rank gt < model < msvf < vesselFM")


if __name__ == "__main__":
    main()
