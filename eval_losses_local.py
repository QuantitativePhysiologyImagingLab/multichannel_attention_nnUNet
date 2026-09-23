"""
Evaluate each loss component on the local dataset without a trained network.
A GT-derived soft prediction is used as net_output so loss magnitudes are meaningful.

Usage:
    python eval_losses_local.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import glob
import torch
import numpy as np
import nibabel as nib

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.tversky_loss import FocalTverskyLoss
from nnunetv2.training.loss.frangi_loss import FrangiLoss
from nnunetv2.training.loss.vein_susceptometry_loss import PhysicsFieldLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

IMAGES_DIR    = "/Users/yousifalkhoury/Downloads/medi_vein_seg/imagesTr"
LABELS_DIR    = "/Users/yousifalkhoury/Downloads/medi_vein_seg/labels"
B0_DIR        = torch.tensor([0.0, 0.0, 1.0])
PHASE_PATH_7T = "/Users/yousifalkhoury/Downloads/medi_vein_seg/unwrapped_phase1_bgrm.nii.nii.gz"
PHASE_PATH_3T = "/Users/yousifalkhoury/Downloads/medi_vein_seg/labels/unwrapped_phase1_bgrm.nii.nii.gz"

HZ_TO_PPM_7T = 1.0 / (42.5774 * 7.0)   # 1/298.04
HZ_TO_PPM_3T = 1.0 / (42.5774 * 3.0)   # 1/127.73

# Maps subject_id -> (bgrm_path, Hz-to-ppm factor).
# TGV subjects are absent (they use their own 0001 channel).
USE_PHASE_FOR = {
    # 7T subjects
    "VEIN_122": (PHASE_PATH_7T, HZ_TO_PPM_7T),
    "VEIN_123": (PHASE_PATH_7T, HZ_TO_PPM_7T),
    "VEIN_124": (PHASE_PATH_7T, HZ_TO_PPM_7T),
    "VEIN_125": (PHASE_PATH_7T, HZ_TO_PPM_7T),
    # 3T MEDI subject
    "VEIN_030": (PHASE_PATH_3T, HZ_TO_PPM_3T),
}

# --- build individual loss modules ---
dc_loss_fn      = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1,
                                               batch_dice=False, do_bg=True, smooth=1e-5)
ce_loss_fn      = RobustCrossEntropyLoss()
tversky_loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
physics_loss_fn = PhysicsFieldLoss(vein_channel=1, chi_blood_ppm=0.1, debug=False)
frangi_loss_fn  = FrangiLoss()


def load_subject(subject_id: str):
    """Load (data, label) tensors for one subject. Returns (1,3,D,H,W), (1,1,D,H,W)."""
    ch0 = nib.load(os.path.join(IMAGES_DIR, f"{subject_id}_0000.nii.gz")).get_fdata(dtype=np.float32)
    ch2 = nib.load(os.path.join(IMAGES_DIR, f"{subject_id}_0002.nii.gz")).get_fdata(dtype=np.float32)
    if subject_id in USE_PHASE_FOR:
        bgrm_path, hz_to_ppm = USE_PHASE_FOR[subject_id]
        ch1 = nib.load(bgrm_path).get_fdata(dtype=np.float32) * hz_to_ppm
    else:
        ch1 = nib.load(os.path.join(IMAGES_DIR, f"{subject_id}_0001.nii.gz")).get_fdata(dtype=np.float32)
    data = torch.from_numpy(np.stack([ch0, ch1, ch2], axis=0)).unsqueeze(0)  # (1, 3, D, H, W)

    lbl_path = os.path.join(LABELS_DIR, f"{subject_id}.nii.gz")
    lbl  = nib.load(lbl_path).get_fdata(dtype=np.float32)
    lbl  = torch.from_numpy(lbl).long().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    return data, lbl


def make_net_output(label: torch.Tensor, noise: float = 0.3) -> torch.Tensor:
    """
    Create 2-class logits from GT label so loss magnitudes are realistic.
    logit[1] is high where vein, logit[0] is high otherwise.
    """
    B, _, D, H, W = label.shape
    vein = (label[:, 0] == 1).float()  # (B,D,H,W)
    bg   = 1.0 - vein
    logits = torch.stack([2.0 * bg - 1.0, 2.0 * vein - 1.0], dim=1).float()  # (B,2,D,H,W)
    logits = logits + noise * torch.randn_like(logits)
    return logits


def eval_subject(subject_id: str):
    print(f"\n{'='*60}")
    print(f"Subject: {subject_id}")
    print(f"{'='*60}")

    data, label = load_subject(subject_id)
    print(f"  data shape : {tuple(data.shape)}  dtype: {data.dtype}")
    print(f"  label shape: {tuple(label.shape)}  unique: {label.unique().tolist()}")

    net_output = make_net_output(label)
    print(f"  net_output : {tuple(net_output.shape)}")

    b0 = B0_DIR.view(1, 3).float()

    with torch.no_grad():
        # --- Dice ---
        dc = dc_loss_fn(net_output, label.float())
        print(f"\n  [Dice]    {float(dc):.6f}")

        # --- CE ---
        ce = ce_loss_fn(net_output, label[:, 0])
        print(f"  [CE]      {float(ce):.6f}")

        # --- Tversky ---
        tv = tversky_loss_fn(net_output, label.float())
        print(f"  [Tversky] {float(tv):.6f}")

        # --- Physics ---
        phys, metrics = physics_loss_fn(
            net_output=net_output,
            data=data,
            b0_dir=b0,
            target=label.float(),
        )
        print(f"  [Physics] {float(phys):.6f}")
        for k, v in metrics.items():
            print(f"            {k}: {float(v):.6f}")

        # --- Frangi ---
        frangi = frangi_loss_fn(net_output=net_output.float(), data=data)
        print(f"  [Frangi]  {float(frangi):.6f}")

        # --- Weighted total (same weights as trainer) ---
        total = (1.0 * ce + 0.5 * dc + 1.0 * tv
                 + 20.0 * phys + 1.0 * frangi)
        print(f"\n  [Total (weighted)] {float(total):.6f}")


if __name__ == "__main__":
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "VEIN_*.nii.gz")))
    subject_ids = [os.path.basename(f).replace(".nii.gz", "") for f in label_files]
    print(f"Found {len(subject_ids)} subjects: {subject_ids}")

    for sid in subject_ids:
        eval_subject(sid)
