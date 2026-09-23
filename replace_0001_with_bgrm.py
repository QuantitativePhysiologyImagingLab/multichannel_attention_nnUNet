"""
Replace VEIN_XXX_0001.nii.gz files with the corresponding romeo background-removed
phase map (unwrapped_phase1_bgrm.nii.nii.gz), converted from Hz to ppm.

  3T subjects (030-053): /home/usif/scratch/QSM_diabetes/{id}/romeo_out/...  @ 3T
  7T subjects (054-149): /home/usif/scratch/7T_vein_seg_training/{id}/romeo_out/... @ 7T

Set DRY_RUN = True to preview without writing anything.
"""

import os
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

DRY_RUN = True   # <-- set False to actually overwrite files

IMAGES_DIR = "/home/usif/scratch/vein_segmentation/frangi_nnUNet_training/nnUNet_raw/Dataset001_vein/imagesTr"

HZ_TO_PPM_3T = 1.0 / (42.5774 * 3.0)   # 1/127.73
HZ_TO_PPM_7T = 1.0 / (42.5774 * 7.0)   # 1/298.04

# ---------- subject mapping (vein_id -> (subject_dir, hz_to_ppm, base_path)) ----------

_3T_BASE = "/home/usif/scratch/QSM_diabetes"
_7T_BASE = "/home/usif/scratch/7T_vein_seg_training"

SUBJECTS = {
    # 3T
    "VEIN_030": ("3TA3534_007_V2", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_031": ("3TA3534_007_V2", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_032": ("3TA3534_007_V2", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_033": ("3TA3534_007_V2", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_034": ("3TA3545_008_V2", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_035": ("3TA3545_008_V2", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_036": ("3TA3545_008_V2", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_037": ("3TA3545_008_V2", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_038": ("3TA3557_002_V2", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_039": ("3TA3557_002_V2", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_040": ("3TA3557_002_V2", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_041": ("3TA3557_002_V2", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_042": ("3TA2837_003_V1", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_043": ("3TA2837_003_V1", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_044": ("3TA2837_003_V1", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_045": ("3TA2837_003_V1", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_046": ("3TA2313_001_V1", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_047": ("3TA2313_001_V1", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_048": ("3TA2313_001_V1", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_049": ("3TA2313_001_V1", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_050": ("3TA2355_002_V1", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_051": ("3TA2355_002_V1", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_052": ("3TA2355_002_V1", HZ_TO_PPM_3T, _3T_BASE),
    "VEIN_053": ("3TA2355_002_V1", HZ_TO_PPM_3T, _3T_BASE),
    # 7T
    "VEIN_054": ("P05", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_055": ("P05", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_056": ("P05", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_057": ("P05", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_058": ("P06", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_059": ("P06", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_060": ("P06", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_061": ("P06", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_062": ("P07", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_063": ("P07", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_064": ("P07", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_065": ("P07", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_066": ("P08", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_067": ("P08", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_068": ("P08", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_069": ("P08", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_070": ("P09", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_071": ("P09", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_072": ("P09", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_073": ("P09", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_074": ("P17", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_075": ("P17", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_076": ("P17", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_077": ("P17", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_078": ("P18", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_079": ("P18", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_080": ("P18", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_081": ("P18", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_082": ("P20", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_083": ("P20", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_084": ("P20", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_085": ("P20", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_086": ("P21", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_087": ("P21", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_088": ("P21", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_089": ("P21", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_090": ("P22", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_091": ("P22", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_092": ("P22", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_093": ("P22", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_094": ("P23", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_095": ("P23", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_096": ("P23", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_097": ("P23", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_098": ("P24", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_099": ("P24", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_100": ("P24", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_101": ("P24", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_102": ("P25", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_103": ("P25", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_104": ("P25", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_105": ("P25", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_106": ("P26", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_107": ("P26", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_108": ("P26", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_109": ("P26", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_110": ("P27", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_111": ("P27", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_112": ("P27", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_113": ("P27", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_114": ("P29", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_115": ("P29", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_116": ("P29", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_117": ("P29", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_118": ("P30", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_119": ("P30", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_120": ("P30", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_121": ("P30", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_122": ("P31", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_123": ("P31", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_124": ("P31", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_125": ("P31", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_126": ("P33", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_127": ("P33", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_128": ("P33", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_129": ("P33", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_130": ("P34", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_131": ("P34", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_132": ("P34", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_133": ("P34", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_134": ("P35", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_135": ("P35", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_136": ("P35", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_137": ("P35", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_138": ("P36", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_139": ("P36", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_140": ("P36", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_141": ("P36", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_142": ("P37", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_143": ("P37", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_144": ("P37", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_145": ("P37", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_146": ("P38", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_147": ("P38", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_148": ("P38", HZ_TO_PPM_7T, _7T_BASE),
    "VEIN_149": ("P38", HZ_TO_PPM_7T, _7T_BASE),
}


def process(vein_id, subject_dir, hz_to_ppm, base_path):
    bgrm_path = os.path.join(base_path, subject_dir, "romeo_out", "unwrapped_phase1_bgrm.nii.nii.gz")
    out_path   = os.path.join(IMAGES_DIR, f"{vein_id}_0001.nii.gz")
    ref_path   = out_path  # read existing file for header/shape

    # --- check source exists ---
    if not os.path.exists(bgrm_path):
        print(f"  [MISSING]  {bgrm_path}")
        return False

    if not os.path.exists(ref_path):
        print(f"  [MISSING]  {ref_path}")
        return False

    # --- load ---
    bgrm_img = nib.load(bgrm_path)
    ref_img  = nib.load(ref_path)

    bgrm_data = bgrm_img.get_fdata(dtype=np.float32) * hz_to_ppm
    ref_shape = ref_img.shape

    # --- resample if shapes differ ---
    if bgrm_data.shape != ref_shape:
        factors = tuple(r / b for r, b in zip(ref_shape, bgrm_data.shape))
        print(f"  [RESAMPLE] {vein_id}: {bgrm_data.shape} -> {ref_shape}  factors={[f'{f:.3f}' for f in factors]}")
        bgrm_data = zoom(bgrm_data, factors, order=1)

    scale = hz_to_ppm * 1  # already applied above; just for display
    print(f"  {'[DRY]' if DRY_RUN else '[WRITE]'} {vein_id}  src={subject_dir}  "
          f"factor={hz_to_ppm:.5f}  ppm_std={bgrm_data[bgrm_data!=0].std():.4f}")

    if not DRY_RUN:
        new_img = nib.Nifti1Image(bgrm_data, ref_img.affine, ref_img.header)
        nib.save(new_img, out_path)

    return True


if __name__ == "__main__":
    if DRY_RUN:
        print("=== DRY RUN — no files will be written ===\n")

    ok = err = 0
    for vein_id, (subject_dir, hz_to_ppm, base_path) in sorted(SUBJECTS.items()):
        success = process(vein_id, subject_dir, hz_to_ppm, base_path)
        if success:
            ok += 1
        else:
            err += 1

    print(f"\nDone: {ok} ok, {err} missing/errors")
    if DRY_RUN:
        print("\nSet DRY_RUN = False at the top of the script to write files.")
