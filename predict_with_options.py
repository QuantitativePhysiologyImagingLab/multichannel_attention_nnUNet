"""
Inference wrapper that sets the QSM reconstruction method for domain conditioning.

Usage:
    python predict_with_method.py \
        -i /path/to/imagesTr \
        -o /path/to/output \
        -d Dataset001_vein \
        -c 3d_fullres \
        -f all \
        --method tgv          # one of: tgv, medi, l1, star, ilsqr

The --method flag tells the FiLM layers which domain embedding to use.
If omitted, defaults to 'tgv'.
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.network_architecture.unet_with_attention import (
    METHOD_TO_IDX, DOMAIN_METHODS, FIELD_STRENGTHS
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',       required=True,  help='Input folder (imagesTr or imagesTs)')
    parser.add_argument('-o',       required=True,  help='Output folder')
    parser.add_argument('-d',       required=True,  help='Dataset name or ID, e.g. Dataset001_vein')
    parser.add_argument('-c',       default='3d_fullres', help='Configuration (default: 3d_fullres)')
    parser.add_argument('-f',       default='all',  help='Fold (default: all)')
    parser.add_argument('-tr',      default='nnUNetTrainerFrangiPhysicsWithAttention',
                        help='Trainer class name')
    parser.add_argument('-p',       default='nnUNetPlans', help='Plans identifier')
    parser.add_argument('--method', default='tgv',
                        choices=[m.lower() for m in DOMAIN_METHODS],
                        help='QSM reconstruction method used for this dataset')
    parser.add_argument('--field', default='7t', choices=['7t', '3t'],
                        help='MRI field strength (default: 7t)')
    parser.add_argument('--step_size',     type=float, default=0.5)
    parser.add_argument('--save_probs',    action='store_true')
    parser.add_argument('--disable_tta',   action='store_true')
    args = parser.parse_args()

    domain_idx = METHOD_TO_IDX[args.method.lower() if args.method.lower() in METHOD_TO_IDX
                                else args.method]
    field_idx  = 1 if args.field.lower() == '3t' else 0
    print(f"[predict_with_method] method='{args.method}' (domain={domain_idx})  "
          f"field='{args.field}' (field_idx={field_idx})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=_find_model_folder(args.d, args.tr, args.p),
        use_folds=(args.f,),
        checkpoint_name='checkpoint_best.pth',
    )

    # inject domain + field indices into the network before prediction
    _set_domain(predictor.network, domain_idx, field_idx)

    predictor.predict_from_raw_data(
        list_of_lists_or_source_folder=args.i,
        output_folder=args.o,
        save_probabilities=args.save_probs,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )


def _set_domain(network, domain_idx: int, field_idx: int = 0):
    """Recursively set default_domain_idx and default_field_idx on the network."""
    target = network
    for attr in ('module', '_orig_mod'):
        if hasattr(target, attr):
            target = getattr(target, attr)
    target.default_domain_idx = domain_idx
    target.default_field_idx  = field_idx
    print(f"[predict_with_method] domain_idx={domain_idx}  field_idx={field_idx}", flush=True)


def _find_model_folder(dataset: str, trainer: str, plans: str) -> str:
    from nnunetv2.paths import nnUNet_results
    import glob
    pattern = os.path.join(nnUNet_results, f'*{dataset}*', f'{trainer}__{plans}__3d_fullres')
    matches = glob.glob(pattern)
    if not matches:
        pattern2 = os.path.join(nnUNet_results, dataset, f'{trainer}__{plans}__3d_fullres')
        matches = glob.glob(pattern2)
    if not matches:
        raise FileNotFoundError(
            f"Could not find model folder for dataset={dataset}, trainer={trainer}, plans={plans}.\n"
            f"Searched: {pattern}\nSet $nnUNet_results or pass the full path manually."
        )
    return matches[0]


if __name__ == '__main__':
    main()
