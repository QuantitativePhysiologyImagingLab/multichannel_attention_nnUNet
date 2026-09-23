"""
Converts an old 3-channel/5-domain nnUNetTrainerFrangiPhysicsWithAttention
checkpoint (PriorGatedUNetWithAttentionInfer) into a warm-start checkpoint
for the new single-channel/6-domain, R2*-aware architecture
(PriorGatedSingleChannelUNet), for use with nnU-Net's standard
`-pretrained_weights` flag.

Verified empirically (see conversation/commit history) against both
architectures' state_dicts:
  - 55 keys (enc2/enc3/enc4/bottleneck/decoder/prior_gate.gamma/
    field_embed/film_*) are already identical in shape between the two
    architectures and need no change here.
  - `domain_embed.weight` differs in shape ((5,32) -> (6,32), for the new
    R2star domain) but nnU-Net's own `load_pretrained_weights`
    (nnunetv2/run/load_pretrained_weights.py) unconditionally skips
    domain_embed/field_embed/film_enc*/film_bottle regardless of what's in
    this file, so it does not need to be touched here either.
  - `enc1.block.0.weight` differs in input-channel count: old (64,3,4,4,4)
    (input was [gated_img, local_field, frangi]) vs new (64,1,4,4,4) (input
    is just the gated image). We keep only the channel-0 slice, since that's
    exactly the sub-filter that operated on the gated image in the old
    architecture too.
  - `prior_gate.prior_to_gate.0.{weight,bias}` (old) and
    `prior_gate.to_gate.{weight,bias}` (new) are the same shape (n_priors=2
    in both) under a different name — renamed here.

Usage:
    python convert_checkpoint_for_r2star.py \\
        --old /path/to/checkpoint_best.pth \\
        --out /path/to/warm_start_for_r2star.pth

Then train with:
    nnUNetv2_train <dataset> 3d_fullres <fold> \\
        -tr nnUNetTrainerFrangiPhysicsWithAttention \\
        -pretrained_weights /path/to/warm_start_for_r2star.pth
(do NOT also pass -c/--continue_training -- these are incompatible together,
and this is a fresh-start-with-warm-weights run, not a resume)
"""
import argparse
import torch

RENAME_MAP = {
    'prior_gate.prior_to_gate.0.weight': 'prior_gate.to_gate.weight',
    'prior_gate.prior_to_gate.0.bias': 'prior_gate.to_gate.bias',
}
SLICE_TO_CHANNEL0 = ('enc1.block.0.weight',)


def _strip_prefixes(key: str) -> str:
    # nnUNetTrainer.save_checkpoint already strips DDP/compile prefixes before
    # saving, but strip defensively in case this checkpoint came from
    # somewhere else.
    for prefix in ('module.', '_orig_mod.'):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def convert(old_sd: dict) -> dict:
    new_sd = {}
    renamed, sliced, copied = [], [], []

    for raw_key, v in old_sd.items():
        key = _strip_prefixes(raw_key)
        new_key = RENAME_MAP.get(key, key)
        if new_key != key:
            renamed.append(f'{key} -> {new_key}')

        if new_key in SLICE_TO_CHANNEL0:
            old_shape = tuple(v.shape)
            new_sd[new_key] = v[:, 0:1].clone()
            sliced.append(f'{new_key}: {old_shape} -> {tuple(new_sd[new_key].shape)}')
        else:
            new_sd[new_key] = v.clone()
            copied.append(new_key)

    print(f'[convert] copied {len(copied)} tensors as-is')
    print(f'[convert] renamed {len(renamed)}:')
    for r in renamed:
        print('   ', r)
    print(f'[convert] sliced {len(sliced)}:')
    for s in sliced:
        print('   ', s)
    print(
        '[convert] NOTE: domain_embed / field_embed / film_enc* / film_bottle '
        'are intentionally left as-is here -- nnU-Net\'s load_pretrained_weights '
        'always skips those regardless, and domain_embed changed shape (5->6 '
        'rows for the new R2star domain) so it MUST be reinitialized anyway.'
    )
    return new_sd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old', required=True, help='Path to the old checkpoint (.pth)')
    ap.add_argument('--out', required=True, help='Path to write the converted checkpoint')
    args = ap.parse_args()

    old_ckpt = torch.load(args.old, map_location='cpu', weights_only=False)
    old_sd = old_ckpt['network_weights']
    new_sd = convert(old_sd)
    torch.save({'network_weights': new_sd}, args.out)
    print(f'[convert] wrote {args.out}')


if __name__ == '__main__':
    main()
