import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerFrangiPhysicsWithAttention import (
    nnUNetTrainerFrangiPhysicsWithAttention,
)
from nnunetv2.training.network_architecture.unet_with_attention import vein_to_domain_idx
from nnunetv2.training.loss.channel_layout import R2STAR_DOMAIN_IDX
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


class nnUNetTrainerFrangiPhysicsWithAttentionR2starFinetune(nnUNetTrainerFrangiPhysicsWithAttention):
    """
    R2*-only fine-tuning stage, meant to be warm-started from the combined
    model's checkpoint_best.pth (via nnU-Net's stock -pretrained_weights
    flag -- do NOT combine with --c, same restriction as every other trainer
    in this file's family: run_training.py refuses to do both at once).

    Rationale: R2* gets its own dedicated capacity via `_R2starAdapter`, a
    small residual conv block spliced in right after enc1 (see
    `PriorGatedSingleChannelUNet` in nnUNetTrainerFrangiPhysicsWithAttention.py),
    gated per-sample by domain_idx so it structurally contributes exactly
    zero to non-R2* samples and is zero-init so it starts as a no-op. Rather
    than diluting that new capacity by also fine-tuning the shared trunk
    (which was shaped by, and is relied on by, the QSM-dominated majority of
    the training set), FREEZE_SHARED_TRUNK defaults to True here: every
    parameter except r2star_adapter.* is frozen, so this stage does nothing
    but train the new adapter block from its zero-init starting point.
    do_split() below additionally restricts BOTH train and validation cases
    to R2* only, so every gradient step is R2*-driven.

    FINETUNE_LR (default 1e-3, vs. the base trainer's 1e-2) applies
    regardless of FREEZE_SHARED_TRUNK. Set FREEZE_SHARED_TRUNK = False on a
    subclass or instance to instead fine-tune every weight (a full,
    standard transfer-learning recipe) if the frozen-trunk adapter-only
    result turns out to be insufficient.

    Run with: -tr nnUNetTrainerFrangiPhysicsWithAttentionR2starFinetune
              -pretrained_weights .../checkpoint_best.pth
    """
    NUM_EPOCHS = 500
    FINETUNE_LR = 1e-3
    FREEZE_SHARED_TRUNK = True

    # Name prefixes left trainable when FREEZE_SHARED_TRUNK is on: just the
    # new R2*-dedicated adapter block. Everything else -- including
    # domain_embed/field_embed, pos_mlp, prior_gate, enc1, and final -- stays
    # frozen at its warm-started (combined-model) values.
    UNFROZEN_PREFIXES = ('r2star_adapter',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = self.FINETUNE_LR

    def do_split(self):
        tr_keys, val_keys = super().do_split()
        tr_keys = [k for k in tr_keys if vein_to_domain_idx(k) == R2STAR_DOMAIN_IDX]
        val_keys = [k for k in val_keys if vein_to_domain_idx(k) == R2STAR_DOMAIN_IDX]
        if len(tr_keys) == 0:
            raise RuntimeError(
                'R2* fine-tune: no R2*-domain cases found in the training split -- '
                'check vein_to_domain_idx()/VEIN_TO_DOMAIN against this dataset\'s case IDs.'
            )
        self.print_to_log_file(
            f'R2* fine-tune: restricted split to {len(tr_keys)} train / {len(val_keys)} val R2* case(s).'
        )
        return tr_keys, val_keys

    def configure_optimizers(self):
        if not self.FREEZE_SHARED_TRUNK:
            return super().configure_optimizers()

        net = self.network
        for attr in ('module', '_orig_mod'):
            if hasattr(net, attr):
                net = getattr(net, attr)

        n_trainable, n_total = 0, 0
        for name, p in net.named_parameters():
            n_total += p.numel()
            if any(name.startswith(pref) for pref in self.UNFROZEN_PREFIXES):
                p.requires_grad = True
                n_trainable += p.numel()
            else:
                p.requires_grad = False
        self.print_to_log_file(
            f'R2* fine-tune (frozen shared trunk): {n_trainable}/{n_total} params trainable.'
        )

        trainable_params = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
