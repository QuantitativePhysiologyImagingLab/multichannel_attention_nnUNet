from nnunetv2.training.nnUNetTrainer.nnUNetTrainerFrangiPhysicsWithAttention import (
    nnUNetTrainerFrangiPhysicsWithAttention,
)


class nnUNetTrainerFrangiPhysicsWithAttentionPhysicsOnly(nnUNetTrainerFrangiPhysicsWithAttention):
    """
    Ablation: physics loss only, Frangi disabled (weight_frangi=0).

    Inherits everything else unchanged from nnUNetTrainerFrangiPhysicsWithAttention
    -- network, data pipeline, R2*/single-channel handling, NaN guard, deep
    supervision, checkpointing -- so this run is directly comparable to the
    combined run and automatically picks up any future fix to the shared
    trainer. Setting weight_frangi=0 also makes compound_losses.py skip the
    Frangi computation (including the eigendecomposition) entirely rather
    than just zero-weighting it, so this is also cheaper per step, not just
    an ablation.

    Run with: -tr nnUNetTrainerFrangiPhysicsWithAttentionPhysicsOnly
    """
    WEIGHT_FRANGI = 0.0
    NUM_EPOCHS = 500
