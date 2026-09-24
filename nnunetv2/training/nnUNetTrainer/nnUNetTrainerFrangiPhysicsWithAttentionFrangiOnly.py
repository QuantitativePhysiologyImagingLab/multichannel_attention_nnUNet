from nnunetv2.training.nnUNetTrainer.nnUNetTrainerFrangiPhysicsWithAttention import (
    nnUNetTrainerFrangiPhysicsWithAttention,
)


class nnUNetTrainerFrangiPhysicsWithAttentionFrangiOnly(nnUNetTrainerFrangiPhysicsWithAttention):
    """
    Ablation: Frangi loss only, physics disabled (weight_physics=0).

    Inherits everything else unchanged from nnUNetTrainerFrangiPhysicsWithAttention
    -- network, data pipeline, R2*/single-channel handling, NaN guard, deep
    supervision, checkpointing -- so this run is directly comparable to the
    combined run and automatically picks up any future fix to the shared
    trainer. Setting weight_physics=0 also makes compound_losses.py skip the
    dipole-field FFT computation entirely rather than just zero-weighting it,
    so this is also cheaper per step, not just an ablation.

    Run with: -tr nnUNetTrainerFrangiPhysicsWithAttentionFrangiOnly
    """
    WEIGHT_PHYSICS = 0.0
    NUM_EPOCHS = 500
