# nnunetv2/training/nnUNetTrainer/nnUNetTrainerWithAttention.py
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.network_architecture.unet_with_attention import UNetWithAttention as _BaseUNetWithAttention

# Subclass so state_dict keys stay IDENTICAL (no "net." prefix),
# but forward() returns only the main logits tensor.
class UNetWithAttentionInfer(_BaseUNetWithAttention):
    def forward(self, x):
        y = super().forward(x)
        return y[0] if isinstance(y, (list, tuple)) else y

class nnUNetTrainerWithAttention_icu(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    ) -> nn.Module:
        # get patch size robustly
        if hasattr(configuration_manager, "patch_size"):
            patch_size = tuple(configuration_manager.patch_size)
        elif isinstance(configuration_manager, (list, tuple)):
            patch_size = tuple(configuration_manager)
        elif hasattr(plans_manager, "configuration_manager") and hasattr(plans_manager.configuration_manager, "patch_size"):
            patch_size = tuple(plans_manager.configuration_manager.patch_size)
        else:
            raise RuntimeError("Cannot determine patch_size")

        # Build WITH DS layers so checkpoint keys (ds1/ds2/ds3) match
        net = UNetWithAttentionInfer(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            patch_size=patch_size,
            deep_supervision=True,   # keep DS heads present for weight loading
        )
        # Optionally respect flag at runtime (won’t affect keys)
        for attr in ("do_ds", "deep_supervision", "enable_deep_supervision"):
            if hasattr(net, attr):
                setattr(net, attr, bool(enable_deep_supervision))
        return net