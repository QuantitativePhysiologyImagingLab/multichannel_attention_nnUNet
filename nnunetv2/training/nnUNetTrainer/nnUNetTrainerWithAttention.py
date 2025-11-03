# trainer file: nnunetv2/training/nnUNetTrainer/nnUNetTrainerWithAttention.py
import numpy as np
import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.network_architecture.unet_with_attention import \
    UNetWithAttention as _BaseUNetWithAttention


# --- Prior gate ---
class PriorGate(nn.Module):
    def __init__(self, in_priors=2):
        super().__init__()
        self.prior_to_gate = nn.Sequential(
            nn.Conv3d(in_priors, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, priors):
        return self.gamma * self.prior_to_gate(priors)  # (B,1,*,*,*)


# --- Base net + input prior gating (keeps original module names!) ---
class PriorGatedUNetWithAttention(_BaseUNetWithAttention):
    def __init__(self, in_channels, out_channels, patch_size, deep_supervision=True):
        super().__init__(in_channels, out_channels, patch_size, deep_supervision)
        n_priors = max(0, in_channels - 1)
        assert n_priors >= 1, "Expected at least one prior channel (C >= 2)."
        self.prior_gate = PriorGate(in_priors=n_priors)

    def forward(self, x):
        img = x[:, 0:1]
        pri = x[:, 1:]                # <— all prior channels, 1..C-1
        A = self.prior_gate(pri)      # (B,1,*,*,*)
        x_mod = torch.cat([img * (1 + A), x[:, 1:]], dim=1)
        return super().forward(x_mod)


# --- Inference shim: return tuple in training, main tensor in eval ---
class PriorGatedUNetWithAttentionInfer(PriorGatedUNetWithAttention):
    def forward(self, x):
        y = super().forward(x)
        # Decide purely by DS flags, not by self.training
        use_ds = bool(getattr(self, "do_ds", False)
                      or getattr(self, "deep_supervision", False)
                      or getattr(self, "enable_deep_supervision", False))
        if isinstance(y, (list, tuple)):
            return y if use_ds else y[0]
        else:
            # Base net returned a single tensor; wrap it if DS expected
            return [y] if use_ds else y


# --- Small transform to drop prior channels sometimes ---
class RandomChannelDropout:
    """
    Works with both nnU-Net v2 transform APIs:
      - {'image': np.ndarray, 'segmentation': np.ndarray}
      - {'data':  np.ndarray, 'seg':           np.ndarray}
    Randomly zeroes prior channels (1,2).
    """
    def __init__(self, p_each=0.4, channels_to_consider=(1, 2)):
        self.p_each = float(p_each)
        self.channels_to_consider = tuple(channels_to_consider)

    def __call__(self, **data_dict):
        # detect which key nnU-Net used
        if 'data' in data_dict:
            k = 'data'
        elif 'image' in data_dict:
            k = 'image'
        else:
            # if neither key present, just return unchanged
            return data_dict

        x = data_dict[k]  # np.ndarray (C, ...)
        C = x.shape[0] if x.ndim >= 1 else 0
        for c in self.channels_to_consider:
            if c < C and np.random.rand() < self.p_each:
                x[c] = 0
        data_dict[k] = x
        return data_dict


class _PrependTransform:
    def __init__(self, first, then): self.first, self.then = first, then
    def __call__(self, **d): return self.then(**self.first(**d))


class nnUNetTrainerWithAttention(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    ) -> nn.Module:
        # ---- robust patch_size retrieval across v2 variants ----
        patch_size = None
        # 1) object with attribute
        if hasattr(configuration_manager, "patch_size"):
            patch_size = tuple(configuration_manager.patch_size)
        # 2) dict-like
        elif isinstance(configuration_manager, dict):
            if "patch_size" in configuration_manager and configuration_manager["patch_size"] is not None:
                patch_size = tuple(configuration_manager["patch_size"])
        # 3) list/tuple passed directly
        elif isinstance(configuration_manager, (list, tuple)):
            patch_size = tuple(configuration_manager)
        # 4) fallback: try plans (some forks stash it there)
        if patch_size is None:
            try:
                # if your trainer has self.configuration name, you could use it here;
                # since we're in @staticmethod, grab the first config that has patch_size
                for cfg_name, cfg in plans_manager.plans.get("configurations", {}).items():
                    if "patch_size" in cfg:
                        patch_size = tuple(cfg["patch_size"])
                        break
            except Exception:
                pass
        if patch_size is None:
            raise RuntimeError("Cannot determine patch_size from configuration_manager/plans.")

        net = PriorGatedUNetWithAttentionInfer(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            patch_size=patch_size,
            deep_supervision=True,
        )
        for attr in ("do_ds", "deep_supervision", "enable_deep_supervision"):
            if hasattr(net, attr):
                setattr(net, attr, bool(enable_deep_supervision))
        return net

    
    def get_training_transforms(self, *args, **kwargs):
        # forward all v2 args to the base implementation
        base = super().get_training_transforms(*args, **kwargs)

        rcd = RandomChannelDropout(p_each=0)

        # If it's a Compose-like with a .transforms list, insert; otherwise wrap.
        if hasattr(base, "transforms") and isinstance(base.transforms, list):
            # base.transforms.insert(0, rcd)
            return base

        class _Prepend:
            def __init__(self, first, then): self.first, self.then = first, then
            def __call__(self, **d): return self.then(**self.first(**d))

        return _Prepend(rcd, base)