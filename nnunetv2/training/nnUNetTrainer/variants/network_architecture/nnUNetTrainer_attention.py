import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Act, Norm

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, dimensions, activation_function):
        super().__init__()
        self.W_g = Convolution(F_g, F_int, dimensions, 1, 1, 0, norm=None, act=activation_function)
        self.W_x = Convolution(F_l, F_int, dimensions, 1, 1, 0, norm=None, act=activation_function)
        self.psi = Convolution(F_int, 1, dimensions, 1, 1, 0, norm=None, act=("SIGMOID", {}))
        self.activation = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(self.psi(g1 + x1))
        return x * psi


class UNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, deep_supervision=False, base_features=32, dimensions=3):
        super().__init__()
        self.deep_supervision = deep_supervision
        act = ("ELU", {"alpha": 1.0})

        # Encoder
        self.enc1 = Convolution(in_channels, base_features, dimensions, 3, 1, 1, norm=None, act=act)
        self.enc2 = Convolution(base_features, base_features*2, dimensions, 3, 2, 1, norm=Norm.INSTANCE, act=act)
        self.enc3 = Convolution(base_features*2, base_features*4, dimensions, 3, 2, 1, norm=Norm.INSTANCE, act=act)
        self.enc4 = Convolution(base_features*4, base_features*8, dimensions, 3, 2, 1, norm=Norm.INSTANCE, act=act)

        # Bottleneck
        self.bottleneck = Convolution(base_features*8, base_features*16, dimensions, 3, 1, 1, norm=Norm.INSTANCE, act=act)

        # Decoder + Attention
        self.up4 = Convolution(base_features*16, base_features*8, dimensions, 2, 2, 0, is_transposed=True, norm=Norm.INSTANCE, act=act)
        self.att4 = AttentionGate(base_features*8, base_features*4, base_features*4, dimensions, act)

        self.up3 = Convolution(base_features*12, base_features*4, dimensions, 2, 2, 0, is_transposed=True, norm=Norm.INSTANCE, act=act)
        self.att3 = AttentionGate(base_features*4, base_features*2, base_features*2, dimensions, act)

        self.up2 = Convolution(base_features*6, base_features*2, dimensions, 2, 2, 0, is_transposed=True, norm=Norm.INSTANCE, act=act)
        self.att2 = AttentionGate(base_features*2, base_features, base_features, dimensions, act)

        self.up1 = Convolution(base_features*3, base_features, dimensions, 2, 2, 0, is_transposed=True, norm=Norm.INSTANCE, act=act)

        # Final layer
        self.final = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        b = self.bottleneck(x4)

        u4 = self.up4(b)
        a4 = self.att4(u4, x3)
        u4 = torch.cat([u4, a4], dim=1)

        u3 = self.up3(u4)
        a3 = self.att3(u3, x2)
        u3 = torch.cat([u3, a3], dim=1)

        u2 = self.up2(u3)
        a2 = self.att2(u2, x1)
        u2 = torch.cat([u2, a2], dim=1)

        u1 = self.up1(u2)
        out = self.final(u1)

        return out if not self.deep_supervision else [out]