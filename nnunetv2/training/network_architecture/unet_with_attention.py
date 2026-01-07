import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transposed=False, output_padding=0):
        super().__init__()
        if transposed:
            conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding)
        else:
            conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

        self.block = nn.Sequential(
            conv,
            nn.InstanceNorm3d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        out = self.block(x)
        # print(f"{'Up' if isinstance(self.block[0], nn.ConvTranspose3d) else 'Down'} block output: {out.shape}")
        return out

def _center_crop_or_pad_3d(t: torch.Tensor, target_spatial):
    """
    t: (B,C,D,H,W)
    target_spatial: (Dt,Ht,Wt)
    Returns t center-cropped and/or zero-padded to target_spatial.
    """
    D, H, W = t.shape[-3:]
    Dt, Ht, Wt = target_spatial

    # ---- center crop if too big ----
    sd = max((D - Dt) // 2, 0)
    sh = max((H - Ht) // 2, 0)
    sw = max((W - Wt) // 2, 0)

    ed = sd + min(Dt, D)
    eh = sh + min(Ht, H)
    ew = sw + min(Wt, W)

    t = t[..., sd:ed, sh:eh, sw:ew]

    # ---- pad if too small ----
    D2, H2, W2 = t.shape[-3:]
    pd = max(Dt - D2, 0)
    ph = max(Ht - H2, 0)
    pw = max(Wt - W2, 0)

    # F.pad uses (W_left,W_right,H_left,H_right,D_left,D_right)
    pad = (pw // 2, pw - pw // 2,
           ph // 2, ph - ph // 2,
           pd // 2, pd - pd // 2)
    if any(p > 0 for p in pad):
        t = F.pad(t, pad, mode="constant", value=0.0)

    return t

def _match_spatial(a: torch.Tensor, b: torch.Tensor):
    """
    Make a and b have identical spatial shape by cropping/padding both to the min spatial size.
    """
    Da, Ha, Wa = a.shape[-3:]
    Db, Hb, Wb = b.shape[-3:]
    target = (min(Da, Db), min(Ha, Hb), min(Wa, Wb))
    a2 = _center_crop_or_pad_3d(a, target)
    b2 = _center_crop_or_pad_3d(b, target)
    return a2, b2


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1),
            nn.InstanceNorm3d(F_int),
            nn.ELU(inplace=True)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1),
            nn.InstanceNorm3d(F_int),
            nn.ELU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # ---- FIX: align spatial shapes BEFORE addition ----
        g1, x1 = _match_spatial(g1, x1)

        psi = self.psi(g1 + x1)  # (B,1,*,*,*)

        # ---- FIX: ensure psi matches x spatial size ----
        if psi.shape[-3:] != x.shape[-3:]:
            psi = _center_crop_or_pad_3d(psi, x.shape[-3:])

        return x * psi


class AttentionDecoder(nn.Module):
    def __init__(self, base_features=64, out_channels=1, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.up4 = ConvBlock(base_features * 16, base_features * 8, 4, 2, 1, transposed=True)
        self.att4 = AttentionGate(base_features * 8, base_features * 4, base_features * 4)

        self.up3 = ConvBlock(base_features * 12, base_features * 4, 4, 2, 1, transposed=True)
        self.att3 = AttentionGate(base_features * 4, base_features * 2, base_features * 2)

        self.up2 = ConvBlock(base_features * 6, base_features * 2, 4, 2, 1, transposed=True)
        self.att2 = AttentionGate(base_features * 2, base_features, base_features)

        self.up1 = ConvBlock(base_features * 3, base_features, 4, 2, 1, transposed=True)

        self.final = nn.Conv3d(base_features, out_channels, kernel_size=1)

        if self.deep_supervision:
            self.ds3 = nn.Conv3d(base_features * 6, out_channels, kernel_size=1)  # 256 + 128 = 384 → 6*64
            self.ds2 = nn.Conv3d(base_features * 3, out_channels, kernel_size=1)  # 128 + 64 = 192 → 3*64
            self.ds1 = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def forward(self, b, x4, x3, x2, x1):
        u4 = self.up4(b)
        # align u4 to skip spatial size (x3) before attention + concat
        if u4.shape[-3:] != x3.shape[-3:]:
            u4 = _center_crop_or_pad_3d(u4, x3.shape[-3:])
        a4 = self.att4(u4, x3)
        u4 = torch.cat([u4, a4], dim=1)

        u3 = self.up3(u4)
        if u3.shape[-3:] != x2.shape[-3:]:
            u3 = _center_crop_or_pad_3d(u3, x2.shape[-3:])
        a3 = self.att3(u3, x2)
        u3 = torch.cat([u3, a3], dim=1)

        u2 = self.up2(u3)
        if u2.shape[-3:] != x1.shape[-3:]:
            u2 = _center_crop_or_pad_3d(u2, x1.shape[-3:])
        a2 = self.att2(u2, x1)
        u2 = torch.cat([u2, a2], dim=1)

        u1 = self.up1(u2)
        out = self.final(u1)

        if self.deep_supervision:
            out3 = nn.functional.interpolate(self.ds3(u3), size=[s // 8 for s in out.shape[2:]], mode='trilinear', align_corners=False)
            out2 = nn.functional.interpolate(self.ds2(u2), size=[s // 4 for s in out.shape[2:]], mode='trilinear', align_corners=False)
            out1 = nn.functional.interpolate(self.ds1(u1), size=[s // 2 for s in out.shape[2:]], mode='trilinear', align_corners=False)

            # Only return a list during training (deep supervision)
            if self.training:
                return [out, out1, out2, out3]

        # Return single tensor in validation/inference
        return out

class UNetWithAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, patch_size=(80, 80, 80), deep_supervision=False, base_features=64):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.enc1 = ConvBlock(in_channels, base_features, 4, 2, 1)
        self.enc2 = ConvBlock(base_features, base_features * 2, 4, 2, 1)
        self.enc3 = ConvBlock(base_features * 2, base_features * 4, 4, 2, 1)
        self.enc4 = ConvBlock(base_features * 4, base_features * 8, 4, 2, 1)

        self.bottleneck = ConvBlock(base_features * 8, base_features * 16, 3, 1, 1)

        self.decoder = AttentionDecoder(base_features, out_channels, deep_supervision)

    def forward(self, x):
        # Disable deep supervision in evaluation / inference
        if not self.training:
            self.decoder.deep_supervision = False
        else:
            self.decoder.deep_supervision = self.deep_supervision

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        b = self.bottleneck(x4)
        return self.decoder(b, x4, x3, x2, x1)