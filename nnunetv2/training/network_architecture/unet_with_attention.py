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

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: learns per-domain scale and shift for a feature map."""
    def __init__(self, embed_dim: int, num_channels: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 2 * num_channels)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, *spatial)   emb: (B, embed_dim)
        gb = self.fc(emb)                          # (B, 2C)
        gamma, beta = gb.chunk(2, dim=1)
        shape = (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)
        gamma = gamma.view(shape) + 1.0            # init near identity
        beta  = beta.view(shape)
        return gamma * x + beta


DOMAIN_METHODS = ['TGV', 'medi', 'l1', 'star', 'ilsqr']
METHOD_TO_IDX  = {m: i for i, m in enumerate(DOMAIN_METHODS)}

# Generated from vein_mapping.xlsx — source of truth for QSM method per subject.
VEIN_TO_DOMAIN = {
    'VEIN_000': 0,  # TGV
    'VEIN_001': 0,  # TGV
    'VEIN_002': 0,  # TGV
    'VEIN_003': 0,  # TGV
    'VEIN_004': 0,  # TGV
    'VEIN_005': 0,  # TGV
    'VEIN_006': 0,  # TGV
    'VEIN_007': 0,  # TGV
    'VEIN_008': 0,  # TGV
    'VEIN_009': 0,  # TGV
    'VEIN_010': 0,  # TGV
    'VEIN_011': 0,  # TGV
    'VEIN_012': 0,  # TGV
    'VEIN_013': 0,  # TGV
    'VEIN_014': 0,  # TGV
    'VEIN_015': 0,  # TGV
    'VEIN_016': 0,  # TGV
    'VEIN_017': 0,  # TGV
    'VEIN_018': 0,  # TGV
    'VEIN_019': 0,  # TGV
    'VEIN_020': 0,  # TGV
    'VEIN_021': 0,  # TGV
    'VEIN_022': 0,  # TGV
    'VEIN_023': 0,  # TGV
    'VEIN_024': 0,  # TGV
    'VEIN_025': 0,  # TGV
    'VEIN_026': 0,  # TGV
    'VEIN_027': 0,  # TGV
    'VEIN_028': 0,  # TGV
    'VEIN_029': 0,  # TGV
    'VEIN_030': 1,  # medi
    'VEIN_031': 2,  # l1
    'VEIN_032': 3,  # star
    'VEIN_033': 4,  # ilsqr
    'VEIN_034': 1,  # medi
    'VEIN_035': 2,  # l1
    'VEIN_036': 3,  # star
    'VEIN_037': 4,  # ilsqr
    'VEIN_038': 1,  # medi
    'VEIN_039': 2,  # l1
    'VEIN_040': 3,  # star
    'VEIN_041': 4,  # ilsqr
    'VEIN_042': 1,  # medi
    'VEIN_043': 2,  # l1
    'VEIN_044': 3,  # star
    'VEIN_045': 4,  # ilsqr
    'VEIN_046': 1,  # medi
    'VEIN_047': 2,  # l1
    'VEIN_048': 3,  # star
    'VEIN_049': 4,  # ilsqr
    'VEIN_050': 1,  # medi
    'VEIN_051': 2,  # l1
    'VEIN_052': 3,  # star
    'VEIN_053': 4,  # ilsqr
    'VEIN_054': 1,  # medi
    'VEIN_055': 2,  # l1
    'VEIN_056': 3,  # star
    'VEIN_057': 4,  # ilsqr
    'VEIN_058': 1,  # medi
    'VEIN_059': 2,  # l1
    'VEIN_060': 3,  # star
    'VEIN_061': 4,  # ilsqr
    'VEIN_062': 1,  # medi
    'VEIN_063': 2,  # l1
    'VEIN_064': 3,  # star
    'VEIN_065': 4,  # ilsqr
    'VEIN_066': 1,  # medi
    'VEIN_067': 2,  # l1
    'VEIN_068': 3,  # star
    'VEIN_069': 4,  # ilsqr
    'VEIN_070': 1,  # medi
    'VEIN_071': 2,  # l1
    'VEIN_072': 3,  # star
    'VEIN_073': 4,  # ilsqr
    'VEIN_074': 1,  # medi
    'VEIN_075': 2,  # l1
    'VEIN_076': 3,  # star
    'VEIN_077': 4,  # ilsqr
    'VEIN_078': 1,  # medi
    'VEIN_079': 2,  # l1
    'VEIN_080': 3,  # star
    'VEIN_081': 4,  # ilsqr
    'VEIN_082': 1,  # medi
    'VEIN_083': 2,  # l1
    'VEIN_084': 3,  # star
    'VEIN_085': 4,  # ilsqr
    'VEIN_086': 1,  # medi
    'VEIN_087': 2,  # l1
    'VEIN_088': 3,  # star
    'VEIN_089': 4,  # ilsqr
    'VEIN_090': 1,  # medi
    'VEIN_091': 2,  # l1
    'VEIN_092': 3,  # star
    'VEIN_093': 4,  # ilsqr
    'VEIN_094': 1,  # medi
    'VEIN_095': 2,  # l1
    'VEIN_096': 3,  # star
    'VEIN_097': 4,  # ilsqr
    'VEIN_098': 1,  # medi
    'VEIN_099': 2,  # l1
    'VEIN_100': 3,  # star
    'VEIN_101': 4,  # ilsqr
    'VEIN_102': 1,  # medi
    'VEIN_103': 2,  # l1
    'VEIN_104': 3,  # star
    'VEIN_105': 4,  # ilsqr
    'VEIN_106': 1,  # medi
    'VEIN_107': 2,  # l1
    'VEIN_108': 3,  # star
    'VEIN_109': 4,  # ilsqr
    'VEIN_110': 1,  # medi
    'VEIN_111': 2,  # l1
    'VEIN_112': 3,  # star
    'VEIN_113': 4,  # ilsqr
    'VEIN_114': 1,  # medi
    'VEIN_115': 2,  # l1
    'VEIN_116': 3,  # star
    'VEIN_117': 4,  # ilsqr
    'VEIN_118': 1,  # medi
    'VEIN_119': 2,  # l1
    'VEIN_120': 3,  # star
    'VEIN_121': 4,  # ilsqr
    'VEIN_122': 1,  # medi
    'VEIN_123': 2,  # l1
    'VEIN_124': 3,  # star
    'VEIN_125': 4,  # ilsqr
    'VEIN_126': 1,  # medi
    'VEIN_127': 2,  # l1
    'VEIN_128': 3,  # star
    'VEIN_129': 4,  # ilsqr
    'VEIN_130': 1,  # medi
    'VEIN_131': 2,  # l1
    'VEIN_132': 3,  # star
    'VEIN_133': 4,  # ilsqr
    'VEIN_134': 1,  # medi
    'VEIN_135': 2,  # l1
    'VEIN_136': 3,  # star
    'VEIN_137': 4,  # ilsqr
    'VEIN_138': 1,  # medi
    'VEIN_139': 2,  # l1
    'VEIN_140': 3,  # star
    'VEIN_141': 4,  # ilsqr
    'VEIN_142': 1,  # medi
    'VEIN_143': 2,  # l1
    'VEIN_144': 3,  # star
    'VEIN_145': 4,  # ilsqr
    'VEIN_146': 1,  # medi
    'VEIN_147': 2,  # l1
    'VEIN_148': 3,  # star
    'VEIN_149': 4,  # ilsqr
}


def vein_to_domain_idx(case_id: str) -> int:
    """Maps VEIN_XXX -> domain index using the Excel mapping. Falls back to 0 (TGV) if unknown."""
    return VEIN_TO_DOMAIN.get(case_id, 0)


class UNetWithAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, patch_size=(80, 80, 80),
                 deep_supervision=False, base_features=64,
                 num_domains=5, domain_embed_dim=32):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_domains = num_domains
        self.default_domain_idx = 0   # used at inference when no idx is supplied

        self.enc1 = ConvBlock(in_channels, base_features, 4, 2, 1)
        self.enc2 = ConvBlock(base_features, base_features * 2, 4, 2, 1)
        self.enc3 = ConvBlock(base_features * 2, base_features * 4, 4, 2, 1)
        self.enc4 = ConvBlock(base_features * 4, base_features * 8, 4, 2, 1)

        self.bottleneck = ConvBlock(base_features * 8, base_features * 16, 3, 1, 1)

        self.decoder = AttentionDecoder(base_features, out_channels, deep_supervision)

        if num_domains > 1:
            self.domain_embed  = nn.Embedding(num_domains, domain_embed_dim)
            self.film_enc1     = FiLMLayer(domain_embed_dim, base_features)
            self.film_enc2     = FiLMLayer(domain_embed_dim, base_features * 2)
            self.film_enc3     = FiLMLayer(domain_embed_dim, base_features * 4)
            self.film_enc4     = FiLMLayer(domain_embed_dim, base_features * 8)
            self.film_bottle   = FiLMLayer(domain_embed_dim, base_features * 16)
        else:
            self.domain_embed = None

    def forward(self, x, domain_idx=None):
        if not self.training:
            self.decoder.deep_supervision = False
        else:
            self.decoder.deep_supervision = self.deep_supervision

        if self.domain_embed is not None:
            if domain_idx is None:
                domain_idx = torch.full((x.shape[0],), self.default_domain_idx,
                                        dtype=torch.long, device=x.device)
            emb = self.domain_embed(domain_idx)   # (B, embed_dim)
            x1 = self.film_enc1(self.enc1(x),          emb)
            x2 = self.film_enc2(self.enc2(x1),         emb)
            x3 = self.film_enc3(self.enc3(x2),         emb)
            x4 = self.film_enc4(self.enc4(x3),         emb)
            b  = self.film_bottle(self.bottleneck(x4), emb)
        else:
            x1 = self.enc1(x)
            x2 = self.enc2(x1)
            x3 = self.enc3(x2)
            x4 = self.enc4(x3)
            b  = self.bottleneck(x4)

        return self.decoder(b, x4, x3, x2, x1)