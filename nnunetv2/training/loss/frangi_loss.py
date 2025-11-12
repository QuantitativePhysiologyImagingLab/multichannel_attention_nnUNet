import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Sequence, Tuple, Union

# -----------------------------
# 1D Gaussian & its derivatives
# -----------------------------
def _gaussian_1d(sigma: float, radius: Optional[int] = None, device=None, dtype=None) -> Tensor:
    if radius is None:
        radius = max(1, int(torch.ceil(torch.tensor(3*sigma)).item()))
    x = torch.arange(-radius, radius+1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    return g

def _gaussian_1d_first_deriv(sigma: float, radius: Optional[int] = None, device=None, dtype=None) -> Tensor:
    # d/dx of Gaussian (unnormalized), normalized to zero mean
    g = _gaussian_1d(sigma, radius, device, dtype)
    x = torch.arange(-(g.numel()//2), g.numel()//2 + 1, device=device, dtype=dtype)
    dg = -(x/(sigma**2)) * g
    dg = dg - dg.mean()  # zero-sum, helps stability
    return dg

def _gaussian_1d_second_deriv(sigma: float, radius: Optional[int] = None, device=None, dtype=None) -> Tensor:
    # d^2/dx^2 of Gaussian (LoG component)
    g = _gaussian_1d(sigma, radius, device, dtype)
    x = torch.arange(-(g.numel()//2), g.numel()//2 + 1, device=device, dtype=dtype)
    d2g = ((x**2)/(sigma**4) - 1/(sigma**2)) * g
    d2g = d2g - d2g.mean()  # zero-sum
    return d2g

# -----------------------------
# Separable 3D convolution
# -----------------------------
def _sep_conv3d(x: Tensor, kx: Tensor, ky: Tensor, kz: Tensor, groups: int = 1) -> Tensor:
    """
    Apply separable conv with reflect padding along z (depth), y (height), x (width).
    x: (B, C, D, H, W); k*: (K,)
    """
    B, C, D, H, W = x.shape
    device, dtype = x.device, x.dtype

    def conv1d_along(x: Tensor, k: Tensor, dim: int) -> Tensor:
        # dim: 0->z, 1->y, 2->x (spatial axes within (D,H,W))
        pad = [0, 0, 0, 0, 0, 0]  # (W_left, W_right, H_top, H_bottom, D_front, D_back)
        ksz = k.numel()
        r = ksz // 2
        if dim == 2:     # x/width
            pad[0] = pad[1] = r
            k3 = k.view(1, 1, 1, 1, ksz)
        elif dim == 1:   # y/height
            pad[2] = pad[3] = r
            k3 = k.view(1, 1, 1, ksz, 1)
        else:            # z/depth
            pad[4] = pad[5] = r
            k3 = k.view(1, 1, ksz, 1, 1)

        x = F.pad(x, pad, mode='reflect')
        weight = k3.to(device=device, dtype=dtype).repeat(C, 1, 1, 1, 1)
        return F.conv3d(x, weight, groups=C)

    x = conv1d_along(x, kz, 0)
    x = conv1d_along(x, ky, 1)
    x = conv1d_along(x, kx, 2)
    return x

# -----------------------------
# Hessian (scale-normalized)
# -----------------------------
def _hessian_3d(I: Tensor, sigma: float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute scale-normalized second-order partials at scale sigma.
    I: (B,1,D,H,W)
    Returns: I_xx, I_yy, I_zz, I_xy, I_xz, I_yz
    """
    device, dtype = I.device, I.dtype
    g  = _gaussian_1d(sigma, device=device, dtype=dtype)
    d1 = _gaussian_1d_first_deriv(sigma, device=device, dtype=dtype)
    d2 = _gaussian_1d_second_deriv(sigma, device=device, dtype=dtype)

    # pure seconds
    I_xx = _sep_conv3d(I, d2, g,  g)
    I_yy = _sep_conv3d(I, g,  d2, g)
    I_zz = _sep_conv3d(I, g,  g,  d2)

    # mixed seconds: d^2/dxdy = d/dx(d/dy)
    I_xy = _sep_conv3d(I, d1, d1, g)
    I_xz = _sep_conv3d(I, d1, g,  d1)
    I_yz = _sep_conv3d(I, g,  d1, d1)

    # Lindeberg normalization for 2nd-order derivatives
    s2 = sigma**2
    I_xx = s2*I_xx; I_yy = s2*I_yy; I_zz = s2*I_zz
    I_xy = s2*I_xy; I_xz = s2*I_xz; I_yz = s2*I_yz
    return I_xx, I_yy, I_zz, I_xy, I_xz, I_yz

# -----------------------------
# Frangi 3D (single scale)
# -----------------------------
def _frangi_3d_single(
    I: Tensor,
    sigma: float,
    alpha: float = 0.5,
    beta: float = 0.5,
    c: float = 15.0,
    bright_vessels: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Single-scale vesselness and axis.
    I: (B,1,D,H,W) normalized
    Returns: V (B,1,D,H,W) in [0,1], axis e_vec (B,3,D,H,W)
    """
    B, C, D, H, W = I.shape

    I_xx, I_yy, I_zz, I_xy, I_xz, I_yz = _hessian_3d(I, sigma)

    # Assemble symmetric Hessian per voxel -> eigendecomp
    # Each of these is (B,1,D,H,W)
    H11, H22, H33 = I_xx, I_yy, I_zz
    H12, H13, H23 = I_xy, I_xz, I_yz

    # Stack to (B,1,D,H,W,3,3)
    H = torch.stack([
        torch.stack([H11, H12, H13], dim=-1),
        torch.stack([H12, H22, H23], dim=-1),
        torch.stack([H13, H23, H33], dim=-1),
    ], dim=-2).contiguous()  # (B,1,D,H,W,3,3)

    # Flatten channel+spatial dims -> (B, D*H*W, 3, 3)
    _, _, D_, H_, W_, _, _ = H.shape
    H = H.view(B, 1, D_, H_, W_, 3, 3).flatten(1, 4)  # (B, D*H*W, 3, 3)

    # Eigendecomposition
    evals, evecs = torch.linalg.eigh(H)  # (B, D*H*W, 3), (B, D*H*W, 3, 3)

    # Sort by |lambda| ascending
    idx = torch.argsort(evals.abs(), dim=-1, stable=True)
    evals = torch.gather(evals, -1, idx)
    evecs = torch.gather(evecs, -1, idx.unsqueeze(-2).expand(-1, -1, 3, -1))

    l1, l2, l3 = evals[..., 0], evals[..., 1], evals[..., 2]  # (B, D*H*W)
    e_axis = evecs[..., 0]                                     # (B, D*H*W, 3)

    # Frangi measures
    eps = 1e-12
    Ra = (l2.abs() / (l3.abs() + eps))
    Rb = (l1.abs() / torch.sqrt(l2.abs() * l3.abs() + eps))
    S  = torch.sqrt((l1**2 + l2**2 + l3**2).clamp_min(eps))

    # Vesselness (Frangi '98 variant)
    expRa = torch.exp(-(Ra**2) / (2*alpha**2))
    expRb = torch.exp(-(Rb**2) / (2*beta**2))
    expS  = 1.0 - torch.exp(-(S**2) / (2*c**2))
    V = expRa * expRb * expS

    # Polarity mask:
    if bright_vessels:
        mask = ((l2 < 0) & (l3 < 0)).float()
    else:
        mask = ((l2 > 0) & (l3 > 0)).float()

    V = (V * mask).view(B, 1, D_, H_, W_).clamp(0, 1)

    # Axis back to (B,3,D,H,W)
    e_axis = e_axis.view(B, D_, H_, W_, 3).permute(0, 4, 1, 2, 3)
    e_axis = e_axis / (e_axis.norm(dim=1, keepdim=True).clamp_min(1e-8))

    return V, e_axis

# -----------------------------
# Multiscale Frangi 3D
# -----------------------------
def frangi_3d(
    I: Tensor,
    sigmas: Sequence[float] = (0.6, 0.9, 1.2, 1.8),
    alpha: float = 0.5,
    beta: float = 0.5,
    c: float = 15.0,
    bright_vessels: bool = True,
    return_scale: bool = False
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """
    Multiscale 3D Frangi vesselness. Autograd-friendly.
    Use 'with torch.no_grad()' on the call site if you want a fixed prior.
    """
    assert I.ndim == 5 and I.shape[1] == 1, "I must be (B,1,D,H,W)"
    B, _, D, H, W = I.shape

    V_best = torch.zeros_like(I)
    axis_best = torch.zeros(B, 3, D, H, W, device=I.device, dtype=I.dtype)
    s_map = torch.zeros_like(I)

    for s in sigmas:
        V_s, axis_s = _frangi_3d_single(I, s, alpha, beta, c, bright_vessels)
        better = V_s > V_best
        V_best = torch.where(better, V_s, V_best)
        axis_best = torch.where(better.expand_as(axis_best), axis_s, axis_best)
        s_map = torch.where(better, torch.full_like(V_s, float(s)), s_map)

    if return_scale:
        return V_best, axis_best, s_map
    return V_best, axis_best

# -----------------------------
# Loss wrapper
# -----------------------------
class FrangiLoss(nn.Module):
    """
    Uses Frangi(QSM) as a gated prior and Frangi(P) as a differentiable shape loss.

    Inputs:
        I_chi: (B,1,D,H,W) z-scored QSM
        P    : (B,1,D,H,W) sigmoid probabilities
        Y    : (B,1,D,H,W) optional imperfect labels
    """
    def __init__(self):
        super().__init__()
        self.sig_img  = (0.6, 0.9, 1.2, 1.8)  # image scales
        self.sig_mask = (0.01, 0.2, 0.3)       # mask/prob scales
        self.alpha_tau = (6.0, 1E-5)          # gate sharpness/midpoint
        self.vein_channel = 1
        
    def _resize_like(self, x, ref, is_mask=False):
        if x.shape[2:] == ref.shape[2:]:
            return x
        mode = 'nearest' if is_mask else 'trilinear'
        return F.interpolate(x, size=ref.shape[2:], mode=mode, align_corners=False if mode=='trilinear' else None)
    
    def forward(self, net_output, data):
        """
        net_output: (B,C,X,Y,Z)
        target:     (B,2,X,Y,Z)  [0]=chi_qsm_ppm, [1]=localfield_ppm, [2]=Frangi_vesselness
        """
        # Cast priors to the net’s dtype/device
        chi_qsm = data[:, 0].to(device=net_output.device, dtype=net_output.dtype)   # ppm
        V_I  = data[:, 2].to(device=net_output.device, dtype=net_output.dtype)   # frangi QSM

        brain_mask = (chi_qsm != 0).to(net_output.dtype)

        probs = torch.softmax(net_output, dim=1)  # (B,C,X,Y,Z)
        vein_p = probs[:, self.vein_channel:self.vein_channel+1]  # (B,1,X,Y,Z)
        vein_eval = (vein_p >= 0.5).to(net_output.dtype)
        valid = (vein_eval > 0) & (brain_mask > 0)

        chi_qsm    = self._resize_like(chi_qsm, net_output, is_mask=False)
        V_I     = self._resize_like(V_I, net_output, is_mask=False)
        brain_mask = self._resize_like(brain_mask, net_output, is_mask=True)

        alpha, tau = self.alpha_tau

        # --- Frangi on image (prior, no grads) ---
        with torch.no_grad():

            V_gate = torch.sigmoid(alpha * (V_I - tau))
            # tiny dilation to bridge small breaks
            V_gate = F.max_pool3d(V_gate, kernel_size=3, stride=1, padding=1)

        # --- Frangi on prediction (differentiable) ---
        # light blur for stability of Hessian on probability map
        P_blur = F.avg_pool3d(F.pad(vein_p, (1,1,1,1,1,1), mode='reflect'), kernel_size=3, stride=1)
        V_P, _ = frangi_3d(
            P_blur, self.sig_mask, 0.8, 0.8, 4.0, True, False
        )

        # --- Core objectives ---
        # Reward tubeness where image suggests vessels
        loss_selfV = -((V_P * V_gate)[valid]).mean()
        # One-sided completion hinge: push V_P >= V_I + margin inside gate
        margin = 0.1
        loss_hinge = ((torch.relu((V_I+margin) - V_P) * V_gate)[valid]).mean()
        # Suppress predictions where image is confidently non-tubular
        non_vessel = (V_I < 0.05).float()
        denom = non_vessel.sum().clamp_min(1)
        loss_bg = (vein_p * non_vessel).sum() / denom

        # Weights (tune as needed)
        loss = (5*loss_selfV + 10*loss_hinge + 10*loss_bg)
        return loss