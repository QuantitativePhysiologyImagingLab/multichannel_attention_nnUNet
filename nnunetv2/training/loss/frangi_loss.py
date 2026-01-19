import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Sequence, Tuple, Union
from torch.cuda.amp import autocast

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

def eigh_3x3_symmetric(H: torch.Tensor):
    """
    H: (B, N, 3, 3) symmetric
    Returns:
       evals: (B, N, 3)
       evecs: (B, N, 3, 3)
    """
    # extract components
    a = H[...,0,0]
    b = H[...,1,1]
    c = H[...,2,2]
    d = H[...,0,1]
    e = H[...,0,2]
    f = H[...,1,2]

    # --- eigenvalues using analytic cubic solution (guaranteed real) ---
    # trace
    m = (a + b + c) / 3.0

    # centered matrix entries
    a_ = a - m
    b_ = b - m
    c_ = c - m

    # p^2 = 1/6 * (sum of squares of centered diag + 2*d^2 + 2*e^2 + 2*f^2)
    p2 = (a_*a_ + b_*b_ + c_*c_ + 2*(d*d + e*e + f*f)) / 6.0
    p = torch.sqrt(p2.clamp_min(1e-12))

    # build matrix B = (1/p) * (H - mI)
    B00 = a_ / p
    B11 = b_ / p
    B22 = c_ / p
    B01 = d / p
    B02 = e / p
    B12 = f / p

    # determinant of B
    detB = (
        B00*(B11*B22 - B12*B12)
        - B01*(B01*B22 - B12*B02)
        + B02*(B01*B12 - B11*B02)
    )

    # clamp for numerical stability
    detB = detB.clamp(-1.0, 1.0)

    phi = torch.acos(detB) / 3.0

    # eigenvalues (closed form)
    eig1 = m + 2*p*torch.cos(phi + 0)
    eig3 = m + 2*p*torch.cos(phi + 2*torch.pi/3)
    eig2 = 3*m - eig1 - eig3     # ensures ordering

    # sort ascending by absolute magnitude (Frangi convention)
    evals = torch.stack([eig1, eig2, eig3], dim=-1)

    # --- eigenvectors ---
    # get eigenvectors for each eigenvalue by solving (H - λI)v = 0
    # do it with torch.linalg.cross + normalization
    evecs = []
    for i in range(3):
        lam = evals[..., i][...,None,None]
        M = H - lam * torch.eye(3, device=H.device, dtype=H.dtype)

        # pick two rows and take cross product
        v = torch.linalg.cross(M[...,0,:], M[...,1,:])
        v = v / (v.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        evecs.append(v)

    evecs = torch.stack(evecs, dim=-1)  # (B,N,3,3)
    return evals, evecs

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
    orig_dtype = H.dtype
    H32 = H.to(torch.float32)

    # make sure it's symmetric & finite
    H32 = 0.5 * (H32 + H32.transpose(-1, -2))
    H32 = torch.nan_to_num(H32, nan=0.0, posinf=1e4, neginf=-1e4)
    H32 = H32.clamp(min=-1e4, max=1e4)

    # eigen-decomp with torch.linalg.eigh (stable & differentiable)
    H_flat = H32.view(-1, 3, 3)                    # (B*DHW, 3, 3)

    # finite_mask = torch.isfinite(H_flat)
    # if not finite_mask.all():
    #     bad_count = (~finite_mask).sum().item()
    #     print(f"[FRANGI EIGH] Non-finite entries in H_flat: {bad_count}", flush=True)
    #     bad_vals = H_flat[~finite_mask]
    #     print("[FRANGI EIGH] Example bad values:", bad_vals[:10], flush=True)
    #     H_flat = torch.nan_to_num(H_flat, nan=0.0, posinf=1e4, neginf=-1e4)
    
    #  analytic symmetric 3x3 eigensolver (no cuSOLVER)
    evals32, evecs32 = eigh_3x3_symmetric(H_flat)

    # reshape back to (B, D*H*W, 3) and (B, D*H*W, 3, 3)
    B, _, D_, H_, W_ = I.shape
    evals32 = evals32.view(B, D_ * H_ * W_, 3)
    evecs32 = evecs32.view(B, D_ * H_ * W_, 3, 3)
    # evals32 = evals32.view(B, -1, 3)               # (B, D*H*W, 3)
    # evecs32 = evecs32.view(B, -1, 3, 3)            # (B, D*H*W, 3, 3)

    evals = evals32.to(orig_dtype)
    evecs = evecs32.to(orig_dtype)

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
        data:       (B,3,X,Y,Z)  [0]=chi_qsm, [1]=localfield, [2]=Frangi(QSM) or 0
        """
        device = net_output.device

        # ---- cast everything to fp32 inside, return in original dtype ----
        net_output32 = net_output.float()
        chi_qsm = data[:, 0:1].to(device=device, dtype=torch.float32)
        V_I     = data[:, 2:3].to(device=device, dtype=torch.float32)

        brain_mask = (chi_qsm != 0).to(torch.float32)

        chi_qsm   = self._resize_like(chi_qsm,   net_output32, is_mask=False)
        V_I       = self._resize_like(V_I,       net_output32, is_mask=False)
        brain_mask= self._resize_like(brain_mask,net_output32, is_mask=True)

        # softmax on fp32 logits
        probs = torch.softmax(net_output32, dim=1)
        vein_p = probs[:, self.vein_channel:self.vein_channel+1]  # (B,1,*,*,*)
        vein_p = vein_p.clamp(0.0, 1.0)

        alpha, tau = self.alpha_tau

        # ---- gate from image Frangi (prior, no grad) ----
        with torch.no_grad():
            V_gate = torch.sigmoid(alpha * (V_I - tau))
            V_gate = F.max_pool3d(V_gate, kernel_size=3, stride=1, padding=1)

        valid = (V_gate > 0.5) & (brain_mask > 0)

        # ---- Frangi on prediction: NO grad through Frangi ----
        with torch.no_grad():
            # smooth prob
            P_blur = F.avg_pool3d(
                F.pad(vein_p, (1,1,1,1,1,1), mode='reflect'),
                kernel_size=3, stride=1
            )
            V_P_prior, _ = frangi_3d(
                P_blur,
                sigmas=self.sig_mask,
                alpha=0.8,
                beta=0.8,
                c=4.0,
                bright_vessels=True,
                return_scale=False
            )
            V_P_prior = V_P_prior.clamp(0.0, 1.0)

        # ---- now ALL grads come from simple comparisons: vein_p → loss ----
        margin = 0.1

        if valid.any():
            vmask = valid

            # encourage high vein prob where Frangi(P) & Frangi(QSM) agree
            # (V_P_prior and V_gate are treated as fixed priors)
            target_tube = (V_P_prior * V_gate).detach()

            # 1) selfV: encourage vein_p ≈ target_tube inside gate
            eps = 1e-6
            pt = vein_p[vmask].clamp(eps, 1-eps)
            tt = target_tube[vmask].clamp(eps, 1-eps)
            loss_selfV = (tt * (tt/pt).log() + (1-tt) * ((1-tt)/(1-pt)).log()).mean()

            # 2) completion hinge: want vein_p >= V_I + margin inside gate
            desired = (V_I + margin).clamp(0.0, 1.0)
            hinge_term = F.softplus(desired - vein_p, beta=10.0)
            loss_hinge = (hinge_term * V_gate)[vmask].mean()
        else:
            loss_selfV = vein_p.new_zeros(())
            loss_hinge = vein_p.new_zeros(())

        # 3) background suppression: penalize vein prob where image is confidently non-tubular
        non_vessel = (V_I < 0.05).float()
        # denom = non_vessel.sum().clamp_min(1.0)
        loss_bg = torch.mean(torch.square(vein_p * non_vessel))

        # ---- combine, clamp → fp16-safe ----
        loss = 5.0 * loss_selfV + 10.0 * loss_hinge + 10.0 * loss_bg

        # snapshot originals for logging
        selfV_raw  = loss_selfV
        hinge_raw  = loss_hinge
        bg_raw     = loss_bg

        # sanitize in-place
        if not torch.isfinite(loss_selfV).item():
            print("[WARN] FrangiLoss non-finite selfV: "
                f"selfV={float(selfV_raw)}, hinge={float(hinge_raw)}, bg={float(bg_raw)}",
                flush=True)
            # loss_selfV = torch.nan_to_num(loss_selfV, nan=0.0, posinf=0.0, neginf=0.0)

        if not torch.isfinite(loss_hinge).item():
            print("[WARN] FrangiLoss non-finite hinge: "
                f"selfV={float(selfV_raw)}, hinge={float(hinge_raw)}, bg={float(bg_raw)}",
                flush=True)
            # loss_hinge = torch.nan_to_num(loss_hinge, nan=0.0, posinf=0.0, neginf=0.0)

        if not torch.isfinite(loss_bg).item():
            print("[WARN] FrangiLoss non-finite bg: "
                f"selfV={float(selfV_raw)}, hinge={float(hinge_raw)}, bg={float(bg_raw)}",
                flush=True)
            # loss_bg = torch.nan_to_num(loss_bg, nan=0.0, posinf=0.0, neginf=0.0)

        # return in same dtype as net_output for AMP / GradScaler
        return loss
