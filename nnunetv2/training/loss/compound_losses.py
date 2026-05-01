import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.loss.vein_susceptometry_loss import PhysicsFieldLoss
from nnunetv2.training.loss.tversky_loss import FocalTverskyLoss
from nnunetv2.training.loss.frangi_loss import FrangiLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.network_architecture.unet_with_attention import DOMAIN_METHODS
from torch import nn
import torch.nn.functional as F

class DeepSupervisionWrapperPassKwargs(nn.Module):
    """
    Wrap a base loss to handle deep supervision safely while forwarding *args/**kwargs.
    - Accepts tensor or list for net_output/target.
    - Resizes target per head to out_i.shape[2:].
    - Aligns weight count to #heads.
    """
    def __init__(self, base_loss: nn.Module, weights):
        super().__init__()
        w = torch.as_tensor(list(weights), dtype=torch.float32)
        w = torch.where(w > 0, w, torch.zeros_like(w))
        s = float(w.sum()) if float(w.sum()) > 0 else 1.0
        self.register_buffer("weights", w / s)
        self.base_loss = base_loss

    def forward(self, net_output, target, *args, **kwargs):
        outs = net_output if isinstance(net_output, (list, tuple)) else [net_output]
        tgts = target if isinstance(target, (list, tuple)) else [target]

        # Use first target as the "source" to resize others if needed
        t0 = tgts[0]
        tgts_resized = []
        for i, out_i in enumerate(outs):
            if i < len(tgts) and tgts[i].shape[2:] == out_i.shape[2:]:
                tgt_i = tgts[i]
            else:
                # nearest for labels; keeps dtype
                tgt_i = F.interpolate(t0.float(), size=out_i.shape[2:], mode='nearest').type_as(t0)
            tgts_resized.append(tgt_i)

        # Align weights length with number of heads
        w = self.weights
        if len(w) != len(outs):
            w = w[:len(outs)] if len(w) > len(outs) else torch.cat([w, w[-1:].repeat(len(outs)-len(w))])

        total = 0.0
        for out_i, tgt_i, wi in zip(outs, tgts_resized, w):
            if float(wi) == 0.0:
                continue
            total = total + float(wi) * self.base_loss(out_i, tgt_i, *args, **kwargs)
        return total

class VeinPhysics_Frangi_DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, vpl_kwargs, weight_ce=2, weight_dice=2,
                 weight_tversky=0.5, weight_physics=2, weight_frangi=0.15,
                 weight_volume=1.0, volume_thresh=1.25,
                 ignore_label=None, dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(VeinPhysics_Frangi_DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_physics = weight_physics
        self.weight_tversky = weight_tversky
        self.weight_frangi = weight_frangi
        self.weight_volume = weight_volume
        self.volume_thresh = volume_thresh
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.vpl = PhysicsFieldLoss(**vpl_kwargs)
        self.tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
        self.frangi = FrangiLoss()

    def forward(self,
                net_output: torch.Tensor,
                target: torch.Tensor,
                data: torch.Tensor,
                b0_dir: torch.Tensor = None,
                domain_idx: torch.Tensor = None) -> torch.Tensor:
        """
        net_output: (B, C, X, Y, Z) logits
        target    : (B, 1 or 2, X, Y, Z) (2 if includes chi/localfield channels for physics)
        data      : (B, C, X, Y, Z) input data tensor
        b0_dir    : (B,3) or (3,) unit vector(s) in image axes
        domain_idx: (B,) long tensor of QSM method indices
        """
        
        # ---- ignore-label handling for Dice/CE ----
        if self.ignore_label is not None:
            assert target.shape[1] >= 1, "target needs at least a class channel"
            mask = target[:, :1] != self.ignore_label
            target_dice = torch.where(mask, target[:, :1], 0)
            num_fg = mask.sum()
        else:
            mask = None
            target_dice = target[:, :1] if target.shape[1] >= 1 else target  # keep class channel for CE/Dice

        # ---- CE & Dice ----
        if self.weight_dice != 0:
            dc_loss = self.dc(net_output, target_dice, loss_mask=mask)
        else:
            dc_loss = self.dc(net_output.detach(), target_dice, loss_mask=mask)
        
        ce_loss = self.ce(net_output, target_dice[:, 0]) if (self.weight_ce != 0 and self.ce is not None and (self.ignore_label is None or num_fg > 0)) else 0.0

        total = self.weight_ce * ce_loss + self.weight_dice * dc_loss

        # ---- Tversky: all samples ----
        if self.weight_tversky != 0:
            tversky_loss = self.tversky(net_output, target_dice)
            total = total + self.weight_tversky * tversky_loss
        else:
            tversky_loss = self.tversky(net_output.detach(), target_dice)

        # ---- Physics term ----
        if self.vpl is not None and self.weight_physics != 0:
            if b0_dir is not None:
                if b0_dir.ndim == 1:
                    B = net_output.shape[0]
                    b0_dir = b0_dir.view(1, 3).repeat(B, 1).to(net_output.device, dtype=net_output.dtype)
                elif b0_dir.ndim == 2:
                    b0_dir = b0_dir.to(net_output.device, dtype=net_output.dtype)
                else:
                    raise ValueError("b0_dir must be shape (3,) or (B,3)")

            phys_loss, _metrics = self.vpl(
                net_output = net_output,
                data       = data,
                b0_dir     = b0_dir,
                target     = target,
            )
            total = total + self.weight_physics * phys_loss

        # ---- Frangi: all samples ----
        if self.frangi is not None and self.weight_frangi != 0:
            with torch.cuda.amp.autocast(enabled=False):
                frangi_loss = self.frangi(net_output=net_output.float(), data=data)
            total = total + self.weight_frangi * frangi_loss.to(total.dtype)

        # ---- Volume limiter: penalise over-prediction beyond volume_thresh × GT ----
        if self.weight_volume > 0:
            with torch.no_grad():
                gt_vein = (target_dice == 1).float()
                gt_vol  = gt_vein.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
            probs_vol = torch.softmax(net_output.float(), dim=1)
            pred_vol  = probs_vol[:, 1:2].sum(dim=(1, 2, 3, 4))
            vol_loss  = torch.relu(pred_vol / gt_vol - self.volume_thresh).mean()
            total = total + self.weight_volume * vol_loss.to(total.dtype)

        if domain_idx is not None:
            methods = [DOMAIN_METHODS[int(i)] for i in domain_idx]
            method_str = ','.join(methods)
        else:
            method_str = 'unknown'

        _lv = locals()
        dc_raw = float(dc_loss)
        wdc    = self.weight_dice * dc_raw

        def _wstr(key, w):
            if key not in _lv:
                return 'n/a'
            raw = float(_lv[key])
            return f'{raw:.4f}(w={w * raw:.4f})'

        phys_raw    = float(_lv['phys_loss'])    if 'phys_loss'    in _lv else None
        frangi_raw  = float(_lv['frangi_loss'])  if 'frangi_loss'  in _lv else None
        vol_raw     = float(_lv['vol_loss'])     if 'vol_loss'     in _lv else None
        tversky_raw = float(_lv['tversky_loss']) if 'tversky_loss' in _lv else None

        print(
            f'[{method_str}] '
            f'DC: {dc_raw:.4f}(w={wdc:.4f})  '
            f'Tversky: {_wstr("tversky_loss", self.weight_tversky)}  '
            f'Phys: {_wstr("phys_loss", self.weight_physics)}  '
            f'Frangi: {_wstr("frangi_loss", self.weight_frangi)}  '
            f'Vol: {_wstr("vol_loss", self.weight_volume)}',
            flush=True
        )

        # Alarm: catch imbalance early so we don't waste a full run
        if phys_raw is not None and abs(self.weight_physics * phys_raw) > abs(wdc):
            print(
                f'[ALARM] Physics overwhelms Dice! '
                f'wPhys={self.weight_physics * phys_raw:.4f} vs wDC={wdc:.4f}  '
                f'→ reduce weight_physics (currently {self.weight_physics})',
                flush=True
            )
        if frangi_raw is not None and abs(self.weight_frangi * frangi_raw) > 0.5 * abs(wdc):
            print(
                f'[ALARM] Frangi > 50% of Dice! '
                f'wFrangi={self.weight_frangi * frangi_raw:.4f} vs wDC={wdc:.4f}  '
                f'→ reduce weight_frangi (currently {self.weight_frangi})',
                flush=True
            )

        return total

class VeinPhysics_DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, vpl_kwargs, weight_ce=1, weight_dice=0.5, 
                 weight_tversky=1, weight_physics=20, ignore_label=None, dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(VeinPhysics_DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_physics = weight_physics
        self.weight_tversky = weight_tversky
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.vpl = PhysicsFieldLoss(**vpl_kwargs)
        self.tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)

    def forward(self,
                net_output: torch.Tensor,
                target: torch.Tensor,
                data: torch.Tensor,
                b0_dir: torch.Tensor = None) -> torch.Tensor:
        """
        net_output: (B, C, X, Y, Z) logits
        target    : (B, 1 or 2, X, Y, Z) (2 if includes chi/localfield channels for physics)
        data      : (B, C, X, Y, Z) input data tensor
        b0_dir    : (B,3) or (3,) unit vector(s) in image axes
        """
        # ---- ignore-label handling for Dice/CE ----
        if self.ignore_label is not None:
            assert target.shape[1] >= 1, "target needs at least a class channel"
            mask = target[:, :1] != self.ignore_label
            target_dice = torch.where(mask, target[:, :1], 0)
            num_fg = mask.sum()
        else:
            mask = None
            target_dice = target[:, :1] if target.shape[1] >= 1 else target  # keep class channel for CE/Dice

        # ---- CE & Dice ----
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if (self.weight_dice != 0 and self.dc is not None) else 0.0
        tversky_loss = self.tversky(net_output, target_dice)
        ce_loss = self.ce(net_output, target_dice[:, 0]) if (self.weight_ce != 0 and self.ce is not None and (self.ignore_label is None or num_fg > 0)) else 0.0

        total = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_tversky*tversky_loss

        # ---- Physics term (optional but enabled here) ----
        if self.vpl is not None:
            # normalize b0_dir shape
            if b0_dir is not None:
                if b0_dir.ndim == 1:  # (3,)
                    # broadcast to batch
                    B = net_output.shape[0]
                    b0_dir = b0_dir.view(1, 3).repeat(B, 1).to(net_output.device, dtype=net_output.dtype)
                elif b0_dir.ndim == 2:  # (B,3)
                    b0_dir = b0_dir.to(net_output.device, dtype=net_output.dtype)
                else:
                    raise ValueError("b0_dir must be shape (3,) or (B,3)")

            phys_loss, _metrics = self.vpl(
                net_output = net_output,
                data       = data,
                b0_dir     = b0_dir,
                target     = target,
            )
            total = total + self.weight_physics*phys_loss

        # print("CE loss: ", ce_loss)
        # print("DC loss: ", dc_loss)
        # print("Phys loss: ", phys_loss)
        # print("Tversky: ", tversky_loss)

        return total


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
