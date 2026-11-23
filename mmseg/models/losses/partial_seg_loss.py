# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


def binary_ce_loss(pred, label, valid_mask=None, **kwargs):
    """Per-sample BCE loss with optional ignore mask.

    Args:
        pred (Tensor): Logits with shape (N, H, W).
        label (Tensor): Binary labels, shape (N, H, W).
        valid_mask (Tensor, optional): ``label != ignore_index`` mask.

    Returns:
        Tensor: Per-sample loss, shape (N,).
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(label, dtype=torch.bool)

    loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
    valid = valid_mask.float()
    loss = (loss * valid).flatten(1).sum(dim=1)
    denom = valid.flatten(1).sum(dim=1).clamp(min=1.0)
    loss = loss / denom
    return loss * valid.flatten(1).any(dim=1).float()


def binary_dice_loss(pred,
                     label,
                     valid_mask=None,
                     smooth=1.0,
                     exponent=2):
    """Per-sample Dice loss with optional ignore mask.

    Args:
        pred (Tensor): Logits with shape (N, H, W).
        label (Tensor): Binary labels, shape (N, H, W).
        valid_mask (Tensor, optional): ``label != ignore_index`` mask.
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(label, dtype=torch.bool)

    pred = torch.sigmoid(pred)
    valid = valid_mask.float()
    pred = (pred * valid).contiguous().view(pred.shape[0], -1).float()
    label = (label * valid).contiguous().view(label.shape[0], -1).float()
    valid = valid_mask.contiguous().view(valid_mask.shape[0], -1)

    num = 2 * torch.sum(torch.mul(pred, label), dim=1) + smooth
    den = torch.sum(pred.pow(exponent) + label.pow(exponent), dim=1) + smooth

    loss = 1. - num / den
    return loss * valid.any(dim=1).float()


def binary_ce_dice_loss(pred, label, valid_mask=None, smooth=1.0, **kwargs):
    loss1 = binary_ce_loss(pred, label, valid_mask=valid_mask, **kwargs)
    loss2 = binary_dice_loss(
        pred, label, valid_mask=valid_mask, smooth=smooth)
    return loss1 + loss2


def binary_loss(pred,
                label,
                loss_func,
                weight=None,
                reduction='mean',
                smooth=1.0,
                ignore_index=255,
                skip_all_bg_classes=None,
                **kwargs):
    """Multi-label binary loss with ignore-index support.

    Args:
        pred_raw (Tensor): Logits with shape (N, C, H, W).
        label_raw (Tensor): Labels with shape (N, C, H, W), values in
            {0, 1, ignore_index}.

    Returns:
        Tensor: Reduced loss.
    """
    # pred = pred.clone()
    label = label.float()
    num_classes = pred.shape[1]

    valid_mask = label != ignore_index
    label = torch.where(valid_mask, label, torch.zeros_like(label))

    skip_all_bg_mask = pred.new_zeros(num_classes, dtype=torch.bool)
    if skip_all_bg_classes is not None:
        if (len(skip_all_bg_classes) == num_classes
                and all(isinstance(x, bool) for x in skip_all_bg_classes)):
            skip_all_bg_mask = torch.tensor(
                skip_all_bg_classes, dtype=torch.bool, device=pred.device)
        else:
            for class_idx in skip_all_bg_classes:
                skip_all_bg_mask[int(class_idx)] = True

    class_losses = []
    class_valid_weights = []
    for i in range(num_classes):
        if isinstance(loss_func, tuple):
            loss_function = loss_func[i]
        else:
            loss_function = loss_func

        class_loss = loss_function(
            pred[:, i],
            label[:, i],
            valid_mask=valid_mask[:, i],
            smooth=smooth,
            **kwargs)

        valid_sample = valid_mask[:, i].flatten(1).any(dim=1).float()
        if skip_all_bg_mask[i]:
            has_fg = (label[:, i] == 1).flatten(1).any(
                dim=1).float()
            class_loss = class_loss * has_fg
            valid_sample = valid_sample * has_fg

        class_losses.append(class_loss)
        class_valid_weights.append(valid_sample)

    loss = torch.stack(class_losses, dim=1)
    valid_weight = torch.stack(class_valid_weights, dim=1)


    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        class_denom = valid_weight.sum(dim=0).clamp(min=1.0)   # [C]
        class_loss = loss.sum(dim=0) / class_denom             # [C]

        # 只保留至少有一个有效样本的类别
        valid_per_class = valid_weight.sum(dim=0) > 0

        if valid_per_class.any():
            loss = class_loss[valid_per_class].mean()
        else:
            loss = loss.sum() * 0.0
        return loss


@MODELS.register_module()
class PartialSegLoss(nn.Module):
    def __init__(self,
                 loss_type='ce',
                 reduction='mean',
                 loss_weight=1.0,
                 smooth=1.0,
                 ignore_index=255,
                 class_loss_types=None,
                 skip_all_bg_classes=None,
                 loss_name='loss_bde',
                 **kwargs):
        super(PartialSegLoss, self).__init__()
        assert loss_type in ['ce', 'dice', 'ce_dice']
        if class_loss_types is not None:
            assert all(t in ['ce', 'dice', 'ce_dice'] for t in class_loss_types)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.class_loss_types = class_loss_types
        self.skip_all_bg_classes = skip_all_bg_classes
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        def get_loss_func(loss_type):
            if loss_type == 'ce':
                return binary_ce_loss
            if loss_type == 'dice':
                return binary_dice_loss
            if loss_type == 'ce_dice':
                return binary_ce_dice_loss
            raise ValueError(f'Unsupported loss type: {loss_type}')

        if self.class_loss_types is not None:
            assert len(self.class_loss_types) == cls_score.shape[1], (
                f'Expect {cls_score.shape[1]} class loss types, '
                f'got {len(self.class_loss_types)}')
            loss_func = tuple(get_loss_func(t) for t in self.class_loss_types)
        else:
            loss_func = get_loss_func(self.loss_type)

        loss_cls = self.loss_weight * binary_loss(
            cls_score,
            label,
            loss_func,
            weight,
            reduction=reduction,
            smooth=self.smooth,
            ignore_index=self.ignore_index,
            skip_all_bg_classes=self.skip_all_bg_classes,
            **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name."""
        return self._loss_name
