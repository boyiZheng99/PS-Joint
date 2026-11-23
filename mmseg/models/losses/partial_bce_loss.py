# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class PartialBCELoss(nn.Module):
    """Partial-label BCE loss for multi-label classification."""

    def __init__(self,
                 margin: float = 0.0,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 loss_name: str = 'loss_cls') -> None:
        super().__init__()
        self.margin = margin
        self.loss_weight = loss_weight
        self.reduction = reduction
        self._loss_name = loss_name

    @property
    def loss_name(self) -> str:
        return self._loss_name

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        input = input.float()
        target = target.float()

        tmp = target.clone()
        target = torch.where(tmp == 0, torch.full_like(tmp, -1.0), target)
        target = torch.where(tmp == -1, torch.zeros_like(tmp), target)

        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        positive_loss = F.binary_cross_entropy_with_logits(
            input, target, reduction='none')
        negative_loss = F.binary_cross_entropy_with_logits(
            -input, -target, reduction='none')

        loss = positive_mask * positive_loss + negative_mask * negative_loss

        if self.reduction == 'mean':
        # 如果 loss 还没有乘过 mask，这里要乘一下
            valid_mask = positive_mask + negative_mask
            class_denom = valid_mask.sum(dim=0).clamp(min=1.0)  # [C]
            class_loss = loss.sum(dim=0) / class_denom          # [C]
            valid_class = valid_mask.sum(dim=0) > 0             # [C]
            if valid_class.any():
                loss = class_loss[valid_class].mean()
            else:
                loss = loss.sum() * 0.0

        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss * self.loss_weight
