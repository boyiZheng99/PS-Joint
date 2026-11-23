import torch
import numpy as np

@torch.no_grad()
def generate_pseudo_labels(
    pred,                 
    gt,                   
    thresh: float = 0.95,  
    ignore_seg_index: int = 255,
    ignore_cls_index: int = -1,
    logits: bool = True, 
    mask = None,
    uncertainty_ratio=0.75,
    uncertain_button=False,
):
    """
    Return:
      pseudo labels
    """
    assert pred.shape == gt.shape, f"pred/gt 通道维不一致: {pred.shape} vs {gt.shape}"
    device = pred.device


    prob = torch.sigmoid(pred) if logits else pred

    # print(prob)

    high_conf = prob > thresh
    low_conf = prob < (1 - thresh)
    # uncertain = ~(high_conf | low_conf)

    uncertainty = -(
        prob * torch.log(prob + 1e-6) +
        (1 - prob) * torch.log(1 - prob + 1e-6)
    )

    uncertainty_threshold = uncertainty_ratio * np.log(2)
    certainty_mask = uncertainty < uncertainty_threshold

    # ---------------- Segmentation pseudo labels  ----------------
    if pred.dim() == 4:
        N, C, H, W = pred.shape

        # 默认全 ignore
        pseudo = torch.full(
            gt.shape, ignore_seg_index, dtype=torch.uint8, device=gt.device
        )

        # 找出无标签通道：只有这些通道才生成伪标签
        pseudo_channel = (gt == ignore_seg_index).flatten(2).all(dim=-1)   # (N, C)
        pseudo_channel_mask = pseudo_channel.unsqueeze(-1).unsqueeze(-1)    # (N, C, 1, 1)


        if not uncertain_button:
            # 只在无标签通道生成伪标签
            pseudo[pseudo_channel_mask & high_conf] = 1
            pseudo[pseudo_channel_mask & low_conf] = 0
        else:
            pseudo[pseudo_channel_mask & high_conf & certainty_mask] = 1
            pseudo[pseudo_channel_mask & low_conf & certainty_mask] = 0

        if mask is not None:
            pseudo[:, mask, :, :] = 255


        return pseudo.to(device)

    # ---------------- classification pseudo labels ----------------
    elif pred.dim() == 2:
        N, C = pred.shape

        pseudo = torch.full(
            gt.shape, ignore_cls_index, device=gt.device
        )

        pseudo_channel_mask = (gt == ignore_cls_index)  # (N, C)

        if not uncertain_button:
            pseudo[pseudo_channel_mask & high_conf] = 1
            pseudo[pseudo_channel_mask & low_conf] = 0

        else:
            pseudo[pseudo_channel_mask & high_conf & certainty_mask] = 1
            pseudo[pseudo_channel_mask & low_conf & certainty_mask] = 0

        if mask is not None:
            pseudo[:,mask] = -1


        # pseudo = torch.full(gt.shape,-1, device=gt.device)
        # unknown = (gt == ignore_cls_index)
        # all_unknown_sample = unknown.all(dim=1)       # 哪些样本全是 -1
        # valid_sample_mask = ~all_unknown_sample       # 样本中至少一个有效标签
        # valid_sample_mask_2d = valid_sample_mask.view(N, 1)  # (N,1)
        # unknown = unknown & valid_sample_mask_2d
        # pseudo[unknown & high_conf] = 1
        # pseudo[unknown & low_conf] = 0
        # pseudo[unknown & uncertain] = ignore_cls_index

        # print(pseudo)

        return pseudo.to(device)

    else:
        raise ValueError(f"不支持的 pred 维度: {pred.dim()} (期望 2 或 4)")



def fuse_pseudo_label_with_mask_fast(p_foundation, p_teacher, u_foundation, u_teacher, gt, uncertainty_threshold=1.0, ignore_index=255, psl_threshold=0.5):
    trust_foundation = (u_foundation <= uncertainty_threshold).float()  
    trust_teacher = (u_teacher <= uncertainty_threshold).float()
    trust_mask = ((trust_foundation + trust_teacher) > 0).float()

    both_trust = trust_foundation * trust_teacher
    only_teacher = (1.0 - trust_foundation) * trust_teacher
    only_foundation = trust_foundation * (1.0 - trust_teacher)

    w_foundation = torch.exp(-u_foundation)
    w_teacher = torch.exp(-u_teacher)
    w_sum = w_foundation + w_teacher + 1e-8
    w_foundation /= w_sum
    w_teacher = 1.0 - w_foundation
    fused = w_foundation * p_foundation + w_teacher * p_teacher

    pseudo_fused = (
        both_trust * fused +
        only_teacher * p_teacher +
        only_foundation * p_foundation
    )

    pseudo_label = torch.full_like(pseudo_fused, ignore_index)

    pseudo_label[pseudo_fused > psl_threshold] = 1
    pseudo_label[pseudo_fused < (1 - psl_threshold)] = 0

    valid_gt_mask = (gt != ignore_index).any(dim=1, keepdim=True)

    # 在这些位置，把 pseudo 全部设为 ignore
    pseudo_label[valid_gt_mask.expand_as(pseudo_label)] = ignore_index
   
    return pseudo_label

@torch.no_grad()
def uncertainty_filter_pseudo_label(
    psl_foundation,
    psl_teacher,
    u_foundation,
    u_teacher,
    uncertainty_threshold=1.0,
    ignore_index=255,
    high_uncertain_with_ignore_index=False,
    filter_both_uncertain=True,
):
    """
    Args:
        psl_foundation: (N, C, H, W), 值一般为 {0,1,ignore_index}
        psl_teacher:    (N, C, H, W), 值一般为 {0,1,ignore_index}
        u_foundation:   (N, C, H, W) 或可广播到该形状
        u_teacher:      (N, C, H, W) 或可广播到该形状
    Returns:
        mask: (N, C, H, W)
    """

    # 输出先初始化为0
    mask = torch.zeros_like(psl_foundation)

    # 1) 只要任意一个是 ignore_index -> ignore_index
    ignore_mask = (psl_foundation == ignore_index) | (psl_teacher == ignore_index)
    mask[ignore_mask] = ignore_index

    # 2) 两个都为1 -> 1
    both_one_mask = (psl_foundation == 1) & (psl_teacher == 1)
    mask[both_one_mask] = 1

    # 2) 两个都为0 -> 0
    both_zero_mask = (psl_foundation == 0) & (psl_teacher == 0)
    mask[both_zero_mask] = 0

    # 3) 一个为0，一个为1
    diff_mask = (
        ((psl_foundation == 0) & (psl_teacher == 1)) |
        ((psl_foundation == 1) & (psl_teacher == 0))
    )

    # 在 diff 的位置上，看不确定性
    if filter_both_uncertain:
        both_uncertain = ((u_foundation > uncertainty_threshold)
                          & (u_teacher > uncertainty_threshold))
    else:
        # Keep uncertainty as a relative ranking signal.  Even when both
        # teachers have high entropy, select the lower-entropy prediction
        # instead of replacing the pseudo label with ignore_index.
        both_uncertain = torch.zeros_like(diff_mask)
    diff_and_both_uncertain = diff_mask & both_uncertain

    # 3.1 两边都不可信 -> 0
    if not high_uncertain_with_ignore_index:
        mask[diff_and_both_uncertain] = 0
    else:
        mask[diff_and_both_uncertain] = ignore_index

    # 3.2 否则选不确定性更小的那个伪标签
    diff_and_not_both_uncertain = diff_mask & (~both_uncertain)

    choose_foundation = diff_and_not_both_uncertain & (u_foundation <= u_teacher)
    choose_teacher = diff_and_not_both_uncertain & (u_foundation > u_teacher)
    

    mask[choose_foundation] = psl_foundation[choose_foundation]
    mask[choose_teacher] = psl_teacher[choose_teacher]
    return mask




def uncertainty_filter_pseudo_label_new(
    psl_foundation,
    psl_teacher,
    u_foundation,
    u_teacher,
    uncertainty_threshold=1.0,
    ignore_index=255,
    high_uncertain_with_ignore_index=False
):
    """
    Args:
        psl_foundation: (N, C, H, W), 值一般为 {0,1,ignore_index}
        psl_teacher:    (N, C, H, W), 值一般为 {0,1,ignore_index}
        u_foundation:   (N, C, H, W) 或可广播到该形状
        u_teacher:      (N, C, H, W) 或可广播到该形状
    Returns:
        mask: (N, C, H, W)
    """

    while u_foundation.dim() < psl_foundation.dim():
        u_foundation = u_foundation.unsqueeze(-1)
    while u_teacher.dim() < psl_teacher.dim():
        u_teacher = u_teacher.unsqueeze(-1)
    u_foundation = u_foundation.expand_as(psl_foundation)
    u_teacher = u_teacher.expand_as(psl_teacher)

    mask = torch.full_like(psl_foundation, ignore_index)

    valid_mask = ((psl_foundation != ignore_index)
                  & (psl_teacher != ignore_index))
    both_uncertain = ((u_foundation > uncertainty_threshold)
                      & (u_teacher > uncertainty_threshold))
    selectable = valid_mask & (~both_uncertain)

    choose_foundation = selectable & (u_foundation <= u_teacher)
    choose_teacher = selectable & (u_foundation > u_teacher)

    mask[choose_foundation] = psl_foundation[choose_foundation]
    mask[choose_teacher] = psl_teacher[choose_teacher]

    if not high_uncertain_with_ignore_index:
        mask[valid_mask & both_uncertain] = 0

    return mask


@torch.no_grad()
def generate_soft_pseudo_labels(
    pred,
    gt,
    thresh: float = 0.95,
    ignore_seg_index: int = 255,
    ignore_cls_index: int = -1,
    logits: bool = True,
    mask=None,
    uncertainty_ratio=0.75,
    uncertain_button=False,
    soft_label=True,
    soft_class_idx=None, 
):
    """
    Return:
      pseudo labels

    hard label:
        confident positive -> 1
        confident negative -> 0

    soft label:
        confident positions -> sigmoid(pred)
        ignored positions -> ignore_index
    """
    assert pred.shape == gt.shape, f"pred/gt 通道维不一致: {pred.shape} vs {gt.shape}"

    device = pred.device
    prob = torch.sigmoid(pred) if logits else pred

    high_conf = prob > thresh
    low_conf = prob < (1 - thresh)

    confident_mask = high_conf | low_conf

    uncertainty = -(
        prob * torch.log(prob + 1e-6) +
        (1 - prob) * torch.log(1 - prob + 1e-6)
    )

    uncertainty_threshold = uncertainty_ratio * np.log(2)
    certainty_mask = uncertainty < uncertainty_threshold

    if uncertain_button:
        confident_mask = confident_mask & certainty_mask

    # ---------------- Segmentation pseudo labels ----------------
    if pred.dim() == 4:
        N, C, H, W = pred.shape

        pseudo = torch.full(
            gt.shape,
            float(ignore_seg_index),
            dtype=prob.dtype,
            device=gt.device
        )

        # 只有 gt 全为 ignore 的通道才生成伪标签
        pseudo_channel = (gt == ignore_seg_index).flatten(2).all(dim=-1)  # [N, C]
        pseudo_channel_mask = pseudo_channel.unsqueeze(-1).unsqueeze(-1)  # [N, C, 1, 1]

        valid_mask = pseudo_channel_mask & confident_mask

        if soft_label:
            pseudo[valid_mask] = prob[valid_mask]

        else:
            pseudo[pseudo_channel_mask & high_conf] = 1
            pseudo[pseudo_channel_mask & low_conf] = 0

            if soft_class_idx is not None:
                cls_valid_mask = valid_mask[:, soft_class_idx, :, :]  # [N, H, W]
                pseudo[:, soft_class_idx, :, :][cls_valid_mask] = \
                    prob[:, soft_class_idx, :, :][cls_valid_mask]

        if mask is not None:
            pseudo[:, mask, :, :] = ignore_seg_index

        return pseudo.to(device)

    # ---------------- Classification pseudo labels ----------------
    elif pred.dim() == 2:
        N, C = pred.shape

        if soft_label:
            pseudo = torch.full(
                gt.shape,
                float(ignore_cls_index),
                dtype=prob.dtype,
                device=gt.device
            )
        else:
            pseudo = torch.full(
                gt.shape,
                ignore_cls_index,
                device=gt.device
            )

        pseudo_channel_mask = gt == ignore_cls_index  # [N, C]
        valid_mask = pseudo_channel_mask & confident_mask

        if soft_label:
            pseudo[valid_mask] = prob[valid_mask]
        else:
            pseudo[pseudo_channel_mask & high_conf] = 1
            pseudo[pseudo_channel_mask & low_conf] = 0

        if mask is not None:
            pseudo[:, mask] = ignore_cls_index

        return pseudo.to(device)

    else:
        raise ValueError(f"不支持的 pred 维度: {pred.dim()} (期望 2 或 4)")



@torch.no_grad()
def hard_psl_to_soft_psl(
    logits,
    psl,
    channel_indices=None,
    ignore_index=255,
):
    """
    Args:
        logits: [B, C, H, W]
        psl: [B, C, H, W]
        channel_indices:
            int 或 list[int]，指定要替换的通道
        ignore_index:
         ignore label，例如 255
        replace_all:
            True 时替换所有通道的非 ignore 位置
            False 时只替换 channel_indices 指定通道

    Return:
        psl_new: float tensor
    """
    assert logits.shape == psl.shape, \
        f'logits shape {logits.shape} must match psl shape {psl.shape}'

    prob = torch.sigmoid(logits)
    psl_new = psl.clone().float()

    if channel_indices is None:
        valid_mask = psl != ignore_index  # [B, C, H, W]

        psl_new = torch.where(
            valid_mask,
            prob.to(psl_new.dtype),
            psl_new
        )

    else:
        if channel_indices is None:
            raise ValueError(
                'channel_indices must be provided when replace_all=False'
            )

        if isinstance(channel_indices, int):
            channel_indices = [channel_indices]

        for c in channel_indices:
            valid_mask = psl[:, c, :, :] != ignore_index  # [B, H, W]

            psl_new[:, c, :, :] = torch.where(
                valid_mask,
                prob[:, c, :, :].to(psl_new.dtype),
                psl_new[:, c, :, :]
            )

    return psl_new
