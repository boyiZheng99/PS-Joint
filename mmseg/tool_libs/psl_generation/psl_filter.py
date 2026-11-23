import torch

@torch.no_grad()
def psl_filter(a, b, ignore_index=255):
    assert a.shape == b.shape

    both_one = (a == 1) & (b == 1)
    both_zero = (a == 0) & (b == 0)

    out = torch.full_like(a, ignore_index)

    out[both_one] = 1
    out[both_zero] = 0

    return out

import torch


@torch.no_grad()
def uncertainty_filter(pseudo_label, uncertainty, threshold=0.75, ignore_index=255):
    assert pseudo_label.shape == uncertainty.shape

    out = pseudo_label.clone()
    mask = (pseudo_label != ignore_index) & (uncertainty > threshold)
    out[mask] = ignore_index

    return out