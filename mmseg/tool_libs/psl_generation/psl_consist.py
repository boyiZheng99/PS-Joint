import torch



def build_masks(pseudo1, pseudo2, ignore_index=255):
    intra_consist_mask = ((pseudo1 == 1) & (pseudo2 == 1)).float()
    # ignore_mask = (pseudo1 == ignore_index) | (pseudo2 == ignore_index)
    inter_diff_mask = (((pseudo1 == 1) & (pseudo2 == 0)) | ((pseudo1 == 0) & (pseudo2 == 1))).float()
    # mask_ignore_or_diff = torch.zeros_like(pseudo1, dtype=torch.long)
    # mask_ignore_or_diff[diff_mask] = 1
    # mask_ignore_or_diff[ignore_mask] = ignore_index

    return intra_consist_mask, inter_diff_mask