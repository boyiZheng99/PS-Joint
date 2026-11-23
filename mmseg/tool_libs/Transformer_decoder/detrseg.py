import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch import Tensor
from typing import List, Optional


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.norm(self.proj(x))
        return x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)       #让每个像素知道自己在第几行和第几列
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale   #让每个像素的位置落在(0,2pi)之间
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  #x_embed形状是(B,H,W)，经过这个变化后，x_embed的形状是(B,H,W,1),再除以dim_t，形状变为(B,H,W,64)
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)    #(B,H,W,64)->(B,H,W,32,2)->(B,H,W,64)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  #(B,H,W,128)->(b,128,H,W)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
        

class DETRsegm(nn.Module):
    def __init__(self,
                text_dim,
                nheads,
                # feature_map_size,
                # patch_size,
                seg_channels,
                query_num = 11,
                down_ratio=1
                ):
        super().__init__()

        # self.hidden_dim = hidden_dim
        self.nheads = nheads


        self.cls_proj = nn.Linear(text_dim, seg_channels)
    
        self.bbox_attention = MHAttentionMap(seg_channels, seg_channels, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(seg_channels + nheads*query_num, seg_channels)

        if down_ratio!=1:
            self.down = nn.Conv2d(seg_channels, seg_channels, kernel_size=down_ratio, stride=down_ratio)

        self.down_ratio = down_ratio
        # self.last_attention_map = None

    def forward(self, seg_feat, cls_feat):


        if self.down_ratio!=1:
            patched_seg_feat = self.down(seg_feat)
        else:
            patched_seg_feat = seg_feat

        cls_feat = self.cls_proj(cls_feat)
        

        # FIXME h_boxes takes the last one computed, keep this in mind
        bbox_mask = self.bbox_attention(cls_feat, patched_seg_feat, mask=None)  #hs:b*q*c，bbox_mask:b*q*n*H*W
        

        bbox_mask = bbox_mask.flatten(1,2)  #bbox_mask:b*(q*n)*H*W

        # self.last_attention_map = bbox_mask.detach()

        seg_feat = self.mask_head(patched_seg_feat, bbox_mask, seg_feat)  #query-guided feature augmentation

        return seg_feat

        

def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dim):
        super().__init__()

        self.lay1 = torch.nn.Conv2d(dim, fpn_dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, fpn_dim)
        self.lay2 = torch.nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, fpn_dim)
        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dim, fpn_dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpn: Tensor):

        x = torch.cat([x, bbox_mask], 1)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        cur_fpn = self.adapter1(fpn)
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay2(x)

        return x

        

class MHAttentionMap(nn.Module):  #每个query和特征图每个空间位置的attention map
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        # Independently normalize every attention head over spatial locations.
        # weights has shape [B, query, head, H, W].
        weights = F.softmax(weights.flatten(3), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights
