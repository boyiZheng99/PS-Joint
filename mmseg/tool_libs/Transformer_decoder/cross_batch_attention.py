import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossBatchChannelAttention(nn.Module):
    def __init__(self, hw_dim, dim_qk=128, dim_v=None):
        super().__init__()
        if dim_v is None:
            dim_v = hw_dim

        self.q_proj = nn.Linear(hw_dim, dim_qk, bias=False)
        self.k_proj = nn.Linear(hw_dim, dim_qk, bias=False)
        self.v_proj = nn.Linear(hw_dim, dim_v, bias=False)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W
        T = B * C

        tokens = x.view(B, C, N).reshape(T, N)   # [T, N]

        q = self.q_proj(tokens)                   # [T, d]
        k = self.k_proj(tokens)                   # [T, d]
        v = self.v_proj(tokens)                   # [T, N] or [T, dv]

        attn = torch.matmul(q, k.t()) / (q.shape[-1] ** 0.5)   # [T, T]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)               # [T, N] or [T, dv]

        if out.shape[-1] != N:
            raise ValueError("如果 dim_v != H*W，后面不能直接 reshape 回 B,C,H,W")

        out = out.view(B, C, H, W)
        return out