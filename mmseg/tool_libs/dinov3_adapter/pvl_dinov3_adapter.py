"""PVL-enhanced DINOv3 ViT-Adapter.

This module is separate from ``dinov3_adapter.py``. It keeps the original
one-way extractor design: spatial-prior tokens ``c`` query ViT tokens ``x``.
PVL uncertainty modeling is fused into the deformable attention value path,
so no separate PVL attention is run before deformable attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .dinov3_adapter import (
    DINOv3_Adapter,
    Extractor,
    InteractionBlockWithCls,
)
from .ms_deform_attn import (
    MSDA,
    MSDeformAttn,
    MSDeformAttnFunction,
    ms_deform_attn_core_pytorch,
)


class PVLMSDeformAttn(MSDeformAttn):
    """MSDeformAttn with PVL mean/variance value modeling.

    Sampling offsets and locations remain the original deformable-attention
    mechanism. PVL predicts value mean and variance, samples during training,
    and attenuates uncertain values before the same deformable aggregation.
    """

    def __init__(self, *args, beta=2.35, stochastic_train=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.stochastic_train = stochastic_train
        self.pvl_value_stats = nn.Linear(self.d_model, self.d_model * 2)
        self.pvl_gate = nn.Parameter(torch.tensor(0.0))
        nn.init.xavier_uniform_(self.pvl_value_stats.weight)
        nn.init.constant_(self.pvl_value_stats.bias, 0.0)

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        batch, query_len, _ = query.shape
        input_len = input_flatten.shape[1]
        expected_len = int((input_spatial_shapes[:, 0] *
                            input_spatial_shapes[:, 1]).sum())
        if expected_len != input_len:
            raise ValueError('input spatial shapes do not match flattened features')

        value_mu, value_logvar = self.pvl_value_stats(input_flatten).chunk(
            2, dim=-1)
        value_var = F.softplus(value_logvar) + 1e-6
        if self.training and self.stochastic_train:
            value_base = value_mu + torch.sqrt(value_var) * torch.randn_like(value_var)
        else:
            value_base = value_mu

        # PVL uncertainty suppresses unreliable value tokens before they are
        # sampled by deformable attention. This is still one attention call.
        uncertainty = value_var.mean(dim=-1, keepdim=True)
        strength = torch.sigmoid(self.pvl_gate)
        value_base = value_base * torch.exp(-self.beta * strength * uncertainty)
        value = self.value_proj(value_base)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(
            batch, input_len, self.n_heads,
            int(self.ratio * self.d_model) // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            batch, query_len, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            batch, query_len, self.n_heads,
            self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            batch, query_len, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :] +
                sampling_offsets / offset_normalizer[None, None, None, :, None, :])
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] +
                sampling_offsets / self.n_points *
                reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError('reference_points must end in 2 or 4')

        if MSDA is None:
            output = ms_deform_attn_core_pytorch(
                value, input_spatial_shapes, sampling_locations,
                attention_weights)
        else:
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index,
                sampling_locations, attention_weights, self.im2col_step)
        return self.output_proj(output)


class PVLExtractor(Extractor):
    """Original deformable extractor with PVL fused into its value path."""

    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1,
                 deform_ratio=1.0, with_cffn=True, cffn_ratio=0.25,
                 drop=0.0, drop_path=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False,
                 pvl_beta=2.35, pvl_stochastic_train=True):
        super().__init__(
            dim=dim, num_heads=num_heads, n_points=n_points,
            n_levels=n_levels, deform_ratio=deform_ratio,
            with_cffn=with_cffn, cffn_ratio=cffn_ratio, drop=drop,
            drop_path=drop_path, norm_layer=norm_layer, with_cp=with_cp)
        self.attn = PVLMSDeformAttn(
            d_model=dim, n_levels=n_levels, n_heads=num_heads,
            n_points=n_points, ratio=deform_ratio, beta=pvl_beta,
            stochastic_train=pvl_stochastic_train)


class PVLInteractionBlockWithCls(InteractionBlockWithCls):
    """Original interaction block using one PVL-deformable extractor."""

    def __init__(self, dim, num_heads=6, n_points=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0.0,
                 drop_path=0.0, with_cffn=True, cffn_ratio=0.25,
                 init_values=0.0, deform_ratio=1.0,
                 extra_extractor=False, with_cp=False, pvl_beta=2.35,
                 pvl_stochastic_train=True):
        super().__init__(
            dim=dim, num_heads=num_heads, n_points=n_points,
            norm_layer=norm_layer, drop=drop, drop_path=drop_path,
            with_cffn=with_cffn, cffn_ratio=cffn_ratio,
            init_values=init_values, deform_ratio=deform_ratio,
            extra_extractor=extra_extractor, with_cp=with_cp)
        self.extractor = PVLExtractor(
            dim=dim, num_heads=num_heads, n_points=n_points,
            n_levels=1, deform_ratio=deform_ratio,
            with_cffn=with_cffn, cffn_ratio=cffn_ratio, drop=drop,
            drop_path=drop_path, norm_layer=norm_layer, with_cp=with_cp,
            pvl_beta=pvl_beta, pvl_stochastic_train=pvl_stochastic_train)
        if self.extra_extractors is not None:
            self.extra_extractors = nn.Sequential(*[
                PVLExtractor(
                    dim=dim, num_heads=num_heads, n_points=n_points,
                    n_levels=1, deform_ratio=deform_ratio,
                    with_cffn=with_cffn, cffn_ratio=cffn_ratio, drop=drop,
                    drop_path=drop_path, norm_layer=norm_layer, with_cp=with_cp,
                    pvl_beta=pvl_beta,
                    pvl_stochastic_train=pvl_stochastic_train)
                for _ in range(2)
            ])


class PVLDINOv3_Adapter(DINOv3_Adapter):
    """DINOv3 Adapter variant with a single PVL-deformable extractor."""

    def __init__(self, backbone, interaction_indexes=(9, 19, 29, 39),
                 pretrain_size=512, conv_inplane=64, n_points=4,
                 deform_num_heads=16, drop_path_rate=0.3, init_values=0.0,
                 with_cffn=True, cffn_ratio=0.25, deform_ratio=0.5,
                 add_vit_feature=True, use_extra_extractor=True,
                 with_cp=True, spm_proj=96, pvl_beta=2.35,
                 pvl_stochastic_train=True):
        super().__init__(
            backbone=backbone, interaction_indexes=interaction_indexes,
            pretrain_size=pretrain_size, conv_inplane=conv_inplane,
            n_points=n_points, deform_num_heads=deform_num_heads,
            drop_path_rate=drop_path_rate, init_values=init_values,
            with_cffn=with_cffn, cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio, add_vit_feature=add_vit_feature,
            use_extra_extractor=use_extra_extractor, with_cp=with_cp,
            spm_proj=spm_proj)
        embed_dim = self.backbone.embed_dim
        self.interactions = nn.Sequential(*[
            PVLInteractionBlockWithCls(
                dim=embed_dim, num_heads=deform_num_heads,
                n_points=n_points,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                with_cffn=with_cffn, cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio, drop_path=drop_path_rate,
                extra_extractor=(i == len(self.interaction_indexes) - 1 and
                                 use_extra_extractor),
                with_cp=with_cp, pvl_beta=pvl_beta,
                pvl_stochastic_train=pvl_stochastic_train)
            for i in range(len(self.interaction_indexes))
        ])
        self.interactions.apply(self._init_weights)
        self.interactions.apply(self._init_deform_weights)
