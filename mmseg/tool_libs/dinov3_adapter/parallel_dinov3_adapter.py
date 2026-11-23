"""Parallel multi-scale DINOv3 adapter for PS-Joint.

Unlike the original serial ViT-Adapter path, each Swin pyramid level is fused
with one DINOv3 stage independently. The fusion operation remains deformable
cross-attention followed by a convolutional FFN.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .dinov3_adapter import DropPath, get_reference_points
from .ms_deform_attn import MSDeformAttn


class SingleScaleConvFFN(nn.Module):
    """ViT-Adapter ConvFFN specialized for one spatial feature level."""

    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, tokens, height, width):
        batch = tokens.shape[0]
        features = self.fc1(tokens)
        features = features.transpose(1, 2).reshape(
            batch, -1, height, width).contiguous()
        features = self.dwconv(features)
        features = features.flatten(2).transpose(1, 2)
        features = self.drop(self.act(features))
        return self.drop(self.fc2(features))


class ParallelScaleExtractor(nn.Module):
    """One independent deformable cross-attention and ConvFFN branch."""

    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 deform_ratio=0.5,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 drop=0.0,
                 drop_path=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.value_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=1,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn_norm = norm_layer(dim)
            self.ffn = SingleScaleConvFFN(
                dim=dim, hidden_dim=int(dim * cffn_ratio), drop=drop)
            self.drop_path = (
                DropPath(drop_path) if drop_path > 0 else nn.Identity())

    def forward(self, query, value, query_height, query_width,
                value_height, value_width):
        def _inner_forward(query_tokens, value_tokens):
            spatial_shapes = torch.as_tensor(
                [(value_height, value_width)],
                dtype=torch.long,
                device=query_tokens.device)
            level_start_index = spatial_shapes.new_zeros((1,))
            reference_points = get_reference_points(
                [(query_height, query_width)], query_tokens.device)
            fused = self.attn(
                self.query_norm(query_tokens),
                reference_points,
                self.value_norm(value_tokens),
                spatial_shapes,
                level_start_index,
                None)
            query_tokens = query_tokens + fused
            if self.with_cffn:
                query_tokens = query_tokens + self.drop_path(
                    self.ffn(
                        self.ffn_norm(query_tokens),
                        query_height,
                        query_width))
            return query_tokens

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(
                _inner_forward, query, value, use_reentrant=False)
        return _inner_forward(query, value)


class ParallelDINOv3Adapter(nn.Module):
    """Fuse four Swin scales with four DINOv3 stages without stage chaining."""

    def __init__(self,
                 backbone,
                 interaction_indexes=(2, 5, 8, 11),
                 swin_channels=(96, 192, 384, 768),
                 n_points=4,
                 deform_num_heads=6,
                 drop_path_rate=0.3,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=0.5,
                 add_vit_feature=True,
                 with_cp=False):
        super().__init__()
        if len(interaction_indexes) != 4 or len(swin_channels) != 4:
            raise ValueError('The parallel adapter requires exactly four stages.')
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.interaction_indexes = tuple(interaction_indexes)
        self.patch_size = self.backbone.patch_size
        self.add_vit_feature = add_vit_feature
        dim = self.backbone.embed_dim

        self.swin_projections = nn.ModuleList([
            nn.Conv2d(channels, dim, kernel_size=1)
            for channels in swin_channels
        ])
        # The 1/4 branch is fused at 1/8 to control query count and memory.
        self.scale1_patch_embed = nn.Conv2d(
            dim, dim, kernel_size=2, stride=2)
        self.scale1_restore = nn.ConvTranspose2d(
            dim, dim, kernel_size=2, stride=2)
        self.level_embed = nn.Parameter(torch.zeros(4, dim))
        self.interactions = nn.ModuleList([
            ParallelScaleExtractor(
                dim=dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                deform_ratio=deform_ratio,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                drop_path=drop_path_rate,
                with_cp=with_cp)
            for _ in range(4)
        ])
        self.output_norms = nn.ModuleList([
            nn.SyncBatchNorm(dim) for _ in range(4)
        ])
        # Initialize only newly added adapter modules. Calling self.apply here
        # would also overwrite the loaded DINOv3 pretrained backbone.
        self.swin_projections.apply(self._init_weights)
        self.scale1_patch_embed.apply(self._init_weights)
        self.scale1_restore.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.interactions.apply(self._init_deform_weights)
        self.output_norms.apply(self._init_weights)
        nn.init.normal_(self.level_embed)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d,
                                 nn.SyncBatchNorm)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = (module.kernel_size[0] * module.kernel_size[1] *
                       module.out_channels // module.groups)
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    @staticmethod
    def _init_deform_weights(module):
        if isinstance(module, MSDeformAttn):
            module._reset_parameters()

    @staticmethod
    def _tokens_to_map(tokens, height, width):
        batch, _, channels = tokens.shape
        return tokens.transpose(1, 2).reshape(
            batch, channels, height, width).contiguous()

    def forward(self, image, spa_prior):
        if spa_prior is None or len(spa_prior) != 4:
            raise ValueError('Parallel adapter requires four Swin feature maps.')

        with torch.autocast(
                device_type=image.device.type,
                dtype=torch.bfloat16,
                enabled=image.is_cuda):
            with torch.no_grad():
                dino_stages = self.backbone.get_intermediate_layers(
                    image,
                    n=self.interaction_indexes,
                    return_class_token=True)

        projected = [
            projection(feature)
            for projection, feature in zip(self.swin_projections, spa_prior)
        ]
        scale1_skip = projected[0]
        projected[0] = self.scale1_patch_embed(projected[0])

        token_height = image.shape[-2] // self.patch_size
        token_width = image.shape[-1] // self.patch_size
        expected_tokens = token_height * token_width
        fused_maps = []
        for index, (feature, stage_output, interaction) in enumerate(zip(
                projected, dino_stages, self.interactions)):
            dino_tokens, _ = stage_output
            if dino_tokens.shape[1] != expected_tokens:
                raise ValueError(
                    'DINO token count does not match the input patch grid: '
                    f'{dino_tokens.shape[1]} versus {expected_tokens}.')
            height, width = feature.shape[-2:]
            query = feature.flatten(2).transpose(1, 2)
            query = query + self.level_embed[index]
            dino_tokens = dino_tokens.to(dtype=query.dtype)
            fused = interaction(
                query=query,
                value=dino_tokens,
                query_height=height,
                query_width=width,
                value_height=token_height,
                value_width=token_width)
            fused_map = self._tokens_to_map(fused, height, width)
            fused_maps.append(fused_map)

        restored_scale1 = self.scale1_restore(fused_maps[0])
        if restored_scale1.shape[-2:] != scale1_skip.shape[-2:]:
            restored_scale1 = F.interpolate(
                restored_scale1,
                size=scale1_skip.shape[-2:],
                mode='bilinear',
                align_corners=False)
        fused_maps[0] = restored_scale1 + scale1_skip

        # Follow the original ViT-Adapter residual placement: form the whole
        # adapter pyramid first, then add the four raw DINOv3 stage maps.
        if self.add_vit_feature:
            for index, (stage_output, fused_map) in enumerate(zip(
                    dino_stages, fused_maps)):
                stage_tokens, _ = stage_output
                vit_map = self._tokens_to_map(
                    stage_tokens.to(dtype=fused_map.dtype),
                    token_height,
                    token_width)
                vit_map = F.interpolate(
                    vit_map,
                    size=fused_map.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
                fused_maps[index] = fused_map + vit_map

        return {
            str(index + 1): norm(feature)
            for index, (norm, feature) in enumerate(zip(
                self.output_norms, fused_maps))
        }
