"""Serial pyramid DINOv3 adapter for PS-Joint.

The adapter carries one spatial stream through four DINOv3 stages:

    Swin 1/4 -> downsample -> CA_1 -> + Swin 1/8 -> CA_2
      -> downsample -> + Swin 1/16 -> CA_3
      -> downsample -> + Swin 1/32 -> CA_4.

The output of each cross-attention block is saved *before* the next Swin
residual is added.  These four outputs form the feature pyramid consumed by
the HRDecoder.  CA_1 is restored from 1/8 to 1/4 for the first pyramid level.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_adapter import DropPath
from .parallel_dinov3_adapter import ParallelScaleExtractor


class DistributionUncertainty(nn.Module):
    """Model uncertainty in channel-wise feature statistics.

    This is the DSU operation from *Uncertainty Modeling for
    Out-of-Distribution Generalization* (ICLR 2022). During training it
    samples the per-instance channel mean and standard deviation from
    Gaussian distributions whose variances are estimated across the batch.
    Evaluation is deterministic because DSU becomes an identity mapping.

    Statistics are accumulated in FP32 so the operation remains stable when
    the adapter is trained with AMP/bfloat16.
    """

    def __init__(self, p=0.5, eps=1e-6, factor=1.0):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f'DSU probability must be in [0, 1], got {p}.')
        self.p = float(p)
        self.eps = float(eps)
        self.factor = float(factor)

    def _reparameterize(self, mean, std):
        return mean + torch.randn_like(std) * self.factor * std

    def forward(self, feature):
        if not self.training or self.p == 0.0:
            return feature
        if torch.rand((), device=feature.device) > self.p:
            return feature

        input_dtype = feature.dtype
        feature_fp32 = feature.float()
        mean = feature_fp32.mean(dim=(2, 3))
        std = (feature_fp32.var(dim=(2, 3), unbiased=False) + self.eps).sqrt()

        # DSU estimates uncertainty of feature statistics across the batch.
        # unbiased=False also keeps this operation valid for batch size one.
        mean_uncertainty = (
            mean.var(dim=0, keepdim=True, unbiased=False) + self.eps
        ).sqrt().expand_as(mean)
        std_uncertainty = (
            std.var(dim=0, keepdim=True, unbiased=False) + self.eps
        ).sqrt().expand_as(std)

        sampled_mean = self._reparameterize(mean, mean_uncertainty)
        sampled_std = self._reparameterize(std, std_uncertainty)
        normalized = (
            feature_fp32 - mean[:, :, None, None]
        ) / std[:, :, None, None]
        perturbed = (
            normalized * sampled_std[:, :, None, None]
            + sampled_mean[:, :, None, None]
        )
        return perturbed.to(dtype=input_dtype)


class FeatureConditionedDistributionUncertainty(nn.Module):
    """Predict the perturbation distribution from each input feature.

    Unlike the original DSU, this module does not estimate uncertainty from
    variation among samples in a mini-batch. A small predictor consumes the
    current sample's channel-wise spatial mean and standard deviation and
    predicts Gaussian distributions for a normalized mean shift and a
    log-standard-deviation shift. Reparameterized samples are then used to
    construct the perturbed feature statistics.

    The multiplicative statistic is modeled in log space, which guarantees a
    positive sampled standard deviation.
    """

    def __init__(self,
                 channels,
                 p=0.5,
                 eps=1e-6,
                 factor=1.0,
                 hidden_ratio=0.25,
                 initial_noise_scale=-4.0):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f'Feature-conditioned DSU probability must be in [0, 1], '
                f'got {p}.')
        hidden_channels = max(16, int(channels * hidden_ratio))
        self.p = float(p)
        self.eps = float(eps)
        self.factor = float(factor)
        self.initial_noise_scale = float(initial_noise_scale)
        self.predictor = nn.Sequential(
            nn.LayerNorm(2 * channels),
            nn.Linear(2 * channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 4 * channels),
        )
        # Start close to an identity mapping. The distribution scales begin
        # at softplus(-4) rather than softplus(0), avoiding strong noise at
        # the start of training while leaving all parameters learnable.
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def _sample(self, location, raw_scale):
        scale = F.softplus(
            raw_scale + self.initial_noise_scale) + self.eps
        return location + torch.randn_like(scale) * self.factor * scale

    def forward(self, feature):
        if not self.training or self.p == 0.0:
            return feature
        if torch.rand((), device=feature.device) > self.p:
            return feature

        input_dtype = feature.dtype
        feature_fp32 = feature.float()
        mean = feature_fp32.mean(dim=(2, 3))
        std = (feature_fp32.var(
            dim=(2, 3), unbiased=False) + self.eps).sqrt()

        prediction = self.predictor(torch.cat((mean, std), dim=1))
        (mean_noise_location, mean_noise_raw_scale,
         logstd_noise_location, logstd_noise_raw_scale) = prediction.chunk(
             4, dim=1)

        normalized_mean_shift = self._sample(
            mean_noise_location, mean_noise_raw_scale)
        logstd_shift = self._sample(
            logstd_noise_location, logstd_noise_raw_scale)

        sampled_mean = mean + normalized_mean_shift * std
        sampled_std = std * torch.exp(logstd_shift.clamp(min=-2.0, max=2.0))
        normalized = (
            feature_fp32 - mean[:, :, None, None]
        ) / std[:, :, None, None]
        perturbed = (
            normalized * sampled_std[:, :, None, None]
            + sampled_mean[:, :, None, None]
        )
        return perturbed.to(dtype=input_dtype)


class SerialPyramidDINOv3Adapter(nn.Module):
    """Fuse Swin and DINOv3 with a four-stage serial feature stream."""

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
                 dsu_p=0.0,
                 dsu_eps=1e-6,
                 dsu_factor=1.0,
                 dsu_type='batch',
                 dsu_hidden_ratio=0.25,
                 dsu_initial_noise_scale=-4.0,
                 with_cp=False):
        super().__init__()
        if len(interaction_indexes) != 4 or len(swin_channels) != 4:
            raise ValueError('The serial pyramid adapter requires four stages.')

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
        # The first Swin level is reduced from 1/4 to 1/8 before CA_1.
        self.first_patch_embed = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.first_restore = nn.ConvTranspose2d(
            dim, dim, kernel_size=2, stride=2)
        # CA_2 -> CA_3 and CA_3 -> CA_4 downsampling transitions.
        self.downsamples = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
        ])
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
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                with_cp=with_cp)
            for _ in range(4)
        ])
        # DSU is applied immediately after every cross-attention/CFFN block.
        # Its output is both propagated to the next serial stage and retained
        # in the final multi-level feature pyramid.
        if dsu_type == 'batch':
            self.dsu_layers = nn.ModuleList([
                DistributionUncertainty(
                    p=dsu_p, eps=dsu_eps, factor=dsu_factor)
                for _ in range(4)
            ])
        elif dsu_type == 'feature':
            self.dsu_layers = nn.ModuleList([
                FeatureConditionedDistributionUncertainty(
                    channels=dim,
                    p=dsu_p,
                    eps=dsu_eps,
                    factor=dsu_factor,
                    hidden_ratio=dsu_hidden_ratio,
                    initial_noise_scale=dsu_initial_noise_scale)
                for _ in range(4)
            ])
        else:
            raise ValueError(
                f'Unsupported dsu_type {dsu_type!r}; expected "batch" or '
                '"feature".')
        self.output_norms = nn.ModuleList([
            nn.SyncBatchNorm(dim) for _ in range(4)
        ])

        self.swin_projections.apply(self._init_weights)
        self.first_patch_embed.apply(self._init_weights)
        self.first_restore.apply(self._init_weights)
        self.downsamples.apply(self._init_weights)
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
        # ParallelScaleExtractor owns MSDeformAttn.  Its constructor's reset
        # has already initialized it; this hook is retained for consistency.
        if hasattr(module, '_reset_parameters'):
            module._reset_parameters()

    @staticmethod
    def _tokens_to_map(tokens, height, width):
        batch, _, channels = tokens.shape
        return tokens.transpose(1, 2).reshape(
            batch, channels, height, width).contiguous()

    @staticmethod
    def _map_to_tokens(feature):
        return feature.flatten(2).transpose(1, 2).contiguous()

    def _cross_attention(self, feature, dino_tokens, interaction, level):
        height, width = feature.shape[-2:]
        query = self._map_to_tokens(feature) + self.level_embed[level]
        fused = interaction(
            query=query,
            value=dino_tokens.to(dtype=query.dtype),
            query_height=height,
            query_width=width,
            value_height=self._token_height,
            value_width=self._token_width)
        return self._tokens_to_map(fused, height, width)

    def forward(self, image, spa_prior):
        if spa_prior is None or len(spa_prior) != 4:
            raise ValueError('Serial pyramid adapter requires four Swin maps.')

        with torch.autocast(
                device_type=image.device.type,
                dtype=torch.bfloat16,
                enabled=image.is_cuda):
            with torch.no_grad():
                dino_stages = self.backbone.get_intermediate_layers(
                    image,
                    n=self.interaction_indexes,
                    return_class_token=True)

        self._token_height = image.shape[-2] // self.patch_size
        self._token_width = image.shape[-1] // self.patch_size
        expected_tokens = self._token_height * self._token_width
        dino_tokens = []
        for stage_tokens, _ in dino_stages:
            if stage_tokens.shape[1] != expected_tokens:
                raise ValueError(
                    'DINO token count does not match the input patch grid: '
                    f'{stage_tokens.shape[1]} versus {expected_tokens}.')
            dino_tokens.append(stage_tokens)

        swin = [projection(feature) for projection, feature in zip(
            self.swin_projections, spa_prior)]

        # CA_1: 1/4 Swin is patched to 1/8.  Save CA_1 before + Swin 1/8.
        state = self.first_patch_embed(swin[0])
        ca1 = self._cross_attention(state, dino_tokens[0],
                                    self.interactions[0], 0)
        ca1 = self.dsu_layers[0](ca1)
        # CA_2 at 1/8; CA_1 is the first saved feature, restored below.
        state = ca1 + swin[1]
        ca2 = self._cross_attention(state, dino_tokens[1],
                                    self.interactions[1], 1)
        ca2 = self.dsu_layers[1](ca2)
        # CA_3 at 1/16.
        state = self.downsamples[0](ca2) + swin[2]
        ca3 = self._cross_attention(state, dino_tokens[2],
                                    self.interactions[2], 2)
        ca3 = self.dsu_layers[2](ca3)
        # CA_4 at 1/32.
        state = self.downsamples[1](ca3) + swin[3]
        ca4 = self._cross_attention(state, dino_tokens[3],
                                    self.interactions[3], 3)
        ca4 = self.dsu_layers[3](ca4)

        ca1 = self.first_restore(ca1)
        if ca1.shape[-2:] != swin[0].shape[-2:]:
            ca1 = F.interpolate(
                ca1, size=swin[0].shape[-2:], mode='bilinear',
                align_corners=False)
        features = [ca1, ca2, ca3, ca4]

        # Follow the original ViT-Adapter residual placement: form the whole
        # serial pyramid first, then add the four raw DINOv3 stage maps.
        if self.add_vit_feature:
            for index, (stage_tokens, feature) in enumerate(zip(
                    dino_tokens, features)):
                vit_map = self._tokens_to_map(
                    stage_tokens.to(dtype=feature.dtype),
                    self._token_height,
                    self._token_width)
                vit_map = F.interpolate(
                    vit_map,
                    size=feature.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
                features[index] = feature + vit_map

        return {
            str(index + 1): norm(feature)
            for index, (norm, feature) in enumerate(zip(
                self.output_norms, features))
        }
