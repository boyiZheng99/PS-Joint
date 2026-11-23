# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List, Optional, Tuple

from .decode_head import BaseDecodeHead
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS

from ..utils import resize
from ...utils import add_prefix

from mmseg.utils import ConfigType, SampleList

from mmseg.tool_libs.Transformer_decoder.detrseg import MaskHeadSmallConv

from mmseg.tool_libs.psl_generation.generate_psl import generate_pseudo_labels


class LightPatchEmbedding(nn.Module):
    """Lightweight 8x patch embedding used by the final PS-Joint."""

    def __init__(self, in_channels, reduced_channels, hidden_dim, patch_size):
        super().__init__()
        self.proj_down = nn.Conv2d(in_channels, reduced_channels, 1)
        self.act = nn.GELU()
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)
        self.proj_up = nn.Conv2d(reduced_channels, hidden_dim, 1)

    def forward(self, x):
        return self.proj_up(self.pool(self.act(self.proj_down(x))))


class ReparameterizedKeyCrossAttention(nn.Module):
    """PVL cross-attention with direct Gaussian key reparameterization."""

    def __init__(self, dim, num_heads, dropout=0., gate_init=0.,
                 eval_strategy='mean', mc_samples=10, eps=1e-6,
                 key_noise_clip=2.35):
        super().__init__()
        if dim % num_heads:
            raise ValueError('dim must be divisible by num_heads')
        if eval_strategy not in ('mean', 'mc'):
            raise ValueError('eval_strategy must be mean or mc')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eval_strategy = eval_strategy
        self.mc_samples = int(mc_samples)
        self.eps = float(eps)
        self.key_noise_clip = key_noise_clip
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, 2 * dim)
        self.v_proj = nn.Linear(dim, 2 * dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def _heads(self, x):
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)

    def _aggregate(self, query, key, value):
        score = torch.einsum('bqhd,bkhd->bhqk', query, key)
        score = score / math.sqrt(self.head_dim)
        attention = F.softmax(score.float(), dim=-1).to(score.dtype)
        return torch.einsum(
            'bhqk,bkhd->bqhd', self.dropout(attention), value)

    def forward(self, query, context):
        query_proj = self._heads(self.q_proj(query))
        key_mu, key_raw_var = self.k_proj(context).chunk(2, dim=-1)
        key_mu = self._heads(self.norm_k(key_mu))
        key_std = torch.sqrt(self._heads(
            F.softplus(key_raw_var) + self.eps))
        value_mu, value_raw_var = self.v_proj(context).chunk(2, dim=-1)
        value_mu = self._heads(self.norm_v(value_mu))
        value_std = torch.sqrt(self._heads(
            F.softplus(value_raw_var) + self.eps))

        if not self.training and self.eval_strategy == 'mean':
            attended = self._aggregate(query_proj, key_mu, value_mu)
        else:
            count = self.mc_samples if not self.training else 1
            samples = []
            for _ in range(count):
                noise = torch.randn_like(key_std)
                if self.key_noise_clip is not None:
                    noise = noise.clamp(
                        -self.key_noise_clip, self.key_noise_clip)
                key = key_mu + noise * key_std
                value = value_mu + torch.randn_like(value_std) * value_std
                samples.append(self._aggregate(query_proj, key, value))
            attended = torch.stack(samples).mean(0)
        attended = self.out_proj(attended.reshape(
            query.shape[0], query.shape[1], self.dim))
        gate = torch.sigmoid(self.gate)
        return gate * attended + (1. - gate) * query


class PVLQueryInteraction(nn.Module):
    """Build disease queries from prompt embeddings and segmentation tokens."""

    def __init__(self, in_channels, cfg, num_queries):
        super().__init__()
        hidden_dim = cfg.get('hidden_dim', 512)
        self.patch_embed = LightPatchEmbedding(
            in_channels, cfg['patch_reduced_channels'], hidden_dim,
            cfg['patch_size'])
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        text_dim = cfg.get('text_dim')
        self.text_query_proj = (
            nn.Linear(text_dim, hidden_dim)
            if text_dim is not None and text_dim != hidden_dim else None)
        self.cross_attention = ReparameterizedKeyCrossAttention(
            dim=hidden_dim,
            num_heads=cfg.get('nheads', 8),
            dropout=cfg.get('dropout', 0.),
            gate_init=cfg.get('gate_init', 0.),
            eval_strategy=cfg.get('eval_strategy', 'mean'),
            mc_samples=cfg.get('mc_samples', 10),
            key_noise_clip=cfg.get('key_noise_clip', 2.35))

    def forward(self, src, text_emb=None):
        patch_map = self.patch_embed(src)
        patch_tokens = patch_map.flatten(2).transpose(1, 2)
        if text_emb is None:
            queries = self.query_embed.weight.unsqueeze(0).expand(
                src.shape[0], -1, -1)
        else:
            queries = (self.text_query_proj(text_emb)
                       if self.text_query_proj is not None else text_emb)
        return self.cross_attention(queries, patch_tokens), patch_map


class QueryVarianceMahalanobisMap(nn.Module):
    """Patch-space Mahalanobis map with query-side uncertainty only."""

    def __init__(self, query_dim, visual_dim, num_heads, semantic_dim=256,
                 dropout=0., eps=1e-6, variance_floor=1e-4,
                 force_fp32=True, **kwargs):
        super().__init__()
        if semantic_dim % num_heads:
            raise ValueError('semantic_dim must be divisible by num_heads')
        self.num_heads = num_heads
        self.head_dim = semantic_dim // num_heads
        self.eps = float(eps)
        self.variance_floor = float(variance_floor)
        self.force_fp32 = bool(force_fp32)
        self.query_semantic = nn.Sequential(
            nn.Linear(query_dim, semantic_dim), nn.GELU(),
            nn.LayerNorm(semantic_dim))
        self.query_mean = nn.Linear(semantic_dim, semantic_dim)
        self.query_variance = nn.Linear(semantic_dim, semantic_dim)
        self.visual_semantic = nn.Sequential(
            nn.Conv2d(visual_dim, semantic_dim, 1), nn.GELU())
        self.visual_norm = nn.LayerNorm(semantic_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_score_map = None

    def _forward_impl(self, query, visual):
        b, q, _ = query.shape
        h, w = visual.shape[-2:]
        query = self.query_semantic(query)
        query_mean = self.query_mean(query).view(
            b, q, self.num_heads, self.head_dim)
        query_var = (F.softplus(self.query_variance(query)) + self.eps)
        query_var = query_var.clamp_min(self.variance_floor).view(
            b, q, self.num_heads, self.head_dim)
        visual = self.visual_semantic(visual)
        visual = self.visual_norm(
            visual.flatten(2).transpose(1, 2)).view(
                b, h, w, self.num_heads, self.head_dim)
        visual = visual.permute(0, 3, 4, 1, 2)
        inverse_var = query_var.reciprocal()
        visual_square = torch.einsum(
            'bndhw,bqnd->bqnhw', visual.square(), inverse_var)
        cross = torch.einsum(
            'bndhw,bqnd->bqnhw', visual, query_mean * inverse_var)
        query_square = (query_mean.square() * inverse_var).sum(dim=3)
        distance = (visual_square - 2. * cross
                    + query_square[..., None, None])
        score = -distance.clamp_min(0.) / math.sqrt(self.head_dim)
        self.last_score_map = score.detach()
        attention = F.softmax(score.flatten(3), dim=-1).view_as(score)
        return self.dropout(attention)

    def forward(self, query, visual):
        if not self.force_fp32:
            return self._forward_impl(query, visual)
        output_dtype = visual.dtype
        with torch.autocast(device_type=query.device.type, enabled=False):
            output = self._forward_impl(query.float(), visual.float())
        return output.to(output_dtype)


class PatchMahalanobisFusion(nn.Module):
    """Fuse patch-space probabilistic maps into full segmentation features."""

    def __init__(self, query_dim, patch_dim, seg_channels, nheads,
                 query_num, cfg):
        super().__init__()
        self.bbox_attention = QueryVarianceMahalanobisMap(
            query_dim, patch_dim, nheads, **cfg)
        self.mask_head = MaskHeadSmallConv(
            seg_channels + nheads * query_num, seg_channels)
        self.last_lowres_attention_map = None
        self.last_attention_map = None

    def forward(self, seg_feat, cls_queries, patch_feat):
        attention = self.bbox_attention(cls_queries, patch_feat)
        self.last_lowres_attention_map = attention.detach()
        attention = F.interpolate(
            attention.flatten(1, 2), size=seg_feat.shape[-2:],
            mode='bilinear', align_corners=False)
        self.last_attention_map = attention.detach()
        return self.mask_head(seg_feat, attention, seg_feat)


@MODELS.register_module()
class PSJointDecodeHead(BaseDecodeHead):
    """Final self-contained PS-Joint decoder; no custom decoder inheritance."""

    def __init__(self, base_head_cfg, hr_settings, trans_decoder_cfg,
                 detrseg_cfg, interaction_cfg, mahalanobis_detrseg_cfg=None,
                 mahalanobis_feature_space='patch',
                 mahalanobis_variance_mode='query_only',
                 cls_num_classes=11, cls_in_channel=768, loss_lr=None,
                 loss_cls=None, loss_hr=None, loss_contrast=None,
                 loss_contrast_low=None, cls_psl_thre=0.95,
                 seg_psl_thre=0.5, input_transform='resize_concat', **kwargs):

        
        channels = base_head_cfg['channels']
        num_classes = base_head_cfg['num_classes']
        in_channels = base_head_cfg['in_channels']

        hidden_dim = trans_decoder_cfg['hidden_dim']
        dropout = trans_decoder_cfg['dropout']
        nheads = trans_decoder_cfg['nheads']
        dim_feedforward = trans_decoder_cfg['dim_feedforward']
        dec_layers = trans_decoder_cfg['dec_layers']
        pre_norm = trans_decoder_cfg['pre_norm']
        feature_map_size = trans_decoder_cfg['feature_map_size']
        patch_size = trans_decoder_cfg['patch_size']
        mask_channels = trans_decoder_cfg['mask_channels']

        seg_nheads = detrseg_cfg['nheads']
        down_ratio = detrseg_cfg['down_ratio']


        super().__init__(
            input_transform=input_transform,
            channels = channels,
            num_classes = num_classes,
            **kwargs
            )

        if loss_cls is not None:
            self.loss_cls = MODELS.build(loss_cls)
        else:
            self.loss_cls = None
            
        self.loss_hr = MODELS.build(loss_hr)

        if loss_lr is not None:
            self.loss_lr = MODELS.build(loss_lr)
        else:
            self.loss_lr = None

        if loss_contrast_low is not None:
            self.loss_contrast_low = MODELS.build(loss_contrast_low)
        else:
            self.loss_contrast_low = None

        if loss_contrast is not None:
            self.loss_contrast = MODELS.build(loss_contrast)
        else:
            self.loss_contrast = None


        self.hr_scale = hr_settings['hr_scale']
        self.divisible = hr_settings.get('divisible', 8)
        self.crop_num = hr_settings.get('crop_num', 4)
        self.scale_ratio = hr_settings.get('scale_ratio', 1)
        self.hr_loss_weight = hr_settings.get('hr_loss_weight', 0.1)


        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(cls_in_channel, 1)
            ) for _ in range(cls_num_classes)
        ])

        self.base_head = MODELS.build(base_head_cfg)

        if mahalanobis_feature_space != 'patch':
            raise ValueError('The final release supports patch space only.')
        if mahalanobis_variance_mode != 'query_only':
            raise ValueError('The final release supports query_only only.')
        self.inter_transformer = PVLQueryInteraction(
            in_channels, interaction_cfg, cls_num_classes)
        self.detrseg = PatchMahalanobisFusion(
            query_dim=interaction_cfg.get('hidden_dim', 512),
            patch_dim=interaction_cfg.get('hidden_dim', 512),
            seg_channels=in_channels,
            nheads=seg_nheads,
            query_num=cls_num_classes,
            cfg=dict(mahalanobis_detrseg_cfg or {}))

        self.mask_proj = ConvModule(num_classes, in_channels, kernel_size=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.cls_proj = nn.Linear(hidden_dim, 1)

        self.cls_psl_thre = cls_psl_thre
        self.seg_psl_thre = seg_psl_thre

        self.multiscale_compressor = ConvModule(
            in_channels=hr_settings['in_channels'],
            out_channels=hr_settings['visual_dim'], kernel_size=1,
            conv_cfg=self.base_head.conv_cfg,
            norm_cfg=self.base_head.norm_cfg,
            act_cfg=self.base_head.act_cfg)

    def _transform_inputs(self, inputs):
        fused = super()._transform_inputs(inputs)
        return self.multiscale_compressor(fused)


    # ------------------------------
    # internal helper functions
    # ------------------------------
    def _get_crop_bbox(self, img_size, hr_scale):  #随机裁剪区域
        margin_h = max(img_size[0] - hr_scale[0], 0)
        margin_w = max(img_size[1] - hr_scale[1], 0)
        offset_h = np.random.randint(0, (margin_h + 1) // self.divisible) * self.divisible
        offset_w = np.random.randint(0, (margin_w + 1) // self.divisible) * self.divisible
        return offset_h, offset_h + hr_scale[0], offset_w, offset_w + hr_scale[1]

    def get_random_hr_scale(self):  #将裁剪区域进行随机的尺寸变换
        if isinstance(self.scale_ratio, (tuple, list)):
            min_r, max_r = self.scale_ratio
            ratio = np.random.random_sample() * (max_r - min_r) + min_r
            return (
                int(self.hr_scale[0] * ratio // self.divisible * self.divisible),
                int(self.hr_scale[1] * ratio // self.divisible * self.divisible),
            )
        return self.hr_scale

    def _resizecrop_feats(self, ori_size, feat, bbox):
        """crop low-res feat, resize back to full feat size"""
        h_grids = ori_size[0] // feat.shape[-2]
        w_grids = ori_size[1] // feat.shape[-1]
        lr_bbox = [bbox[0] // h_grids, bbox[1] // h_grids, bbox[2] // w_grids, bbox[3] // w_grids]
        lr_crop_feat = feat[..., lr_bbox[0]:lr_bbox[1], lr_bbox[2]:lr_bbox[3]]
        hr_feat = resize(lr_crop_feat, feat.shape[2:], mode='bilinear', align_corners=self.align_corners)
        return hr_feat

    def hr_slide_inference(self, ori_size, lr_feats):
        h_stride, w_stride = h_crop, w_crop = self.hr_scale
        h_img, w_img = ori_size
        bs = lr_feats.shape[0]
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        crop_boxes = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1, x1 = h_idx * h_stride, w_idx * w_stride
                y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
                y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
                crop_boxes.append([y1, y2, x1, x2])   #计算滑动窗口的尺寸

        crop_feats = torch.cat([self._resizecrop_feats(ori_size,lr_feats,bbox) for bbox in crop_boxes])

        crop_logits = self.base_head([crop_feats])


        if ori_size > self.hr_scale:
            crop_logits = resize(crop_logits,self.hr_scale,mode='bilinear',align_corners=self.align_corners)\
            .reshape(-1,bs,self.num_classes,*self.hr_scale)

        else:
            crop_logits = resize(crop_logits,ori_size,mode='bilinear',align_corners=self.align_corners)\
            .reshape(-1,bs,self.num_classes,*ori_size)
        

        return {'logits': crop_logits, 'feats':crop_feats, 'boxes': crop_boxes}

    def _get_valid_indices(self, batch_data_samples: SampleList, flag: str) -> List[int]:
        valid_indices = []
        for idx, data_sample in enumerate(batch_data_samples):
            if data_sample.get(flag, True):
                valid_indices.append(idx)
        return valid_indices

    def _stack_batch_cls(self, batch_data_samples: SampleList) -> Optional[torch.Tensor]:
        cls_labels = []
        for data_sample in batch_data_samples:
            cls_label = data_sample.get('gt_label', None)
            if cls_label is None:
                return None
            cls_labels.append(cls_label)
        return torch.stack(cls_labels, dim=0)


    def forward(self, inputs: List[torch.Tensor], img_size, text_emb=None, set_cross_attn=True):

        # cls_logits = [layer(inputs[-1]) for layer in self.cls_heads]  # list of (N,1)
        # cls_logits = torch.cat(cls_logits, dim=1)  # (N, cls_num_classes)

        # print(f'inputs[0] {inputs[0].shape}')
        # print(f'inputs[1] {inputs[1].shape}')
        # print(f'inputs[2] {inputs[2].shape}')
        # print(f'inputs[3] {inputs[3].shape}')


        # LR 分支
        lr_feat = self._transform_inputs(inputs)

        if text_emb is not None:
            if set_cross_attn:
                cls_suff, patch_feat = self.inter_transformer(
                    lr_feat, text_emb=text_emb)

                lr_feat = self.detrseg(lr_feat, cls_suff, patch_feat)

                # cls_feat = cls_suff

                cls_logits = self.cls_proj(cls_suff)

                cls_logits = cls_logits.squeeze(-1)
            else:
                cls_logits = self.cls_proj(text_emb).squeeze(-1)
        else:
            cls_logits = [layer(inputs[-1]) for layer in self.cls_heads]  # list of (N,1)
            cls_logits = torch.cat(cls_logits, dim=1)  # (N, cls_num_classes)
            # print("xxxx")
        
        lr_logits, decode_feat = self.base_head([lr_feat], return_fea=True)
       

        up_lr_logit = resize(lr_logits, img_size, mode='bilinear', align_corners=self.align_corners)


        fuse_logits = up_lr_logit.clone()
        for logit,_,bbox in zip(*self.hr_slide_inference(img_size, lr_feat).values()):
            slc = slice(bbox[0],bbox[1]),slice(bbox[2],bbox[3])
            fuse_logits[:,:,slc[0],slc[1]] = (logit + up_lr_logit[:,:,slc[0],slc[1]])/2

        if not self.training:
            return fuse_logits, cls_logits

        return fuse_logits, cls_logits, lr_logits, lr_feat, decode_feat


    def loss(self, inputs: Tuple[torch.Tensor], batch_data_samples: SampleList, text_emb,
             train_cfg: ConfigType) -> dict:
        img_size = batch_data_samples[0].metainfo['img_shape']
        seg_logits, cls_logits, lr_logits, lr_feat, _ = self.forward(inputs, img_size, text_emb)
        return self.loss_by_feat(seg_logits, cls_logits, lr_logits, lr_feat, batch_data_samples)

    def get_psl_labels(self, seg_logits, cls_logits, batch_data_samples):
        seg_psl = generate_pseudo_labels(seg_logits, self._stack_batch_gt(batch_data_samples), thresh=self.seg_psl_thre)
        cls_psl = generate_pseudo_labels(cls_logits, self._stack_batch_cls(batch_data_samples), thresh=self.cls_psl_thre)
        return seg_psl, cls_psl

    def get_gts(self, batch_data_samples: SampleList):
        seg_label = self._stack_batch_gt(batch_data_samples)
        cls_label = self._stack_batch_cls(batch_data_samples)
        return seg_label, cls_label

    def loss_by_feat(self, seg_logits: torch.Tensor, cls_logits: Optional[torch.Tensor], lr_logits:torch.Tensor, lr_feat: torch.Tensor,
                     batch_data_samples: SampleList) -> dict:

        losses = {}

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        seg_indices = self._get_valid_indices(batch_data_samples, 'has_seg')
        if seg_indices:
            seg_samples = [batch_data_samples[i] for i in seg_indices]
            seg_label = self._stack_batch_gt(seg_samples)
            seg_logits_valid = seg_logits[seg_indices]
            seg_logits_valid = resize(
                input=seg_logits_valid,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            for loss_decode in losses_decode:
                if loss_decode.loss_name not in losses:
                    losses[loss_decode.loss_name] = loss_decode(
                        seg_logits_valid, seg_label)
                else:
                    losses[loss_decode.loss_name] += loss_decode(
                        seg_logits_valid, seg_label)

            if self.hr_loss_weight>0:

                img_size = batch_data_samples[0].metainfo['img_shape']

                hr_scale = self.get_random_hr_scale()
                hr_bboxes = [self._get_crop_bbox(img_size, hr_scale) for _ in range(self.crop_num)]

                lr_feat = lr_feat[seg_indices]

                hr_feats = torch.cat([self._resizecrop_feats(img_size, lr_feat, bbox)
                                    for bbox in hr_bboxes])

                hr_logits = self.base_head([hr_feats])
                hr_labels = torch.cat([seg_label[..., y1:y2, x1:x2] for y1, y2, x1, x2 in hr_bboxes])
                hr_logits = resize(hr_logits, hr_labels.shape[2:], mode='bilinear',align_corners=self.align_corners)

                loss_hr = self.hr_loss_weight * self.loss_hr(hr_logits, hr_labels)

                if isinstance(loss_hr, dict):
                    losses.update(loss_hr)
                else:
                    losses[self.loss_hr.loss_name] = loss_hr

            # lr_logits_valid = lr_logits[seg_indices]
            # lr_fuse_logits = resize(seg_logits, lr_logits.shape[2:], mode='bilinear',align_corners=self.align_corners)
            # # lr_fuse_logits_valid = lr_fuse_logits[seg_indices]
            # loss_distill = self.loss_distill(lr_logits, lr_fuse_logits, label=None)
            # losses.update(loss_distill)

            if self.loss_lr is not None:
                lr_logits_valid = lr_logits[seg_indices]
                lr_seg_label = resize(seg_label.float(), lr_logits_valid.shape[2:], mode='nearest').long()
                loss_lr = self.loss_lr(lr_logits_valid, lr_seg_label)
                
                if isinstance(loss_lr, dict):
                    losses.update(loss_lr)
                else:
                    losses[self.loss_lr.loss_name] = loss_lr
                       
        else:
            zero = seg_logits.sum() * 0.0
            for loss_decode in losses_decode:
                losses[loss_decode.loss_name] = zero

        if self.loss_cls is not None and cls_logits is not None:
            cls_indices = self._get_valid_indices(batch_data_samples, 'has_cls')
            if cls_indices:
                cls_samples = [batch_data_samples[i] for i in cls_indices]
                cls_logits_valid = cls_logits[cls_indices]
                cls_label = self._stack_batch_cls(cls_samples)
                if cls_label is not None:
                    loss_cls = self.loss_cls(cls_logits_valid, cls_label)
                    if isinstance(loss_cls, dict):
                        losses.update(loss_cls)
                    else:
                        losses[self.loss_cls.loss_name] = loss_cls
                else:
                    losses[self.loss_cls.loss_name] = cls_logits.sum() * 0.0
            else:
                losses[self.loss_cls.loss_name] = cls_logits.sum() * 0.0

        

        if self.loss_contrast is not None:
            loss_contrast = self.loss_contrast(seg_logits)
            losses.update(loss_contrast)

        if self.loss_contrast_low is not None:
            loss_contrast_low = self.loss_contrast_low(lr_logits)
            losses.update(loss_contrast_low)
        
        return losses

    def predict(self, inputs: Tuple[torch.Tensor], batch_img_metas: List[dict], text_emb,
                test_cfg: ConfigType, set_text_emb_cross_attn=True):
    
        img_size = batch_img_metas[0]['img_shape']
        seg_logits, cls_logits = self.forward(inputs, img_size, text_emb, set_text_emb_cross_attn)
        seg_logits = self.predict_by_feat(seg_logits, batch_img_metas)
        return seg_logits, cls_logits
    

        


