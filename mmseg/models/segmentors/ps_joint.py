# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from ..utils import resize
from .base import BaseSegmentor


from mmseg.tool_libs.clip.models import *
from mmseg.tool_libs.clip.utils import tokenize
from mmseg.tool_libs.psl_generation.psl_filter import psl_filter

from mmengine.model import ExponentialMovingAverage
from mmengine.optim import OptimWrapper

from mmseg.tool_libs.psl_generation.generate_psl import uncertainty_filter_pseudo_label
from mmseg.tool_libs.psl_generation.psl_filter import psl_filter, uncertainty_filter
from mmseg.tool_libs.psl_generation import ramps

from pathlib import Path
from PIL import Image

from copy import deepcopy




def masked_mean(x, mask, eps=1e-8):
    return (x * mask).sum() / (mask.sum() + eps)

def binary_entropy(prob, eps=1e-8):
    prob = prob.clamp(min=eps, max=1 - eps)
    return -(prob * torch.log(prob) + (1 - prob) * torch.log(1 - prob))


@MODELS.register_module()
class PSJoint(BaseSegmentor):
    """PS-Joint: partially supervised joint segmentation and classification.

    The Python registry name is ``PSJoint`` because Python identifiers cannot
    contain a hyphen; the method name used in the paper remains ``PS-Joint``.
    """

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 dino_backbone: ConfigType,
                 dino_decode_head: ConfigType,
                 text_encoder: ConfigType,
                 class_names: ConfigType,
                 context_length:ConfigType,
                 class_descriptions: OptConfigType=None,
                 token_embed_dim=512, 
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 text_encoder_pretrained: Optional[str] = None,
                 ema_decay=0.999,
                 gen_psl=False,
                 psl_iter=0,
                 uncertainty_threshold=0.75,
                 cls_psl_loss=None,
                 seg_psl_loss=None,
                 seg_lr_psl_loss=None,
                 f_dist_dino_loss=None,
                 f_dist_t_loss = None,
                 f_dist_dino_low_loss=None,
                 f_dist_t_low_loss = None,
                 pred_dist_dino_loss=None,
                 pred_dist_dino_cls_loss=None,
                 cls_psl_uncertain_filter_button=False,
                 cls_high_uncertain_with_ignore_index=True,
                 seg_high_uncertain_with_ignore_index=True,
                 two_teacher_psl_filter_button=False,
                 single_psl_uncertain_filter=False,
                 filter_both_uncertain=True,
                 specific_expert_psl_button=False,
                 stu_psl_update=False,
                 stu_cls_psl_loss=None,
                 stu_seg_psl_loss=None,
                 vis_psl=False,
                 vis_psl_dir='work_dirs/psl_visualizations',
                 vis_psl_interval=500,
                 vis_psl_max_samples=2,
                 vis_psl_max_classes=4,
                 vis_pred=False,
                 set_extra_spatial_prior=False,
                 set_text_emb=True,
                 set_text_emb_cross_attn=True,
                #  normal_cls_ignore=False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.ema_decay = ema_decay
        self.gen_psl = gen_psl
        self.psl_iter = psl_iter

        self.cls_psl_loss = MODELS.build(cls_psl_loss)
        self.seg_psl_loss = MODELS.build(seg_psl_loss)

        if f_dist_dino_loss is not None:
            self.f_dist_dino_loss = MODELS.build(f_dist_dino_loss)
        else:
            self.f_dist_dino_loss = None

        if f_dist_t_loss is not None:
            self.f_dist_t_loss = MODELS.build(f_dist_t_loss)
        else:
            self.f_dist_t_loss = None

        if f_dist_dino_low_loss is not None:
            self.f_dist_dino_low_loss = MODELS.build(f_dist_dino_low_loss)
        else:
            self.f_dist_dino_low_loss = None

        if f_dist_t_low_loss is not None:
            self.f_dist_t_low_loss = MODELS.build(f_dist_t_low_loss)
        else:
            self.f_dist_t_low_loss = None

        if pred_dist_dino_loss is not None:
            self.pred_dist_dino_loss = MODELS.build(pred_dist_dino_loss)
        else:
            self.pred_dist_dino_loss = None

        if pred_dist_dino_cls_loss is not None:
            self.pred_dist_dino_cls_loss = MODELS.build(pred_dist_dino_cls_loss)
        else:
            self.pred_dist_dino_cls_loss = None


        if seg_lr_psl_loss is not None:
            self.seg_lr_psl_loss = MODELS.build(seg_lr_psl_loss)
        else:
            self.seg_lr_psl_loss = None


        self.uncertainty_threshold = uncertainty_threshold
        self.cls_psl_uncertain_filter_button = cls_psl_uncertain_filter_button
        self.cls_high_uncertain_with_ignore_index = cls_high_uncertain_with_ignore_index
        self.seg_high_uncertain_with_ignore_index = seg_high_uncertain_with_ignore_index
        self.specific_expert_psl_button=specific_expert_psl_button
        # self.normal_cls_ignore = normal_cls_ignore

        self.two_teacher_psl_filter_button = two_teacher_psl_filter_button
        self.single_psl_uncertain_filter = single_psl_uncertain_filter
        self.filter_both_uncertain = filter_both_uncertain

        self.stu_psl_update = stu_psl_update

        self.vis_psl = vis_psl
        self.vis_psl_dir = Path(vis_psl_dir)
        self.vis_psl_interval = max(int(vis_psl_interval), 1)
        self.vis_psl_max_samples = max(int(vis_psl_max_samples), 1)
        self.vis_psl_max_classes = max(int(vis_psl_max_classes), 1)
        self.vis_pred =  vis_pred



        if stu_cls_psl_loss is not None:
            self.stu_cls_psl_loss = MODELS.build(stu_cls_psl_loss)
        else:
            self.stu_cls_psl_loss = None

        if stu_seg_psl_loss is not None:
            self.stu_seg_psl_loss = MODELS.build(stu_seg_psl_loss)
        else:
            self.stu_seg_psl_loss = None


        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

        if text_encoder_pretrained is not None:
            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            text_encoder.pretrained = text_encoder_pretrained

        self.num_classes = len(class_names)
        self.context_length = context_length

        
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.text_encoder = MODELS.build(text_encoder); self.text_encoder.init_weights()

        

        # self.teacher = deepcopy(self)
        # self._freeze_teacher()
        # if hasattr(self.teacher, 'teacher'):
        #     del self.teacher.teacher


        self.dino_backbone = MODELS.build(dino_backbone)
        self.dino_decode_head = MODELS.build(dino_decode_head)
        self.dino_text_encoder = MODELS.build(text_encoder); self.dino_text_encoder.init_weights()
        
        # requires_grad False
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False

        for name, param in self.dino_text_encoder.named_parameters():
            param.requires_grad = False
        
        # coop
        self.text_encoder.to('cuda')
        text_length = self.text_encoder.context_length - self.context_length
        prompt_num = self.context_length

        
        if class_descriptions is None:
            self.texts = torch.cat([tokenize(c, context_length=text_length) for c in class_names]).to('cuda')
        else:
            self.texts = torch.cat([tokenize(class_descriptions[c], context_length=text_length) for c in class_names]).to('cuda')

        self.contexts = nn.Parameter(torch.randn(1, prompt_num, token_embed_dim))
        self.dino_contexts = nn.Parameter(torch.randn(1, prompt_num, token_embed_dim))

        teacher_model = nn.Module()
        teacher_model.backbone = deepcopy(self.backbone)
        teacher_model.decode_head = deepcopy(self.decode_head)
        teacher_model.contexts = deepcopy(self.contexts)
        # teacher_model.text_encoder = deepcopy(self.text_encoder)

        self.teacher = teacher_model
        self._freeze_teacher()

        # self.teacher = ExponentialMovingAverage(
        #     teacher_model,
        #     momentum=1-ema_decay,  # ≈ decay 0.999
        #     update_buffers=True
        # )

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.set_extra_spatial_prior = set_extra_spatial_prior
        self.set_text_emb = set_text_emb
        self.set_text_emb_cross_attn = set_text_emb_cross_attn

        # self.save_dim8_pred = bool(
        #     self.test_cfg.get('save_dim8_pred', False)
        # ) if self.test_cfg is not None else False
        # self.dim8_pred_channel = int(
        #     self.test_cfg.get('dim8_pred_channel', 7)
        # ) if self.test_cfg is not None else 7
        # self.dim8_pred_save_dir = self.test_cfg.get(
        #     'dim8_pred_save_dir',
        #     'work_dirs/dim8_pred_dataset'
        # ) if self.test_cfg is not None else 'work_dirs/dim8_pred_dataset'
        # self.dim8_pred_binary_threshold = self.test_cfg.get(
        #     'dim8_pred_binary_threshold', None
        # ) if self.test_cfg is not None else None

        assert self.with_decode_head

    def _freeze_teacher(self):
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def ema_update_teacher(self):
        """用 student 参数 EMA 更新 teacher 参数"""
        student_state = self.state_dict()
        teacher_state = self.teacher.state_dict()

        for k, v_t in teacher_state.items():
            if k not in student_state:
                continue
            v_s = student_state[k]
            if not v_t.dtype.is_floating_point:
                v_t.copy_(v_s)
            else:
                v_t.mul_(self.ema_decay).add_(v_s.detach(), alpha=1 - self.ema_decay)


    #  training visualization 

    def _should_vis_psl(self):
        if not self.vis_psl:
            return False
        cur_iter = int(getattr(self, 'iter', 0))
        if cur_iter < self.psl_iter:
            return False
        if cur_iter % self.vis_psl_interval != 0:
            return False
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _tensor_to_rgb_array(self, img: Tensor):
        img = img.detach().float().cpu()
        mean = getattr(self.data_preprocessor, 'mean', None)
        std = getattr(self.data_preprocessor, 'std', None)
        if mean is not None and std is not None:
            mean = mean.detach().float().cpu().view(-1, 1, 1)
            std = std.detach().float().cpu().view(-1, 1, 1)
            img = img * std + mean
        else:
            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min + 1e-6) * 255
        img = img[:3].permute(1, 2, 0).numpy()
        return np.clip(img, 0, 255).astype(np.uint8)

    def _tensor_to_gray_array(self, tensor: Tensor, vmin=None, vmax=None):
        data = tensor.detach().float().cpu()
        if vmin is None:
            vmin = float(data.min())
        if vmax is None:
            vmax = float(data.max())
        data = (data - vmin) / (vmax - vmin + 1e-6)
        return np.clip(data.numpy() * 255, 0, 255).astype(np.uint8)

    def _tensor_to_heatmap_array(self, tensor: Tensor, vmin=None, vmax=None):
        data = tensor.detach().float().cpu()
        finite_mask = torch.isfinite(data)
        if not finite_mask.any():
            data = torch.zeros_like(data)
            finite_mask = torch.ones_like(data, dtype=torch.bool)
        if vmin is None:
            vmin = float(data[finite_mask].min())
        if vmax is None:
            vmax = float(data[finite_mask].max())
        data = torch.nan_to_num(data, nan=vmin, posinf=vmax, neginf=vmin)
        data = ((data - vmin) / (vmax - vmin + 1e-6)).clamp(0, 1).numpy()

        # Turbo-like compact heatmap: dark blue -> cyan -> yellow -> red.
        stops = np.array([
            [48, 18, 59],
            [50, 101, 189],
            [33, 145, 140],
            [144, 215, 67],
            [253, 231, 37],
            [240, 91, 35],
            [180, 4, 38],
        ], dtype=np.float32)
        scaled = data * (len(stops) - 1)
        low = np.clip(np.floor(scaled).astype(np.int64), 0, len(stops) - 1)
        high = np.clip(low + 1, 0, len(stops) - 1)
        frac = (scaled - low)[..., None]
        rgb = stops[low] * (1.0 - frac) + stops[high] * frac
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _psl_to_gray_array(self, tensor: Tensor, ignore_index):
        data = tensor.detach().cpu()
        out = torch.zeros_like(data, dtype=torch.uint8)
        out[data > 0] = 255
        return out.numpy()

    def _safe_sample_name(self, data_sample: SegDataSample, sample_idx: int):
        img_path = data_sample.metainfo.get('img_path', None)
        if img_path is None:
            return f'sample_{sample_idx:02d}'
        return Path(img_path).stem.replace(' ', '_')

    def _select_vis_classes(self, sample_idx, cls_logits, final_seg_psl):
        selected = []
        if final_seg_psl is not None:
            seg_i = final_seg_psl[sample_idx]
            pos = (seg_i == 1).flatten(1).any(dim=1)
            selected.extend(torch.nonzero(pos, as_tuple=False).flatten().tolist())
        if cls_logits is not None:
            probs = torch.sigmoid(cls_logits[sample_idx]).detach()
            topk = min(self.vis_psl_max_classes, probs.numel())
            selected.extend(torch.topk(probs, k=topk).indices.tolist())
        unique = []
        for cls_idx in selected:
            if cls_idx not in unique:
                unique.append(int(cls_idx))
            if len(unique) >= self.vis_psl_max_classes:
                break
        return unique

    def _save_attention_maps(self, save_dir: Path, attn: Optional[Tensor],
                             source: str, img_size, sample_idx: int,
                             class_indices: List[int],
                             image_array: Optional[np.ndarray] = None):
        if attn is None:
            return
        attn = attn.detach()
        if attn.dim() == 5:
            attn = attn.mean(dim=2)
        attn = resize(
            attn,
            size=img_size,
            mode='bilinear',
            align_corners=False,
            warning=False)
        for cls_idx in class_indices:
            if cls_idx >= attn.shape[1]:
                continue
            arr = self._tensor_to_heatmap_array(attn[sample_idx, cls_idx])
            Image.fromarray(arr, mode='RGB').save(
                save_dir / f'{source}_qk_attn_cls{cls_idx:02d}.png')
            if image_array is not None:
                overlay = (
                    0.45 * image_array.astype(np.float32)
                    + 0.55 * arr.astype(np.float32))
                Image.fromarray(
                    np.clip(overlay, 0, 255).astype(np.uint8),
                    mode='RGB').save(
                        save_dir / f'{source}_qk_attn_overlay_cls{cls_idx:02d}.png')

    def _dump_psl_visualizations(self, inputs: Tensor, data_samples: SampleList,
                                 seg_psl: Optional[Tensor],
                                 dino_seg_psl: Tensor,
                                 seg_filter_psl: Tensor,
                                 cls_psl: Optional[Tensor],
                                 dino_cls_psl: Tensor,
                                 cls_filter_psl: Tensor,
                                 seg_entropy: Optional[Tensor],
                                 seg_dino_entropy: Optional[Tensor],
                                 cls_entropy: Optional[Tensor],
                                 cls_dino_entropy: Optional[Tensor],
                                 cls_logits: Tensor,
                                 cls_dino_logits: Tensor):
        cur_iter = int(getattr(self, 'iter', 0))
        root = self.vis_psl_dir / f'iter_{cur_iter:07d}'
        root.mkdir(parents=True, exist_ok=True)
        batch = min(inputs.shape[0], self.vis_psl_max_samples)
        img_size = inputs.shape[2:]
        swin_attn = getattr(self.decode_head.detrseg, 'last_attention_map', None)
        dino_attn = getattr(self.dino_decode_head.detrseg, 'last_attention_map', None)

        for sample_idx in range(batch):
            sample_name = self._safe_sample_name(data_samples[sample_idx], sample_idx)
            save_dir = root / f'{sample_idx:02d}_{sample_name}'
            save_dir.mkdir(parents=True, exist_ok=True)
            image_array = self._tensor_to_rgb_array(inputs[sample_idx])
            Image.fromarray(image_array).save(save_dir / 'image.png')

            class_indices = self._select_vis_classes(
                sample_idx, cls_dino_logits, seg_filter_psl)
            class_names = [
                self.class_names[i] if hasattr(self, 'class_names') else str(i)
                for i in class_indices
            ]
            (save_dir / 'selected_classes.txt').write_text(
                '\n'.join(f'{i}: {name}' for i, name in zip(class_indices, class_names)),
                encoding='utf-8')

            self._save_attention_maps(save_dir, swin_attn, 'swin', img_size,
                                      sample_idx, class_indices, image_array)
            self._save_attention_maps(save_dir, dino_attn, 'dino', img_size,
                                      sample_idx, class_indices, image_array)

            swin_cls_i = None if cls_psl is None else cls_psl[sample_idx].detach().cpu()
            dino_cls_i = dino_cls_psl[sample_idx].detach().cpu()
            final_cls_i = cls_filter_psl[sample_idx].detach().cpu()
            swin_unc_i = None if cls_entropy is None else cls_entropy[sample_idx].detach().cpu()
            dino_unc_i = cls_dino_entropy[sample_idx].detach().cpu()
            cls_rows = ['class,swin_psl,dino_psl,final_psl,swin_uncert,dino_uncert']
            for cls_idx in range(dino_cls_psl.shape[1]):
                swin_label = '' if swin_cls_i is None else int(swin_cls_i[cls_idx])
                swin_unc = '' if swin_unc_i is None else float(swin_unc_i[cls_idx])
                cls_rows.append(
                    f'{cls_idx},{swin_label},{int(dino_cls_i[cls_idx])},'
                    f'{int(final_cls_i[cls_idx])},{swin_unc},'
                    f'{float(dino_unc_i[cls_idx])}')
            (save_dir / 'cls_pseudo_labels.csv').write_text(
                '\n'.join(cls_rows), encoding='utf-8')

            for cls_idx in class_indices:
                if seg_psl is not None:
                    Image.fromarray(
                        self._psl_to_gray_array(seg_psl[sample_idx, cls_idx], 255),
                        mode='L').save(save_dir / f'swin_psl_cls{cls_idx:02d}.png')
                Image.fromarray(
                    self._psl_to_gray_array(dino_seg_psl[sample_idx, cls_idx], 255),
                    mode='L').save(save_dir / f'dino_psl_cls{cls_idx:02d}.png')
                Image.fromarray(
                    self._psl_to_gray_array(seg_filter_psl[sample_idx, cls_idx], 255),
                    mode='L').save(save_dir / f'final_filtered_psl_cls{cls_idx:02d}.png')
                if seg_entropy is not None:
                    Image.fromarray(
                        self._tensor_to_heatmap_array(
                            seg_entropy[sample_idx, cls_idx], vmin=0.0),
                        mode='RGB').save(save_dir / f'swin_uncert_cls{cls_idx:02d}.png')
                if seg_dino_entropy is not None:
                    Image.fromarray(
                        self._tensor_to_heatmap_array(
                            seg_dino_entropy[sample_idx, cls_idx], vmin=0.0),
                        mode='RGB').save(save_dir / f'dino_uncert_cls{cls_idx:02d}.png')
        #  training visualization 

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:

        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]):

        inputs = resize(inputs, self.decode_head.hr_scale, mode='bilinear',align_corners=False)

        x = self.extract_feat(inputs)

        if not self.set_extra_spatial_prior:
            x_dino = self.dino_backbone(inputs, extra_spm=False, spa_prior=x)
            # print("swin1")
        else:
            x_dino = self.dino_backbone(inputs, extra_spm=True)

        # x_dino = self.dino_backbone(inputs, extra_spm=False, spa_prior=x)

        b_size = x[0].shape[0]
        
        dino_text_emb = self.dino_text_encoder(self.texts, context=self.dino_contexts).expand(b_size,-1,-1)


        if self.set_text_emb:
            seg_logits, cls_logits = self.dino_decode_head.predict(
            x_dino, batch_img_metas, dino_text_emb, self.test_cfg, self.set_text_emb_cross_attn)
        else:
            seg_logits, cls_logits = self.dino_decode_head.predict(
            x_dino, batch_img_metas, None, self.test_cfg)
            # print("swin2")

        return seg_logits, cls_logits


    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList, text_emb) -> dict:
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples, text_emb,
                                            self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses


    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses
    

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:

        img_size = inputs.shape[2:]

        inputs = resize(inputs, self.decode_head.hr_scale, mode='bilinear',align_corners=False)
        
        x = self.extract_feat(inputs)

        if not self.set_extra_spatial_prior:
            x_dino = self.dino_backbone(inputs, extra_spm=False, spa_prior=x)
        else:
            x_dino = self.dino_backbone(inputs, extra_spm=True)
        
        b_size = x[0].shape[0]

        text_emb = self.text_encoder(self.texts, context=self.contexts).expand(b_size,-1,-1)
        dino_text_emb = self.dino_text_encoder(self.texts, context=self.dino_contexts).expand(b_size,-1,-1)
        losses = dict()

        if self.set_text_emb:
            seg_logits, cls_logits, lr_logits, lr_feat, decode_feat =  self.decode_head.forward(x, img_size, text_emb, self.set_text_emb_cross_attn)
        else:
            seg_logits, cls_logits, lr_logits, lr_feat, decode_feat =  self.decode_head.forward(x, img_size, None)

        # if not (self.gen_psl and self.iter>=self.psl_iter): 
        loss_decode = self.decode_head.loss_by_feat(seg_logits, cls_logits, lr_logits, lr_feat, data_samples)
        losses.update(loss_decode)

        if self.set_text_emb:
            seg_dino_logits, cls_dino_logits, lr_dino_logits, dino_lr_feat, dino_decode_feat = self.dino_decode_head.forward(x_dino, img_size, dino_text_emb, self.set_text_emb_cross_attn)
        else:
            seg_dino_logits, cls_dino_logits, lr_dino_logits, dino_lr_feat, dino_decode_feat = self.dino_decode_head.forward(x_dino, img_size, None)

    
        loss_dino_decode = self.dino_decode_head.loss_by_feat(seg_dino_logits, cls_dino_logits, lr_dino_logits, dino_lr_feat, data_samples)
        losses.update(loss_dino_decode)

        if self.gen_psl and self.iter>=self.psl_iter:
            with torch.no_grad():
                dino_seg_psl, dino_cls_psl = self.dino_decode_head.get_psl_labels(seg_dino_logits, cls_dino_logits, data_samples)
                
                if self.single_psl_uncertain_filter:
                    seg_dino_probs = torch.sigmoid(seg_dino_logits)
                    seg_dino_entropy = binary_entropy(seg_dino_probs)
                    cls_dino_probs = torch.sigmoid(cls_dino_logits)
                    cls_dino_entropy = binary_entropy(cls_dino_probs)
                    seg_filter_psl = uncertainty_filter(dino_seg_psl, seg_dino_entropy, threshold=self.uncertainty_threshold, ignore_index=255)
                    cls_filter_psl = uncertainty_filter(dino_cls_psl, cls_dino_entropy, threshold=self.uncertainty_threshold, ignore_index=-1)

                elif self.two_teacher_psl_filter_button:
                    seg_psl, cls_psl = self.decode_head.get_psl_labels(seg_logits, cls_logits, data_samples)
                    seg_dino_probs = torch.sigmoid(seg_dino_logits)
                    seg_dino_entropy = binary_entropy(seg_dino_probs)
                    cls_dino_probs = torch.sigmoid(cls_dino_logits)
                    cls_dino_entropy = binary_entropy(cls_dino_probs)

                    seg_probs = torch.sigmoid(seg_logits)
                    seg_entropy = binary_entropy(seg_probs)
                    cls_probs = torch.sigmoid(cls_logits)
                    cls_entropy = binary_entropy(cls_probs)
                    seg_filter_psl = uncertainty_filter_pseudo_label(dino_seg_psl, seg_psl, seg_dino_entropy, seg_entropy, uncertainty_threshold=self.uncertainty_threshold, ignore_index=255, high_uncertain_with_ignore_index=self.seg_high_uncertain_with_ignore_index, filter_both_uncertain=self.filter_both_uncertain)
                    cls_filter_psl = uncertainty_filter_pseudo_label(dino_cls_psl, cls_psl, cls_dino_entropy, cls_entropy, uncertainty_threshold=self.uncertainty_threshold, ignore_index=-1, high_uncertain_with_ignore_index=self.cls_high_uncertain_with_ignore_index, filter_both_uncertain=self.filter_both_uncertain)
                    
                elif self.specific_expert_psl_button:
                    seg_psl, cls_psl = self.decode_head.get_psl_labels(seg_logits, cls_logits, data_samples)
                    seg_filter_psl = seg_psl
                    cls_filter_psl = cls_psl 
                else:
                    seg_filter_psl = dino_seg_psl
                    cls_filter_psl = dino_cls_psl


                if self._should_vis_psl():
                    self._dump_psl_visualizations(
                        inputs, data_samples, seg_psl, dino_seg_psl,
                        seg_filter_psl, cls_psl, dino_cls_psl,
                        cls_filter_psl, seg_entropy, seg_dino_entropy,
                        cls_entropy, cls_dino_entropy, cls_logits,
                        cls_dino_logits)


            loss_cls_psl = self.cls_psl_loss(cls_dino_logits, cls_filter_psl)
            loss_seg_psl = self.seg_psl_loss(seg_dino_logits, seg_filter_psl)

            if isinstance(loss_seg_psl, dict):
                losses.update(loss_seg_psl)
            else:
                losses['loss_seg_psl'] = loss_seg_psl

            if isinstance(loss_cls_psl, dict):
                losses.update(loss_cls_psl)
            else:
                losses['loss_cls_psl'] = loss_cls_psl

            if self.stu_psl_update:
                loss_stu_cls_psl = self.stu_cls_psl_loss(cls_logits, cls_filter_psl)
                loss_stu_seg_psl = self.stu_seg_psl_loss(seg_logits, seg_filter_psl)

                if isinstance(loss_stu_seg_psl, dict):
                    losses.update(loss_stu_seg_psl)
                else:
                    losses['loss_stu_seg_psl'] = loss_stu_seg_psl

                if isinstance(loss_stu_cls_psl, dict):
                    losses.update(loss_stu_cls_psl)
                else:
                    losses['loss_stu_cls_psl'] = loss_stu_cls_psl

        
        return losses

    def _save_all_channel_binary_masks(self,
                                       i_seg_prob: Tensor,
                                       data_sample: SegDataSample,
                                       sample_idx: int,
                                       save_root: str = 'work_dirs/pred_mask_vis',
                                       threshold: float = 0.5):
        """
        Save original image and binary masks for all prediction channels.

        Args:
            i_seg_prob: shape [C, H, W] or [1, C, H, W], after sigmoid.
            data_sample: current SegDataSample.
            sample_idx: index in current batch.
            save_root: root directory for saving visualization results.
            threshold: threshold used to convert probability map to 0/1 mask.
        """

        # [1, C, H, W] -> [C, H, W]
        if i_seg_prob.dim() == 4:
            assert i_seg_prob.shape[0] == 1, \
                f'Expected batch size 1, but got {i_seg_prob.shape}'
            i_seg_prob = i_seg_prob.squeeze(0)

        assert i_seg_prob.dim() == 3, \
            f'Expected i_seg_prob with shape [C, H, W], but got {i_seg_prob.shape}'

        num_classes = i_seg_prob.shape[0]

        img_meta = data_sample.metainfo
        img_path = img_meta.get('img_path', None)

        if img_path is not None:
            img_name = Path(img_path).stem.replace(' ', '_')
        else:
            img_name = f'sample_{sample_idx:06d}'

        save_dir = Path(save_root) / img_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存原图
        if img_path is not None and Path(img_path).exists():
            try:
                ori_img = Image.open(img_path).convert('RGB')
                ori_img.save(save_dir / '00_original.png')
            except Exception as e:
                print_log(
                    f'Failed to save original image from {img_path}: {e}',
                    logger='current',
                    level=logging.WARNING)

        # 2. 保存每个通道的二值 mask
        prob = i_seg_prob.detach().float().cpu()

        for c in range(num_classes):
            binary_mask = (prob[c] >= threshold).to(torch.uint8).numpy()

            # 保存为可视化图片：0 -> 黑色，1 -> 白色
            binary_mask_vis = binary_mask * 255

            save_path = save_dir / f'class_{c:02d}_mask.png'
            Image.fromarray(binary_mask_vis, mode='L').save(save_path)


                
                

        # if self.with_auxiliary_head:
        #     loss_aux = self._auxiliary_head_forward_train(x, data_samples)
        #     losses.update(loss_aux)

        # if self.gen_psl and self.iter>=self.psl_iter:
        #     with torch.no_grad():
        #         x_t = self.teacher.backbone(inputs)
        #         text_emb_t = self.text_encoder(self.texts, context=self.teacher.contexts).expand(b_size,-1,-1)

        #         # seg_psl, cls_psl = self.teacher.decode_head.loss(x_t, data_samples, text_emb_t, self.train_cfg, gen_psl=True)
        #         seg_logits_t, cls_logits_t, _, t_lr_feat, t_decode_feat  = self.teacher.decode_head.forward(x_t, img_size, text_emb_t)
        #         seg_psl, cls_psl = self.teacher.decode_head.get_psl_labels(seg_logits_t, cls_logits_t, data_samples)

        #     if self.f_dist_t_loss is not None:
        #         f_dist_t_loss = self.f_dist_t_loss(decode_feat, t_decode_feat)
        #         losses.update(f_dist_t_loss)
            
        #     if self.f_dist_t_low_loss is not None:
        #         f_dist_t_low_loss = self.f_dist_t_low_loss(lr_feat, t_lr_feat)
        #         losses.update(f_dist_t_low_loss)


        # if self.gen_psl and self.iter>=self.psl_iter:

        #     filter_cls_psl = psl_filter(dino_cls_psl, cls_psl, ignore_index=-1)
        #     seg_dino_probs = torch.sigmoid(seg_dino_logits)
        #     seg_probs_t = torch.sigmoid(seg_logits_t)
        #     seg_dino_entropy = binary_entropy(seg_dino_probs)
        #     seg_t_entropy = binary_entropy(seg_probs_t)                                        
        #     seg_fuse_psl = uncertainty_filter_pseudo_label(dino_seg_psl, seg_psl, seg_dino_entropy, seg_t_entropy, uncertainty_threshold=self.uncertainty_threshold, ignore_index=255, high_uncertain_with_ignore_index=self.seg_high_uncertain_with_ignore_index)

        #     if self.cls_psl_uncertain_filter_button:
        #         cls_dino_probs = torch.sigmoid(cls_dino_logits)
        #         cls_probs_t = torch.sigmoid(cls_logits_t)
        #         cls_dino_entropy = binary_entropy(cls_dino_probs)
        #         cls_t_entropy = binary_entropy(cls_probs_t)
        #         cls_fuse_psl = uncertainty_filter_pseudo_label(dino_cls_psl, cls_psl, cls_dino_entropy, cls_t_entropy, uncertainty_threshold=self.uncertainty_threshold, ignore_index=-1, high_uncertain_with_ignore_index=self.cls_high_uncertain_with_ignore_index)
        #         # if self.normal_cls_ignore:
        #         #     cls_fuse_psl[:, 0] = -1  #将normal类的伪标签都设为-1

        #         # cols = [0, 3, 4, 5]
        #         # cls_fuse_psl[:, cols] = filter_cls_psl[:, cols].to(cls_fuse_psl.dtype)

        #         loss_cls_psl = self.cls_psl_loss(cls_logits, cls_fuse_psl)
        #     else:
        #         loss_cls_psl = self.cls_psl_loss(cls_logits, filter_cls_psl)

        #     loss_seg_psl = self.seg_psl_loss(seg_logits, seg_fuse_psl)

        

        #     if self.seg_lr_psl_loss is not None:
        #         lr_seg_fuse_psl = resize(seg_fuse_psl, lr_logits.shape[2:], mode='nearest').long()
        #         loss_seg_lr_psl = self.seg_lr_psl_loss(lr_logits, lr_seg_fuse_psl)
        #         if isinstance(loss_seg_lr_psl, dict):
        #             losses.update(loss_seg_lr_psl)
        #         else:
        #             losses[self.seg_lr_psl_loss.loss_name] = loss_seg_lr_psl




    def inference(self, inputs: Tensor,
                  batch_img_metas: List[dict]):
        assert self.test_cfg.get('mode', 'whole') in ['whole'], \
            'Only "whole" test mode is supported for seg-cls.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARNING)
        return self.encode_decode(inputs, batch_img_metas)

    # def _get_dim8_pred_save_path(self, data_sample: SegDataSample,
    #                              sample_idx: int):
    #     img_meta = data_sample.metainfo
    #     img_path = img_meta.get('img_path', None)
    #     if img_path is not None:
    #         path = Path(img_path).with_suffix('.png')
    #         safe_parts = [
    #             part.replace(':', '').replace('\\', '_').replace('/', '_')
    #             for part in path.parts[-4:]
    #         ]
    #         filename = '__'.join(safe_parts)
    #     else:
    #         filename = f'{sample_idx:06d}.png'
    #     return Path(self.dim8_pred_save_dir) / 'images' / filename

    # def _save_dim8_prediction(self, i_seg_prob: Tensor,
    #                           data_sample: SegDataSample,
    #                           sample_idx: int):
    #     if not self.save_dim8_pred:
    #         return
    #     if i_seg_prob.shape[0] <= self.dim8_pred_channel:
    #         raise ValueError(
    #             f'dim8_pred_channel={self.dim8_pred_channel} is out of '
    #             f'range for prediction with {i_seg_prob.shape[0]} channels.')

    #     channel_prob = i_seg_prob[self.dim8_pred_channel].detach().cpu()
    #     if self.dim8_pred_binary_threshold is not None:
    #         channel_prob = (
    #             channel_prob >= self.dim8_pred_binary_threshold).to(torch.uint8)
    #         pred_array = channel_prob.numpy() * 255
    #     else:
    #         pred_array = (
    #             channel_prob.clamp(0, 1).numpy() * 255).astype(np.uint8)

    #     save_path = self._get_dim8_pred_save_path(data_sample, sample_idx)
    #     save_path.parent.mkdir(parents=True, exist_ok=True)
    #     Image.fromarray(pred_array, mode='L').save(save_path)


    def postprocess_result(self,
                           seg_logits: Tensor,
                           cls_logits: Optional[Tensor],
                           data_samples: OptSampleList = None) -> SampleList:
        batch_size, _, h_img, w_img = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                if 'img_padding_size' in img_meta:
                    padding_size = img_meta['img_padding_size']
                else:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                padding_left, padding_right, padding_top, padding_bottom = padding_size
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:h_img - padding_bottom,
                                          padding_left:w_img - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)

                # save_class_mask(i_seg_logits, 8)

                # print(i_seg_logits.shape)
            else:
                i_seg_logits = seg_logits[i]

            i_seg_prob = i_seg_logits.sigmoid()


            if self.vis_pred:
                self._save_all_channel_binary_masks(
                    i_seg_prob=i_seg_prob,
                    data_sample=data_samples[i],
                    sample_idx=i,
                    save_root='./pred_mask_vis',
                    threshold=0.5
                )

            # self._save_dim8_prediction(i_seg_prob, data_samples[i], i)

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg': PixelData(**{'data': i_seg_prob})
            })

            if cls_logits is not None:
                data_samples[i].set_field(
                    cls_logits[i],
                    'pred_cls',
                    dtype=torch.Tensor)

        return data_samples

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:

        # print(data_samples)
        
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits, cls_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, cls_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tuple[Tensor]:
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)
