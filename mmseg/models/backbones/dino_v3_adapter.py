import torch
from mmengine.model import BaseModule
from mmseg.registry import MODELS
import torch.nn as nn
from mmseg.tool_libs.segdino.dino_lora.LoRA import LoRA
import math
from mmseg.tool_libs.dinov3_adapter.dinov3_adapter import DINOv3_Adapter

@MODELS.register_module()
class Dinov3Adapter(BaseModule):
    def __init__(self, repo_dir, dino_ckpt, dino_size='b',
                 use_injector=False, spm_proj=96):
        super().__init__()
        if dino_size == "b":
            backbone = torch.hub.load(repo_dir, 'dinov3_vitb16', source='local', weights=dino_ckpt)
        elif dino_size =="s":
            backbone = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=dino_ckpt)
        else:
            backbone = torch.hub.load(repo_dir, 'dinov3_vitl16', source='local', weights=dino_ckpt)


        self.dino_size = dino_size

        self.intermediate_layer_idx = {
            's': [2, 5, 8, 11],
            'b': [2, 5, 8, 11], 
            "l": [5, 11, 17, 23],
        }

        self.backbone = DINOv3_Adapter(
            backbone=backbone,
            interaction_indexes=self.intermediate_layer_idx[dino_size],   # 按你的模型层数改
            with_cp=False,
            deform_num_heads=6,
            use_injector=use_injector,
            spm_proj=spm_proj,
        )

        self.backbone.eval()



    def forward(self, x, extra_spm=True, spa_prior=None):
        # patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        # features = self.backbone.get_intermediate_layers(
        #     x, n = self.intermediate_layer_idx[self.dino_size]
        #     )
        
        features = self.backbone(x, extra_spm, spa_prior)

        out = []

        for value in features.values():
            out.append(value)

        # for i, x in enumerate(features):
        #     x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
        #     out.append(x)

        return out

        # if not self.return_class_token:
        #     return features

        # else:
        #     patch_features = []
        #     cls_tokens = []
        #     for feat in features:
        #         patch_token, cls_token = features
        #         patch_features.append(patch_token)
        #         cls_tokens.append(cls_token)

        #     return [patch_features , cls_tokens]




