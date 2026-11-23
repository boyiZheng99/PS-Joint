# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class Blur(BaseTransform):
    """Apply Gaussian blur with a random sigma."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def transform(self, results: dict) -> dict:
        img = results['img']
        img = Image.fromarray(img.astype(np.uint8))
        if random.random() < self.p:
            sigma = np.random.uniform(0.1, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        results['img'] = np.array(img)
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


@TRANSFORMS.register_module()
class ColorJitter(BaseTransform):
    """Apply color jitter using torchvision."""

    def __init__(self, p: float = 0.8) -> None:
        self.p = p
        self._jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)

    def transform(self, results: dict) -> dict:
        if random.random() < self.p:
            img = results['img']
            img = Image.fromarray(img.astype(np.uint8))
            img = self._jitter(img)
            results['img'] = np.array(img).astype(np.float32)
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'
