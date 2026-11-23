# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Sequence

import numpy as np
from PIL import Image

from mmseg.registry import METRICS
from .iou_metric import IoUMetric


@METRICS.register_module()
class IoUMetricIgnoreMissing(IoUMetric):
    """IoU metric that skips samples without segmentation labels."""

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            has_seg = data_sample.get('has_seg', True)
            if not self.format_only and has_seg:
                gt_sem_seg = data_sample.get('gt_sem_seg', None)
                if gt_sem_seg is not None:
                    label = gt_sem_seg['data'].squeeze().to(pred_label)
                    self.results.append(
                        self.intersect_and_union(
                            pred_label, label, num_classes, self.ignore_index))
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)
