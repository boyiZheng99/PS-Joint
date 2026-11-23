# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from sklearn.metrics import average_precision_score

from mmseg.registry import METRICS
from .seg_cls_metric import compute_classification_metrics


def _sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def _softmax(x, axis=None):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


@METRICS.register_module()
class ClsAUPRMetric(BaseMetric):
    """Classification metric with mAUPR support (macro one-vs-rest)."""

    def __init__(self,
                 cls_mode: str = 'multiclass',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.cls_mode = cls_mode

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            cls_pred = data_sample.get('pred_cls', None)
            cls_gt = data_sample.get('gt_label', None)
            if cls_pred is None or cls_gt is None:
                continue
            if not data_sample.get('has_cls', True):
                continue
            if isinstance(cls_pred, torch.Tensor):
                cls_pred = cls_pred.detach().cpu().numpy()
            if isinstance(cls_gt, torch.Tensor):
                cls_gt = cls_gt.detach().cpu().numpy()
            self.results.append(dict(cls_pred=cls_pred, cls_gt=cls_gt))

    def _compute_maupr(self, cls_preds: np.ndarray,
                       cls_gts: np.ndarray) -> Optional[float]:
        num_classes = cls_gts.shape[1]
        ap_list = []

        if self.cls_mode == 'multiclass':
            valid_mask = (cls_gts != -1).any(axis=1)
            if not valid_mask.any():
                return None
            preds = cls_preds[valid_mask]
            gts = cls_gts[valid_mask]
            probs = _softmax(preds, axis=1)
            gt_idx = gts.argmax(axis=1)
            for c in range(num_classes):
                y_true = (gt_idx == c).astype(int)
                if y_true.sum() == 0:
                    continue
                ap_list.append(average_precision_score(y_true, probs[:, c]))
        else:
            probs = _sigmoid(cls_preds)
            for c in range(num_classes):
                valid_mask = cls_gts[:, c] != -1
                if valid_mask.sum() == 0:
                    continue
                y_true = cls_gts[valid_mask, c].astype(int)
                if y_true.sum() == 0:
                    continue
                ap_list.append(average_precision_score(y_true, probs[valid_mask, c]))

        if not ap_list:
            return None
        return float(np.mean(ap_list))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if not results:
            return {}

        cls_preds = np.stack([r['cls_pred'] for r in results], axis=0)
        cls_gts = np.stack([r['cls_gt'] for r in results], axis=0)

        metrics = OrderedDict()
        cls_metrics, valid_classes = compute_classification_metrics(
            cls_preds, cls_gts, mode=self.cls_mode)

        cls_names = self.dataset_meta.get('cls_classes', None)
        if cls_names is None:
            cls_names = [str(i) for i in range(cls_preds.shape[1])]

        if cls_metrics:
            print_log('Per-class classification results:', logger)
            for i, m in zip(valid_classes, cls_metrics):
                print_log(
                    f'  Class {cls_names[i]}: Acc={m["Acc"]:.3f}, '
                    f'F1={m["F1"]:.3f}, Prec={m["Prec"]:.3f}, '
                    f'Rec={m["Rec"]:.3f}, Kappa={m["kappa"]:.3f}, '
                    f'QWK={m["weighted_kappa"]:.3f}',
                    logger)

            metrics['cls_acc'] = float(np.mean([m['Acc'] for m in cls_metrics])) * 100
            metrics['cls_f1'] = float(np.mean([m['F1'] for m in cls_metrics])) * 100
            metrics['cls_prec'] = float(np.mean([m['Prec'] for m in cls_metrics])) * 100
            metrics['cls_rec'] = float(np.mean([m['Rec'] for m in cls_metrics])) * 100
            metrics['cls_kappa'] = float(np.mean([m['kappa'] for m in cls_metrics]))
            metrics['cls_qwk'] = float(np.mean([m['weighted_kappa'] for m in cls_metrics]))

        maupr = self._compute_maupr(cls_preds, cls_gts)
        if maupr is not None:
            metrics['cls_mAUPR'] = maupr * 100

        return metrics
