from collections import OrderedDict
import json
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             precision_score, recall_score, roc_auc_score, auc)

from mmseg.registry import METRICS


@METRICS.register_module()
class NoSegGTClsMetric(BaseMetric):
    """Joint segmentation + classification metric for DDR."""

    def __init__(self,
                 target_classes = ('Drusen', 'Scar', 'Detachment'),
                 area_threshold = 0,
                 ignore_index: int = 255,
                 nan_to_num: Optional[float] = 0,
                 compute_all_black: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 keep_results: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.compute_all_black = compute_all_black
        self.target_classes = target_classes
        self.area_threshold = area_threshold
        self.output_dir = output_dir
        self.keep_results = keep_results
        if self.output_dir is not None:
            import os
            os.makedirs(self.output_dir, exist_ok=True)

        # self.cls_mode = cls_mode

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:

        for sample in data_samples:
         
            pred_seg = sample['pred_sem_seg']['data'].detach().cpu().numpy()
            classes = self.dataset_meta.get('classes', None)
            target_indices = [classes.index(c) for c in self.target_classes]
            pred_seg = pred_seg[target_indices]

            binary_mask = pred_seg > 0.5
            pixel_count = binary_mask.sum(axis=(1, 2))
            pred_cls = (pixel_count > self.area_threshold).astype(np.int32)

            gt_cls = sample['gt_label'].detach().cpu().numpy()

            self.results.append({
                'pred': pred_cls,
                'gt': gt_cls
            })
            
    def compute_metrics(self, results):

        logger: MMLogger = MMLogger.get_current_instance()
        
        if len(results) == 0:
            return {}

        cls_preds = np.stack([r['pred'] for r in results], axis=0)
        cls_gts = np.stack([r['gt'] for r in results], axis=0)

        metrics = {}
        metrics_per_class = []

        
        for i in range(len(self.target_classes)):
            pred_col = cls_preds[:, i]
            gt_col = cls_gts[:, i]

            acc = accuracy_score(gt_col, pred_col)
            # Every retinal classification target in this run, including DR,
            # is binary.  F1/precision/recall therefore use positive class 1.
            f1 = f1_score(gt_col, pred_col, average='binary', pos_label=1,
                          zero_division=0)
            prec = precision_score(gt_col, pred_col, average='binary', pos_label=1,
                                   zero_division=0)
            rec = recall_score(gt_col, pred_col, average='binary', pos_label=1,
                               zero_division=0)
            kappa = cohen_kappa_score(gt_col, pred_col)
            qwk = cohen_kappa_score(gt_col, pred_col, weights='quadratic')

            metrics_per_class.append(dict(
                Acc=acc, F1=f1, Prec=prec, Rec=rec, kappa=kappa, weighted_kappa=qwk
            ))

        print_log("Per-class classification results:", logger)
        for i, m in enumerate(metrics_per_class):
            print_log(f"Class {self.target_classes[i]}: Acc={m['Acc']*100:.3f}, F1={m['F1']*100:.3f}, "
                    f"Prec={m['Prec']*100:.3f}, Rec={m['Rec']*100:.3f}, Kappa={m['kappa']*100:.3f}, weight_Kappa={m['weighted_kappa']*100:.3f}", logger)

        per_class = OrderedDict(
            (str(self.target_classes[i]), OrderedDict(
                Acc=float(m['Acc'] * 100),
                F1=float(m['F1'] * 100),
                Prec=float(m['Prec'] * 100),
                Rec=float(m['Rec'] * 100),
                Kappa=float(m['kappa'] * 100),
                QWK=float(m['weighted_kappa'] * 100),
            ))
            for i, m in enumerate(metrics_per_class))


        metrics['cls_acc'] = float(np.mean([m['Acc'] for m in metrics_per_class])) * 100
        metrics['cls_f1'] = float(np.mean([m['F1'] for m in metrics_per_class])) * 100
        metrics['cls_prec'] = float(np.mean([m['Prec'] for m in metrics_per_class])) * 100
        metrics['cls_rec'] = float(np.mean([m['Rec'] for m in metrics_per_class])) * 100
        metrics['cls_kappa'] = float(np.mean([m['kappa'] for m in metrics_per_class]))
        metrics['cls_qwk'] = float(np.mean([m['weighted_kappa'] for m in metrics_per_class]))

        if self.output_dir is not None:
            metrics_path = osp.join(self.output_dir, 'metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                payload = {k: float(v) for k, v in metrics.items()}
                payload['per_class'] = {'classification': per_class}
                json.dump(payload, f,
                          indent=2, sort_keys=True)
            print_log(f'Test metrics saved to {metrics_path}', logger)

        return metrics

 
