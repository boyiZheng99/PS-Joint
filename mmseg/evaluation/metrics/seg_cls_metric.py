# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import json
from typing import Dict, List, Optional, Sequence, Tuple

import os
import os.path as osp

import mmcv

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             precision_score, recall_score, roc_auc_score, auc)

from mmseg.registry import METRICS


SIGMOID_THRESH = 0.5


def _sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def _softmax(x, axis=None):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _safe_kappa(y_true, y_pred, weights=None):
    # Avoid sklearn warnings when only one label is present.
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if labels.size < 2:
        return 1.0 if np.all(y_true == y_pred) else 0.0
    return cohen_kappa_score(y_true, y_pred, weights=weights)


def _compute_confusion(pred, target, thresh=SIGMOID_THRESH, compute_all_black=True):
    valid = (target != 255)
    if valid.sum() == 0:
        return None

    if not compute_all_black and target.sum() == 0:
        return None

    pred_bin = (pred > thresh).astype(np.uint8)[valid]
    label = (target > 0).astype(np.uint8)[valid]

    tp = np.sum((pred_bin == 1) & (label == 1))
    p = np.sum(pred_bin == 1)
    fn = np.sum(label == 1) - tp
    diff = np.abs(pred_bin - label)
    mae_sum = diff.sum()
    mae_count = diff.size
    return p, tp, fn, mae_sum, mae_count


def _sigmoid_metrics(preds, targets, num_classes, compute_all_black=True):
    total_p = np.zeros((num_classes,), dtype=np.float64)
    total_tp = np.zeros((num_classes,), dtype=np.float64)
    total_fn = np.zeros((num_classes,), dtype=np.float64)
    mae_sum = np.zeros((num_classes,), dtype=np.float64)
    mae_count = np.zeros((num_classes,), dtype=np.float64)

    threshs = np.linspace(0, 1, 11)
    maupr = np.zeros((num_classes,), dtype=np.float64)

    for c in range(num_classes):
        p_list, tp_list, fn_list = [], [], []
        for thresh in threshs:
            p_c, tp_c, fn_c = 0, 0, 0
            for pred, target in zip(preds, targets):
                res = _compute_confusion(pred[c], target[c], thresh, compute_all_black)
                if res is None:
                    continue
                p, tp, fn, ms, mc = res
                p_c += p
                tp_c += tp
                fn_c += fn

                if np.isclose(thresh, SIGMOID_THRESH):
                    mae_sum[c] += ms
                    mae_count[c] += mc
                    total_p[c] += p
                    total_tp[c] += tp
                    total_fn[c] += fn
            p_list.append(p_c)
            tp_list.append(tp_c)
            fn_list.append(fn_c)

        if mae_count[c] > 0:
            p_arr = np.array(p_list)
            tp_arr = np.array(tp_list)
            fn_arr = np.array(fn_list)
            ppv = np.divide(
                tp_arr,
                p_arr,
                out=np.ones_like(tp_arr, dtype=np.float64),
                where=p_arr > 0)
            sens = np.divide(
                tp_arr,
                tp_arr + fn_arr,
                out=np.zeros_like(tp_arr, dtype=np.float64),
                where=(tp_arr + fn_arr) > 0)
            if ppv.sum() > 0 and sens.sum() > 0:
                maupr[c] = auc(sens, ppv)

    total_mae = np.divide(mae_sum, np.maximum(mae_count, 1))
    valid_mask = mae_count > 0
    return total_p, total_tp, total_fn, maupr, total_mae, valid_mask


def _roc(preds, targets, num_classes, compute_all_black):
    auc_roc = np.zeros((num_classes,), dtype=np.float32)
    count = np.zeros((num_classes,), dtype=np.int32)

    for c in range(num_classes):
        for pred, target in zip(preds, targets):
            # print(pred[c].shape)
            # print(target[c].shape)
            pred_c = pred[c].flatten()
            target_c = target[c].flatten()
            valid = (target_c != 255)
            if valid.sum() == 0:
                continue
            pred_c = pred_c[valid]
            target_c = target_c[valid]

            # ROC AUC is undefined when only one class is present.
            if np.unique(target_c).size < 2:
                continue
            if not compute_all_black and target_c.sum() == 0:
                continue
            try:
                auc_c = roc_auc_score(target_c, pred_c)
                auc_roc[c] += auc_c
                count[c] += 1
            except ValueError:
                continue
    return np.divide(auc_roc, np.maximum(count, 1))


def multi_lesions_metrics(results, gt_seg_maps, num_classes, compute_all_black=True, nan_to_num=None):
    auc_roc = _roc(results, gt_seg_maps, num_classes, compute_all_black)
    total_p, total_tp, total_fn, maupr, mae, valid_mask = _sigmoid_metrics(
        results, gt_seg_maps, num_classes, compute_all_black)

    ppv = total_tp / total_p
    se = total_tp / (total_tp + total_fn)
    fscore = np.divide(
        2 * ppv * se,
        ppv + se,
        out=np.zeros_like(ppv, dtype=np.float64),
        where=(ppv + se) > 0)
    dice = 2 * total_tp / (total_p + total_tp + total_fn)
    iou = total_tp / (total_p + total_fn)

    ret_metrics = OrderedDict()
    ret_metrics['IoU'] = iou
    ret_metrics['AUPR'] = maupr
    ret_metrics['se'] = se
    ret_metrics['sp'] = ppv
    ret_metrics['Fscore'] = fscore
    ret_metrics['dice'] = dice
    ret_metrics['auc_roc'] = auc_roc
    ret_metrics['mae'] = mae

    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics, valid_mask


def compute_classification_metrics(cls_preds, cls_gts, if_include_multi_class=True):

    num_classes = cls_gts.shape[1]
    metrics_per_class = []
    valid_classes = []

    # print(if_include_multi_class)

    for i in range(num_classes):
        gt_col = cls_gts[:, i]
        if i != (num_classes-1):
            pred_col = cls_preds[:, i]
            pred_col = _sigmoid(pred_col)
            pred_col = (pred_col > 0.5).astype(int)
        else:
            if if_include_multi_class:
                pred_grad = cls_preds[:, i:]  #grading预测
                pred_grad = _softmax(pred_grad, axis=1)
                pred_col = np.argmax(pred_grad, axis=1)
                # A five-grade DR model is also evaluated on OOD datasets
                # whose single D column is binary. In that case, map the
                # predicted grade 0 to no DR and grades 1..4 to DR. True DDR
                # grading datasets contain labels above 1 and retain their
                # original five-class evaluation.
                known_targets = gt_col[gt_col != -1]
                if (known_targets.size > 0 and
                        np.all(np.isin(known_targets, (0, 1)))):
                    pred_col = (pred_col > 0).astype(int)
            else:
                # A binary-label test set can also be evaluated with a
                # five-grade DR checkpoint.  Such a checkpoint has four
                # extra logits: the final five outputs are No_DR and grades
                # 1--4.  Collapse the predicted grade to binary DR here.
                # A regular binary checkpoint has the same width as the GT
                # and keeps the original sigmoid path.
                if cls_preds.shape[1] == num_classes + 4:
                    pred_grad = _softmax(cls_preds[:, i:], axis=1)
                    pred_col = (np.argmax(pred_grad, axis=1) > 0).astype(int)
                else:
                    pred_col = cls_preds[:, i]
                    pred_col = _sigmoid(pred_col)
                    pred_col = (pred_col > 0.5).astype(int)
                # print(pred_col)


        # if np.all(gt_col == -1):  #跳过全为-1的类别
        #     continue
        mask = gt_col != -1

        # 如果这个类别全是 -1，跳过
        if np.sum(mask) == 0:
            continue

        gt_valid = gt_col[mask]
        pred_valid = pred_col[mask]

        acc = accuracy_score(gt_valid, pred_valid)
        # Every retinal classification task in this experiment, including
        # DR, is binary: 0=no disease and 1=disease.  Report only class 1.
        f1 = f1_score(gt_valid, pred_valid, average='binary', pos_label=1,
                      zero_division=0)
        prec = precision_score(gt_valid, pred_valid, average='binary', pos_label=1,
                               zero_division=0)
        rec = recall_score(gt_valid, pred_valid, average='binary', pos_label=1,
                           zero_division=0)
        kappa = cohen_kappa_score(gt_valid, pred_valid)
        qwk = cohen_kappa_score(gt_valid, pred_valid, weights='quadratic')

    
        metrics_per_class.append(dict(
            Acc=acc, F1=f1, Prec=prec, Rec=rec, kappa=kappa, weighted_kappa=qwk
        ))

        valid_classes.append(i)

    return metrics_per_class, valid_classes


# def compute_classification_metrics(cls_preds, cls_gts, mode='multilabel'):
#     num_classes = cls_gts.shape[1]
#     metrics_per_class = []
#     valid_classes = []

#     if mode == 'multiclass':
#         valid_mask = (cls_gts != -1).any(axis=1)
#         if not valid_mask.any():
#             return metrics_per_class, valid_classes
#         gt_idx = cls_gts[valid_mask].argmax(axis=1)
#         pred_idx = _softmax(cls_preds[valid_mask], axis=1).argmax(axis=1)
#         for i in range(num_classes):
#             gt_binary = (gt_idx == i)
#             pred_binary = (pred_idx == i)
#             acc = accuracy_score(gt_binary, pred_binary)
#             f1 = f1_score(gt_binary, pred_binary, average='binary', zero_division=0)
#             prec = precision_score(gt_binary, pred_binary, average='binary', zero_division=0)
#             rec = recall_score(gt_binary, pred_binary, average='binary', zero_division=0)
#             kappa = _safe_kappa(gt_binary, pred_binary)
#             qwk = _safe_kappa(gt_binary, pred_binary, weights='quadratic')
#             metrics_per_class.append(dict(
#                 Acc=acc, F1=f1, Prec=prec, Rec=rec, kappa=kappa, weighted_kappa=qwk
#             ))
#             valid_classes.append(i)
#         return metrics_per_class, valid_classes

#     for i in range(num_classes):
#         gt_col = cls_gts[:, i]
#         valid_mask = gt_col != -1
#         if valid_mask.sum() == 0:
#             continue
#         pred_col = _sigmoid(cls_preds[:, i])[valid_mask]
#         pred_col = (pred_col > 0.5).astype(int)
#         gt_valid = gt_col[valid_mask].astype(int)

#         acc = accuracy_score(gt_valid, pred_col)
#         f1 = f1_score(gt_valid, pred_col, average='binary', zero_division=0)
#         prec = precision_score(gt_valid, pred_col, average='binary', zero_division=0)
#         rec = recall_score(gt_valid, pred_col, average='binary', zero_division=0)
#         kappa = _safe_kappa(gt_valid, pred_col)
#         qwk = _safe_kappa(gt_valid, pred_col, weights='quadratic')

#         metrics_per_class.append(dict(
#             Acc=acc, F1=f1, Prec=prec, Rec=rec, kappa=kappa, weighted_kappa=qwk
#         ))
#         valid_classes.append(i)

#     return metrics_per_class, valid_classes


# def _compute_overall_kappa(cls_preds: np.ndarray,
#                            cls_gts: np.ndarray,
#                            mode: str) -> Tuple[Optional[float], Optional[float]]:
#     if cls_preds.size == 0 or cls_gts.size == 0:
#         return None, None
#     if mode == 'multiclass':
#         valid_mask = (cls_gts != -1).any(axis=1)
#         if not valid_mask.any():
#             return None, None
#         gt_idx = cls_gts[valid_mask].argmax(axis=1)
#         pred_idx = _softmax(cls_preds[valid_mask], axis=1).argmax(axis=1)
#         return _safe_kappa(gt_idx, pred_idx), _safe_kappa(gt_idx, pred_idx, weights='quadratic')

#     valid_mask = cls_gts != -1
#     if not valid_mask.any():
#         return None, None
#     gt_flat = cls_gts[valid_mask].astype(int)
#     pred_flat = (_sigmoid(cls_preds) > SIGMOID_THRESH).astype(int)[valid_mask]
#     return _safe_kappa(gt_flat, pred_flat), _safe_kappa(gt_flat, pred_flat, weights='quadratic')

def _compute_seg_stats_one_sample(pred, target, num_classes, compute_all_black=True):
    threshs = np.linspace(0, 1, 11)

    p_05 = np.zeros(num_classes, dtype=np.float64)
    tp_05 = np.zeros(num_classes, dtype=np.float64)
    fn_05 = np.zeros(num_classes, dtype=np.float64)
    mae_sum = np.zeros(num_classes, dtype=np.float64)
    mae_count = np.zeros(num_classes, dtype=np.float64)

    p_all = np.zeros((num_classes, len(threshs)), dtype=np.float64)
    tp_all = np.zeros((num_classes, len(threshs)), dtype=np.float64)
    fn_all = np.zeros((num_classes, len(threshs)), dtype=np.float64)

    valid_mask = np.zeros(num_classes, dtype=bool)

    for c in range(num_classes):
        # for t_idx, thresh in enumerate(threshs):
        res = _compute_confusion(pred[c], target[c], SIGMOID_THRESH, compute_all_black)
        if res is None:
            continue

        p, tp, fn, ms, mc = res
            # p_all[c, t_idx] += p
            # tp_all[c, t_idx] += tp
            # fn_all[c, t_idx] += fn
        valid_mask[c] = True

            # if np.isclose(thresh, SIGMOID_THRESH):
        p_05[c] += p
        tp_05[c] += tp
        fn_05[c] += fn
        mae_sum[c] += ms
        mae_count[c] += mc

    return dict(
        p_05=p_05,
        tp_05=tp_05,
        fn_05=fn_05,
        mae_sum=mae_sum,
        mae_count=mae_count,
        # p_all=p_all,
        # tp_all=tp_all,
        # fn_all=fn_all,
        valid_mask=valid_mask
    )


@METRICS.register_module()
class SegClsMetric(BaseMetric):
    """Joint segmentation + classification metric for DDR."""

    def __init__(self,
                 ignore_index: int = 255,
                 nan_to_num: Optional[float] = 0,
                 compute_all_black: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 if_include_multi_class=True,
                 output_dir: Optional[str] = None,
                 keep_results: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.compute_all_black = compute_all_black

        self.if_include_multi_class = if_include_multi_class
        self.output_dir = output_dir
        self.keep_results = keep_results
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        # self.cls_mode = cls_mode


        # self.save_vis = True
        # self.vis_save_dir = 'work_dirs/new_pred_gt_vis'
        # self.vis_threshold = 0.5
        # self.vis_classes=('EX', 'HE', 'MA', 'SE')

    # def _save_pred_gt_vis(self, seg_pred: np.ndarray, seg_gt: np.ndarray,
    #                       data_sample: dict) -> None:
    #     """Save thresholded prediction and GT masks for one test sample."""
    #     if seg_pred is None or seg_gt is None:
    #         return

    #     metainfo = data_sample.get('metainfo', {})
    #     img_path = metainfo.get('img_path', metainfo.get('filename', None))
    #     if img_path is None:
    #         stem = f'sample_{len(self.results):06d}'
    #     else:
    #         stem = osp.splitext(osp.basename(img_path))[0]

    #     class_names = getattr(self, 'dataset_meta', {}).get('classes', None)
    #     if class_names is None:
    #         class_names = [str(i) for i in range(seg_gt.shape[0])]

    #     for c in range(seg_gt.shape[0]):
    #         cls_name = str(class_names[c]) if c < len(class_names) else str(c)
    #         if self.vis_classes is not None and cls_name not in self.vis_classes:
    #             continue

    #         pred_dir = osp.join(self.vis_save_dir, cls_name, 'pred')
    #         gt_dir = osp.join(self.vis_save_dir, cls_name, 'gt')
    #         os.makedirs(pred_dir, exist_ok=True)
    #         os.makedirs(gt_dir, exist_ok=True)

    #         pred_mask = (seg_pred[c] > self.vis_threshold).astype(np.uint8) * 255

    #         gt_c = seg_gt[c]
    #         gt_mask = np.zeros((*gt_c.shape, 3), dtype=np.uint8)
    #         gt_mask[gt_c == 0] = (0, 0, 0)
    #         gt_mask[gt_c == 1] = (255, 255, 255)
    #         gt_mask[gt_c == self.ignore_index] = (255, 0, 0)

    #         file_name = f'{stem}.png'
    #         mmcv.imwrite(pred_mask, osp.join(pred_dir, file_name))
    #         mmcv.imwrite(gt_mask, osp.join(gt_dir, file_name))
    
    # def _save_pred_gt_vis(self, seg_pred: np.ndarray, seg_gt: np.ndarray,
    #                       data_sample: dict) -> None:
    #     """Save thresholded prediction and GT masks for one test sample."""
    #     if seg_pred is None or seg_gt is None:
    #         return

    #     metainfo = data_sample.get('metainfo', {})
    #     img_path = metainfo.get('img_path', metainfo.get('filename', None))
    #     if img_path is None:
    #         stem = f'sample_{len(self.results):06d}'
    #     else:
    #         stem = osp.splitext(osp.basename(img_path))[0]

    #     pred_dir = osp.join(self.vis_save_dir, 'pred')
    #     gt_dir = osp.join(self.vis_save_dir, 'gt')
    #     os.makedirs(pred_dir, exist_ok=True)
    #     os.makedirs(gt_dir, exist_ok=True)

    #     class_names = getattr(self, 'dataset_meta', {}).get('classes', None)
    #     if class_names is None:
    #         class_names = [str(i) for i in range(seg_gt.shape[0])]

    #     for c in range(seg_gt.shape[0]):
    #         cls_name = str(class_names[c]) if c < len(class_names) else str(c)
    #         pred_mask = (seg_pred[c] > self.vis_threshold).astype(np.uint8) * 255

    #         gt_c = seg_gt[c]
    #         gt_mask = np.zeros((*gt_c.shape, 3), dtype=np.uint8)
    #         gt_mask[gt_c == 0] = (0, 0, 0)
    #         gt_mask[gt_c == 1] = (255, 255, 255)
    #         gt_mask[gt_c == self.ignore_index] = (255, 0, 0)

    #         file_name = f'{stem}_{c:02d}_{cls_name}.png'
    #         mmcv.imwrite(pred_mask, osp.join(pred_dir, file_name))
    #         mmcv.imwrite(gt_mask, osp.join(gt_dir, file_name))

    # def _save_pred_gt_vis(self, seg_pred: np.ndarray, seg_gt: np.ndarray,
    #                       data_sample: dict) -> None:
    #     """Save thresholded prediction and GT masks for one test sample."""
    #     if seg_pred is None or seg_gt is None:
    #         return

    #     metainfo = data_sample.get('metainfo', {})
    #     img_path = metainfo.get('img_path', metainfo.get('filename', None))
    #     if img_path is None:
    #         stem = f'sample_{len(self.results):06d}'
    #     else:
    #         stem = osp.splitext(osp.basename(img_path))[0]

    #     pred_dir = osp.join(self.vis_save_dir, 'pred')
    #     gt_dir = osp.join(self.vis_save_dir, 'gt')
    #     os.makedirs(pred_dir, exist_ok=True)
    #     os.makedirs(gt_dir, exist_ok=True)

    #     class_names = getattr(self, 'dataset_meta', {}).get('classes', None)
    #     if class_names is None:
    #         class_names = [str(i) for i in range(seg_gt.shape[0])]

    #     for c in range(seg_gt.shape[0]):
    #         cls_name = str(class_names[c]) if c < len(class_names) else str(c)
    #         pred_mask = (seg_pred[c] > self.vis_threshold).astype(np.uint8) * 255

    #         gt_c = seg_gt[c].copy()
    #         gt_c[gt_c == self.ignore_index] = 0
    #         gt_mask = (gt_c > 0).astype(np.uint8) * 255

    #         file_name = f'{stem}_{c:02d}_{cls_name}.png'
    #         mmcv.imwrite(pred_mask, osp.join(pred_dir, file_name))
    #         mmcv.imwrite(gt_mask, osp.join(gt_dir, file_name))
    # def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
    #     for data_sample in data_samples:
    #         seg_pred = data_sample['pred_sem_seg']['data']
    #         has_seg = data_sample.get('has_seg', True)
    #         has_cls = data_sample.get('has_cls', True)
    #         seg_gt = data_sample.get('gt_sem_seg', None)
    #         seg_gt = seg_gt['data'] if seg_gt is not None else None
    #         cls_pred = data_sample.get('pred_cls', None)
    #         cls_gt = data_sample.get('gt_label', None)

    #         if not has_seg:
    #             seg_gt = None
    #         if not has_cls:
    #             cls_gt = None

    #         if isinstance(seg_pred, torch.Tensor):
    #             seg_pred = seg_pred.detach().cpu().numpy()
    #         if isinstance(seg_gt, torch.Tensor):
    #             seg_gt = seg_gt.detach().cpu().numpy()
    #         if isinstance(cls_pred, torch.Tensor):
    #             cls_pred = cls_pred.detach().cpu().numpy()
    #         if isinstance(cls_gt, torch.Tensor):
    #             cls_gt = cls_gt.detach().cpu().numpy()

    #         self.results.append(dict(
    #             seg_pred=seg_pred,
    #             seg_gt=seg_gt,
    #             cls_pred=cls_pred,
    #             cls_gt=cls_gt))

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            # Decoupled classification models intentionally return only
            # ``pred_cls``.  Joint models return both fields.  The metric
            # must therefore evaluate whichever prediction branch exists.
            pred_sem_seg = data_sample.get('pred_sem_seg', None)
            seg_pred = pred_sem_seg['data'] if pred_sem_seg is not None else None
            has_seg = data_sample.get('has_seg', True)
            has_cls = data_sample.get('has_cls', True)
            seg_gt = data_sample.get('gt_sem_seg', None)
            seg_gt = seg_gt['data'] if seg_gt is not None else None
            cls_pred = data_sample.get('pred_cls', None)
            cls_gt = data_sample.get('gt_label', None)

            if not has_seg:
                seg_gt = None
            if not has_cls:
                cls_gt = None

            result = dict()

            if seg_pred is not None and seg_gt is not None:
                if isinstance(seg_pred, torch.Tensor):
                    seg_pred = seg_pred.detach().cpu().numpy()
                if isinstance(seg_gt, torch.Tensor):
                    seg_gt = seg_gt.detach().cpu().numpy()

                # if self.save_vis:
                #     self._save_pred_gt_vis(seg_pred, seg_gt, data_sample)

                num_classes = seg_gt.shape[0]
                seg_stats = _compute_seg_stats_one_sample(
                    seg_pred, seg_gt, num_classes, self.compute_all_black)
                result['seg_stats'] = seg_stats

            if cls_pred is not None and cls_gt is not None:
                if isinstance(cls_pred, torch.Tensor):
                    cls_pred = cls_pred.detach().cpu().numpy()
                if isinstance(cls_gt, torch.Tensor):
                    cls_gt = cls_gt.detach().cpu().numpy()
                result['cls_pred'] = cls_pred
                result['cls_gt'] = cls_gt

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        metrics = OrderedDict()
        details = OrderedDict()

        seg_stats_list = [r['seg_stats'] for r in results if 'seg_stats' in r]
        cls_preds = [r['cls_pred'] for r in results if 'cls_gt' in r]
        cls_gts = [r['cls_gt'] for r in results if 'cls_gt' in r]

        if seg_stats_list:
            num_classes = seg_stats_list[0]['p_05'].shape[0]
            threshs = np.linspace(0, 1, 11)

            total_p = np.zeros(num_classes, dtype=np.float64)
            total_tp = np.zeros(num_classes, dtype=np.float64)
            total_fn = np.zeros(num_classes, dtype=np.float64)
            total_mae_sum = np.zeros(num_classes, dtype=np.float64)
            total_mae_count = np.zeros(num_classes, dtype=np.float64)

            # total_p_all = np.zeros((num_classes, len(threshs)), dtype=np.float64)
            # total_tp_all = np.zeros((num_classes, len(threshs)), dtype=np.float64)
            # total_fn_all = np.zeros((num_classes, len(threshs)), dtype=np.float64)
            valid_mask = np.zeros(num_classes, dtype=bool)

            for s in seg_stats_list:
                total_p += s['p_05']
                total_tp += s['tp_05']
                total_fn += s['fn_05']
                total_mae_sum += s['mae_sum']
                total_mae_count += s['mae_count']
                # total_p_all += s['p_all']
                # total_tp_all += s['tp_all']
                # total_fn_all += s['fn_all']
                valid_mask |= s['valid_mask']

            ppv = np.divide(
                total_tp, total_p,
                out=np.ones_like(total_tp, dtype=np.float64),
                where=total_p > 0)
            se = np.divide(
                total_tp, total_tp + total_fn,
                out=np.zeros_like(total_tp, dtype=np.float64),
                where=(total_tp + total_fn) > 0)
            fscore = np.divide(
                2 * ppv * se, ppv + se,
                out=np.zeros_like(ppv, dtype=np.float64),
                where=(ppv + se) > 0)
            dice = np.divide(
                2 * total_tp, total_p + total_tp + total_fn,
                out=np.zeros_like(total_tp, dtype=np.float64),
                where=(total_p + total_tp + total_fn) > 0)
            iou = np.divide(
                total_tp, total_p + total_fn,
                out=np.zeros_like(total_tp, dtype=np.float64),
                where=(total_p + total_fn) > 0)
            mae = np.divide(
                total_mae_sum, np.maximum(total_mae_count, 1))

            # maupr = np.zeros(num_classes, dtype=np.float64)
            # for c in range(num_classes):
            #     if total_mae_count[c] <= 0:
            #         continue
            #     ppv_curve = np.divide(
            #         total_tp_all[c], total_p_all[c],
            #         out=np.ones_like(total_tp_all[c], dtype=np.float64),
            #         where=total_p_all[c] > 0)
            #     se_curve = np.divide(
            #         total_tp_all[c], total_tp_all[c] + total_fn_all[c],
            #         out=np.zeros_like(total_tp_all[c], dtype=np.float64),
            #         where=(total_tp_all[c] + total_fn_all[c]) > 0)
            #     if ppv_curve.sum() > 0 and se_curve.sum() > 0:
            #         maupr[c] = auc(se_curve, ppv_curve)

            ret_metrics = OrderedDict(
                IoU=iou,
                # AUPR=maupr,
                se=se,
                sp=ppv,
                Fscore=fscore,
                dice=dice,
                mae=mae,
            )

            class_names = self.dataset_meta.get('classes', None)
            if class_names is None:
                class_names = [str(i) for i in range(num_classes)]

            summary_str = 'Per-class segmentation results:\n'
            line_format = '{:<15} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n'
            summary_str += line_format.format(
                'Class', 'IoU', 'Dice', 'Fscore', 'PPV', 'S', 'MAE')
            for i in range(num_classes):
                if not valid_mask[i]:
                    continue
                summary_str += line_format.format(
                    class_names[i],
                    f'{ret_metrics["IoU"][i]*100:.3f}',
                    f'{ret_metrics["dice"][i]*100:.3f}',
                    f'{ret_metrics["Fscore"][i]*100:.3f}',
                    f'{ret_metrics["sp"][i]*100:.3f}',
                    f'{ret_metrics["se"][i]*100:.3f}',
                    # f'{ret_metrics["AUPR"][i]*100:.3f}',
                    f'{ret_metrics["mae"][i]*100:.4f}')
            print_log(summary_str, logger)

            seg_details = OrderedDict()
            for i, class_name in enumerate(class_names):
                if not valid_mask[i]:
                    continue
                seg_details[str(class_name)] = OrderedDict(
                    IoU=float(ret_metrics['IoU'][i] * 100),
                    Dice=float(ret_metrics['dice'][i] * 100),
                    Fscore=float(ret_metrics['Fscore'][i] * 100),
                    PPV=float(ret_metrics['sp'][i] * 100),
                    S=float(ret_metrics['se'][i] * 100),
                    MAE=float(ret_metrics['mae'][i] * 100),
                )
            details['segmentation'] = seg_details

            for key, val in ret_metrics.items():
                val = np.where(valid_mask, val, np.nan)
                metrics[f'seg_m{key}'] = float(np.nanmean(val)) * 100

        if cls_preds and cls_gts:
            cls_preds = np.stack(cls_preds, axis=0)
            cls_gts = np.stack(cls_gts, axis=0)
            cls_metrics, valid_classes = compute_classification_metrics(
                cls_preds, cls_gts, self.if_include_multi_class)

            cls_names = self.dataset_meta.get('cls_classes', None)
            if cls_names is None:
                cls_names = [str(i) for i in range(cls_preds.shape[1])]

            print_log('Per-class classification results:', logger)
            for i, m in zip(valid_classes, cls_metrics):
                print_log(
                    f'Class {cls_names[i]}: Acc={m["Acc"]*100:.3f}, '
                    f'F1={m["F1"]*100:.3f}, Prec={m["Prec"]*100:.3f}, '
                    f'Rec={m["Rec"]*100:.3f}, Kappa={m["kappa"]*100:.3f}, '
                    f'QWK={m["weighted_kappa"]*100:.3f}',
                    logger)

            cls_names = list(cls_names)
            details['classification'] = OrderedDict(
                (str(cls_names[i]), OrderedDict(
                    Acc=float(m['Acc'] * 100),
                    F1=float(m['F1'] * 100),
                    Prec=float(m['Prec'] * 100),
                    Rec=float(m['Rec'] * 100),
                    Kappa=float(m['kappa'] * 100),
                    QWK=float(m['weighted_kappa'] * 100),
                ))
                for i, m in zip(valid_classes, cls_metrics))

            if cls_metrics:
                metrics['cls_acc'] = float(np.mean([m['Acc'] for m in cls_metrics])) * 100
                metrics['cls_f1'] = float(np.mean([m['F1'] for m in cls_metrics])) * 100
                metrics['cls_prec'] = float(np.mean([m['Prec'] for m in cls_metrics])) * 100
                metrics['cls_rec'] = float(np.mean([m['Rec'] for m in cls_metrics])) * 100
                # Keep summary classification metrics in percentage units,
                # matching the per-class log lines and the other cls_* values.
                metrics['cls_kappa'] = float(np.mean([m['kappa'] for m in cls_metrics])) * 100
                metrics['cls_qwk'] = float(np.mean([m['weighted_kappa'] for m in cls_metrics])) * 100

        if self.output_dir is not None:
            metrics_path = osp.join(self.output_dir, 'metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                payload = {k: float(v) for k, v in metrics.items()}
                payload['per_class'] = details
                json.dump(payload, f,
                          indent=2, sort_keys=True)
            print_log(f'Test metrics saved to {metrics_path}', logger)

        return metrics

    # def compute_metrics(self, results: list) -> Dict[str, float]:
    #     logger: MMLogger = MMLogger.get_current_instance()
    #     seg_preds = [r['seg_pred'] for r in results if r['seg_gt'] is not None]
    #     seg_gts = [r['seg_gt'] for r in results if r['seg_gt'] is not None]
    #     cls_preds = [r['cls_pred'] for r in results if r['cls_gt'] is not None]
    #     cls_gts = [r['cls_gt'] for r in results if r['cls_gt'] is not None]

    #     metrics = OrderedDict()
    #     if seg_preds and seg_gts:
    #         num_classes = seg_gts[0].shape[0]
    #         ret_metrics, valid_mask = multi_lesions_metrics(
    #             seg_preds, seg_gts, num_classes, compute_all_black=self.compute_all_black, nan_to_num=self.nan_to_num)

    #         class_names = self.dataset_meta.get('classes', None)
    #         if class_names is None:
    #             class_names = [str(i) for i in range(num_classes)]

    #         summary_str = 'Per-class segmentation results:\n'
    #         line_format = '{:<15} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n'
    #         summary_str += line_format.format(
    #             'Class', 'IoU', 'Dice', 'Fscore', 'PPV', 'S', 'AUPR', 'AUC', 'MAE')
    #         for i in range(num_classes):
    #             if not valid_mask[i]:
    #                 continue
    #             summary_str += line_format.format(
    #                 class_names[i],
    #                 f'{ret_metrics["IoU"][i]*100:.3f}', f'{ret_metrics["dice"][i]*100:.3f}',
    #                 f'{ret_metrics["Fscore"][i]*100:.3f}',
    #                 f'{ret_metrics["sp"][i]*100:.3f}', f'{ret_metrics["se"][i]*100:.3f}',
    #                 f'{ret_metrics["AUPR"][i]*100:.3f}', f'{ret_metrics["auc_roc"][i]*100:.3f}',
    #                 f'{ret_metrics["mae"][i]*100:.4f}')
    #         print_log(summary_str, logger)

    #         for key, val in ret_metrics.items():
    #             val = np.where(valid_mask, val, np.nan)
    #             metrics[f'seg_m{key}'] = float(np.nanmean(val)) * 100

    #     if cls_preds and cls_gts:
    #         cls_preds = np.stack(cls_preds, axis=0)
    #         cls_gts = np.stack(cls_gts, axis=0)
    #         cls_metrics, valid_classes = compute_classification_metrics(
    #             cls_preds, cls_gts)

    #         cls_names = self.dataset_meta.get('cls_classes', None)
    #         if cls_names is None:
    #             cls_names = [str(i) for i in range(cls_preds.shape[1])]

    #         print_log('Per-class classification results:', logger)
    #         for i, m in zip(valid_classes, cls_metrics):
    #             print_log(
    #                 f'Class {cls_names[i]}: Acc={m["Acc"]:.3f}, '
    #                 f'F1={m["F1"]:.3f}, Prec={m["Prec"]:.3f}, '
    #                 f'Rec={m["Rec"]:.3f}, Kappa={m["kappa"]:.3f}, '
    #                 f'QWK={m["weighted_kappa"]:.3f}',
    #                 logger)

    #         if cls_metrics:
    #             metrics['cls_acc'] = float(np.mean([m['Acc'] for m in cls_metrics])) * 100
    #             metrics['cls_f1'] = float(np.mean([m['F1'] for m in cls_metrics])) * 100
    #             metrics['cls_prec'] = float(np.mean([m['Prec'] for m in cls_metrics])) * 100
    #             metrics['cls_rec'] = float(np.mean([m['Rec'] for m in cls_metrics])) * 100
    #             metrics['cls_kappa'] = float(np.mean([m['kappa'] for m in cls_metrics]))
    #             metrics['cls_qwk'] = float(np.mean([m['weighted_kappa'] for m in cls_metrics]))

        

            # overall_kappa, overall_qwk = _compute_overall_kappa(
            #     cls_preds, cls_gts, self.cls_mode)
            # if overall_kappa is not None:
            #     metrics['cls_kappa_overall'] = float(overall_kappa)
            #     metrics['cls_qwk_overall'] = float(overall_qwk)
            #     print_log(
            #         f'Overall classification: Kappa={overall_kappa:.3f}, '
            #         f'QWK={overall_qwk:.3f}',
            #         logger)

        
