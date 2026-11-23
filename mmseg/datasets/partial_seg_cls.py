# Copyright (c) OpenMMLab. All rights reserved.
import csv
import os.path as osp
from typing import Dict, List, Optional, Sequence

import mmengine.fileio as fileio
import numpy as np

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PartialSegClsDataset(BaseSegDataset):
    """DDR dataset for joint lesion segmentation + DR grading classification.

    The segmentation labels are stored per-class under ``seg_map_path``:
    ``{seg_map_path}/{class_name}/{filename}{seg_map_suffix}``.
    Classification labels are read from a CSV file where the first column is
    the filename and the remaining columns are label values.
    """

    METAINFO = dict(
        classes=('EX','HE','MA','SE','RV','OC','OD','Atrophy','Detachment','Drusen','Scar'),
        # palette=[[255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 128, 255]])
    )


    def __init__(self,
                 cls_label_file: Optional[str] = None,
                 cls_classes: Optional[Sequence[str]] = None,
                 **kwargs) -> None:
        self.cls_label_file = cls_label_file
        self.cls_classes = list(cls_classes) if cls_classes is not None else None
        self._cls_label_header: Optional[List[str]] = None
        self._cls_label_map: Dict[str, np.ndarray] = {}
        super().__init__(**kwargs)

    def _load_cls_labels(self, path: str) -> Dict[str, np.ndarray]:
        if not path or not fileio.exists(path, backend_args=self.backend_args):
            return {}

        raw_label_map: Dict[str, np.ndarray] = {}
        with fileio.get_local_path(path, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and len(header) > 1:
                    self._cls_label_header = [h.strip() for h in header[1:]]  #表头
                for row in reader:
                    if not row:
                        continue
                    filename = row[0].strip()
                    labels = [float(x) for x in row[1:]]
                    raw_label_map[filename] = np.asarray(labels, dtype=np.float32)

        # Project legacy retinal labels to the requested class list.  The
        # requested list determines the task: five-grade configurations keep
        # the five grade columns, while binary configurations request ``DR``
        # (or legacy ``D``) and collapse the grades to {0, 1}.
        if self.cls_classes is None or self._cls_label_header is None:
            return raw_label_map

        header_index = {
            name: index for index, name in enumerate(self._cls_label_header)
        }
        dr_grade_names = (
            'No_DR', 'mild_DR', 'moderate_DR', 'severe_DR',
            'Proliferative_DR')

        def project_binary_dr(values: np.ndarray) -> float:
            """Project a legacy DR label source to one binary value."""
            if 'D' in header_index:
                d_value = values[header_index['D']]
                return -1.0 if d_value < 0 else float(d_value > 0)

            grade_indices = [
                header_index[name] for name in dr_grade_names
                if name in header_index]
            if not grade_indices:
                return -1.0

            grade_values = values[grade_indices]
            if not np.any(grade_values != -1):
                return -1.0

            # No_DR is the negative grade.  Any known positive grade means DR.
            positive_indices = [
                header_index[name] for name in dr_grade_names[1:]
                if name in header_index]
            return float(np.any(values[positive_indices] > 0))

        def project_one(values: np.ndarray) -> np.ndarray:
            projected = []
            for class_name in self.cls_classes:
                if class_name in header_index:
                    projected.append(values[header_index[class_name]])
                    continue

                if class_name in ('DR', 'D'):
                    projected.append(project_binary_dr(values))
                    continue

                # A class absent from a partial label file is unknown.
                projected.append(-1.0)
            return np.asarray(projected, dtype=np.float32)

        return {
            filename: project_one(values)
            for filename, values in raw_label_map.items()
        }

    def _resolve_img_name(self, name: str) -> str:
        return name if name.endswith(self.img_suffix) else name + self.img_suffix

    def _resolve_seg_name(self, img_name: str) -> str:
        if img_name.endswith(self.img_suffix):
            return img_name[:-len(self.img_suffix)] + self.seg_map_suffix
        return img_name + self.seg_map_suffix

    def load_data_list(self) -> List[dict]:
        data_list: List[dict] = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if img_dir is None:
            raise ValueError('`data_prefix.img_path` must be set for DDRSegClsDataset.')

        if self.cls_label_file and not self._cls_label_map:
            cls_label_file = self.cls_label_file
            if self.data_root is not None and not osp.isabs(cls_label_file):
                cls_label_file = osp.join(self.data_root, cls_label_file)
            self._cls_label_map = self._load_cls_labels(cls_label_file)

        if self.cls_classes is None and self._cls_label_header is not None:
            self.cls_classes = list(self._cls_label_header)

        if self.cls_classes is not None:
            self._metainfo.setdefault('cls_classes', self.cls_classes)

        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), f'Failed to load ann_file: {self.ann_file}'
            lines = fileio.list_from_file(self.ann_file, backend_args=self.backend_args)
            img_names = [self._resolve_img_name(line.strip()) for line in lines]
        else:
            img_names = fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args)
            img_names = sorted(img_names)

        for img_name in img_names:
            seg_name = self._resolve_seg_name(img_name)
            data_info = dict(
                img_path=osp.join(img_dir, img_name),
                seg_map_path=seg_name,
                seg_map_root=ann_dir,
                cls_label=self._cls_label_map.get(img_name),
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[])
            data_list.append(data_info)

        return data_list
