# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import os.path as osp
from pathlib import Path
from typing import Dict, Optional, Union

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile

from mmseg.registry import TRANSFORMS
from mmseg.utils import datafrombytes

try:
    from osgeo import gdal
except ImportError:
    gdal = None


@TRANSFORMS.register_module()
class LoadAnnotations(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadMultiSegClsAnnotations(BaseTransform):
    """Load multi-class binary masks and classification labels.

    The segmentation masks are stored per-class under ``seg_map_root`` with
    filenames matching ``seg_map_path``.

    Required Keys:

    - img
    - seg_map_root (optional)
    - seg_map_path (optional)
    - cls_label (optional)

    Added Keys:

    - gt_seg_map (np.ndarray): HWC binary masks with ignore value 255.
    - gt_label (np.ndarray): classification label vector.
    - seg_fields (list)
    """

    def __init__(self,
                 classes,
                 num_cls,
                 reduce_zero_label=False,
                 imdecode_backend='pillow',
                 backend_args=None,
                 test=False) -> None:
        self.classes = list(classes)
        self.num_cls = num_cls
        self.reduce_zero_label = reduce_zero_label
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args
        self.test=test

    def _get_hw(self, results: dict) -> tuple:
        if self.test:
            return results['ori_shape'][:2]
        if 'img_shape' in results:
            return results['img_shape'][:2]
        if 'img' in results:
            return results['img'].shape[:2]
        raise ValueError('`img_shape` or `img` must be set before loading annotations.')

    def transform(self, results: dict) -> dict:
        if 'seg_fields' not in results:
            results['seg_fields'] = []

        seg_map_name = results.get('seg_map_path', None)
        seg_root = results.get('seg_map_root', None)
        if seg_root is None and seg_map_name is not None:
            seg_root = osp.dirname(seg_map_name)
            seg_map_name = osp.basename(seg_map_name)

        height, width = self._get_hw(results)
        masks = []
        has_seg = False
        for cls_name in self.classes:
            mask = None
            if seg_root is not None and seg_map_name is not None:
                mask_path = osp.join(seg_root, cls_name, seg_map_name)
                if fileio.exists(mask_path, backend_args=self.backend_args):
                    has_seg = True
                    img_bytes = fileio.get(mask_path, backend_args=self.backend_args)
                    mask = mmcv.imfrombytes(
                        img_bytes, flag='grayscale',
                        backend=self.imdecode_backend).astype(np.uint8)
                    mask = (mask > 127).astype(np.uint8)

            if mask is None:
                mask = np.full((height, width), 255, dtype=np.uint8)

            if self.reduce_zero_label:
                mask[mask == 0] = 255
                mask = mask - 1
                mask[mask == 254] = 255

            # print(f'{cls_name}: {np.unique(mask)}')

            masks.append(mask)


        if masks:
            results['gt_seg_map'] = np.stack(masks, axis=-1)    
            results['seg_fields'].append('gt_seg_map')  #告诉后续所有 transform：gt_seg_map 是“分割标签”，需要跟图像一起做变换（如 resize / flip）
        results['has_seg'] = has_seg

        cls_label = results.get('cls_label', None)
        if cls_label is None:
            cls_label = np.full((self.num_cls,), -1, dtype=np.float32)
        cls_label = np.asarray(cls_label, dtype=np.float32)
        results['gt_label'] = cls_label
        results['has_cls'] = bool(np.any(cls_label != -1))

        return results

@TRANSFORMS.register_module()
class LoadImageFromNDArray(LoadImageFromFile):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadSingleBinaryCXRAnnotation(BaseTransform):
    """Load one binary CXR mask as a standard semantic segmentation map.

    This is intended for plain binary segmentation baselines on the CXR data
    layout where masks are stored as:
    ``{seg_map_root}/{class_name}/{filename}``.

    The output label map uses ``0`` for background, ``1`` for the selected
    class, and ``255`` for ignored/missing labels.
    """

    def __init__(self,
                 class_name='Pneumothorax',
                 missing_as_background=False,
                 imdecode_backend='pillow',
                 backend_args=None,
                 test=False) -> None:
        self.class_name = class_name
        self.missing_as_background = missing_as_background
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args
        self.test = test

    def _get_hw(self, results: dict) -> tuple:
        if self.test and 'ori_shape' in results:
            return results['ori_shape'][:2]
        if 'img_shape' in results:
            return results['img_shape'][:2]
        if 'img' in results:
            return results['img'].shape[:2]
        raise ValueError('`img_shape` or `img` must be set before loading annotations.')

    def transform(self, results: dict) -> dict:
        if 'seg_fields' not in results:
            results['seg_fields'] = []

        seg_map_name = results.get('seg_map_path', None)
        seg_root = results.get('seg_map_root', None)
        if seg_root is None and seg_map_name is not None:
            seg_root = osp.dirname(seg_map_name)
            seg_map_name = osp.basename(seg_map_name)

        mask = None
        if seg_root is not None and seg_map_name is not None:
            mask_path = osp.join(seg_root, self.class_name, seg_map_name)
            if fileio.exists(mask_path, backend_args=self.backend_args):
                img_bytes = fileio.get(mask_path, backend_args=self.backend_args)
                mask = mmcv.imfrombytes(
                    img_bytes,
                    flag='grayscale',
                    backend=self.imdecode_backend).astype(np.uint8)
                mask = (mask > 127).astype(np.uint8)

        if mask is None:
            height, width = self._get_hw(results)
            fill_value = 0 if self.missing_as_background else 255
            mask = np.full((height, width), fill_value, dtype=np.uint8)

        results['gt_seg_map'] = mask.astype(np.uint8)
        results['seg_fields'].append('gt_seg_map')
        return results


@TRANSFORMS.register_module()
class LoadBiomedicalImageFromFile(BaseTransform):
    """Load an biomedical mage from file.

    Required Keys:

    - img_path

    Added Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities, and data type is float32
        if set to_float32 = True, or float64 if decode_backend is 'nifti' and
        to_float32 is False.
    - img_shape
    - ori_shape

    Args:
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an float64 array.
            Defaults to True.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 decode_backend: str = 'nifti',
                 to_xyz: bool = False,
                 to_float32: bool = True,
                 backend_args: Optional[dict] = None) -> None:
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']

        data_bytes = fileio.get(filename, self.backend_args)
        img = datafrombytes(data_bytes, backend=self.decode_backend)

        if self.to_float32:
            img = img.astype(np.float32)

        if len(img.shape) == 3:
            img = img[None, ...]

        if self.decode_backend == 'nifti':
            img = img.transpose(0, 3, 2, 1)

        if self.to_xyz:
            img = img.transpose(0, 3, 2, 1)

        results['img'] = img
        results['img_shape'] = img.shape[1:]
        results['ori_shape'] = img.shape[1:]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadBiomedicalAnnotation(BaseTransform):
    """Load ``seg_map`` annotation provided by biomedical dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'gt_seg_map': np.ndarray (X, Y, Z) or (Z, Y, X)
        }

    Required Keys:

    - seg_map_path

    Added Keys:

    - gt_seg_map (np.ndarray): Biomedical seg map with shape (Z, Y, X) by
        default, and data type is float32 if set to_float32 = True, or
        float64 if decode_backend is 'nifti' and to_float32 is False.

    Args:
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        to_float32 (bool): Whether to convert the loaded seg map to a float32
            numpy array. If set to False, the loaded image is an float64 array.
            Defaults to True.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See :class:`mmengine.fileio` for details.
            Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 decode_backend: str = 'nifti',
                 to_xyz: bool = False,
                 to_float32: bool = True,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__()
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        data_bytes = fileio.get(results['seg_map_path'], self.backend_args)
        gt_seg_map = datafrombytes(data_bytes, backend=self.decode_backend)

        if self.to_float32:
            gt_seg_map = gt_seg_map.astype(np.float32)

        if self.decode_backend == 'nifti':
            gt_seg_map = gt_seg_map.transpose(2, 1, 0)

        if self.to_xyz:
            gt_seg_map = gt_seg_map.transpose(2, 1, 0)

        results['gt_seg_map'] = gt_seg_map
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadBiomedicalData(BaseTransform):
    """Load an biomedical image and annotation from file.

    The loading data format is as the following:

    .. code-block:: python

        {
            'img': np.ndarray data[:-1, X, Y, Z]
            'seg_map': np.ndarray data[-1, X, Y, Z]
        }


    Required Keys:

    - img_path

    Added Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.
    - img_shape
    - ori_shape

    Args:
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 with_seg=False,
                 decode_backend: str = 'numpy',
                 to_xyz: bool = False,
                 backend_args: Optional[dict] = None) -> None:  # noqa
        self.with_seg = with_seg
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        data_bytes = fileio.get(results['img_path'], self.backend_args)
        data = datafrombytes(data_bytes, backend=self.decode_backend)
        # img is 4D data (N, X, Y, Z), N is the number of protocol
        img = data[:-1, :]

        if self.decode_backend == 'nifti':
            img = img.transpose(0, 3, 2, 1)

        if self.to_xyz:
            img = img.transpose(0, 3, 2, 1)

        results['img'] = img
        results['img_shape'] = img.shape[1:]
        results['ori_shape'] = img.shape[1:]

        if self.with_seg:
            gt_seg_map = data[-1, :]
            if self.decode_backend == 'nifti':
                gt_seg_map = gt_seg_map.transpose(2, 1, 0)

            if self.to_xyz:
                gt_seg_map = gt_seg_map.transpose(2, 1, 0)
            results['gt_seg_map'] = gt_seg_map
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'with_seg={self.with_seg}, '
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class InferencerLoader(BaseTransform):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFile', **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadImageFromNDArray', **kwargs))

    def transform(self, single_input: Union[str, np.ndarray, dict]) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if isinstance(single_input, str):
            inputs = dict(img_path=single_input)
        elif isinstance(single_input, np.ndarray):
            inputs = dict(img=single_input)
        elif isinstance(single_input, dict):
            inputs = single_input
        else:
            raise NotImplementedError

        if 'img' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)


@TRANSFORMS.register_module()
class LoadSingleRSImageFromFile(BaseTransform):
    """Load a Remote Sensing mage from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is a float64 array.
            Defaults to True.
    """

    def __init__(self, to_float32: bool = True):
        self.to_float32 = to_float32

        if gdal is None:
            raise RuntimeError('gdal is not installed')

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        ds = gdal.Open(filename)
        if ds is None:
            raise Exception(f'Unable to open file: {filename}')
        img = np.einsum('ijk->jki', ds.ReadAsArray())

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str


@TRANSFORMS.register_module()
class LoadMultipleRSImageFromFile(BaseTransform):
    """Load two Remote Sensing mage from file.

    Required Keys:

    - img_path
    - img_path2

    Modified Keys:

    - img
    - img2
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is a float64 array.
            Defaults to True.
    """

    def __init__(self, to_float32: bool = True):
        if gdal is None:
            raise RuntimeError('gdal is not installed')
        self.to_float32 = to_float32

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        filename2 = results['img_path2']

        ds = gdal.Open(filename)
        ds2 = gdal.Open(filename2)

        if ds is None:
            raise Exception(f'Unable to open file: {filename}')
        if ds2 is None:
            raise Exception(f'Unable to open file: {filename2}')

        img = np.einsum('ijk->jki', ds.ReadAsArray())
        img2 = np.einsum('ijk->jki', ds2.ReadAsArray())

        if self.to_float32:
            img = img.astype(np.float32)
            img2 = img2.astype(np.float32)

        if img.shape != img2.shape:
            raise Exception(f'Image shapes do not match:'
                            f' {img.shape} vs {img2.shape}')

        results['img'] = img
        results['img2'] = img2
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str


@TRANSFORMS.register_module()
class LoadDepthAnnotation(BaseTransform):
    """Load ``depth_map`` annotation provided by depth estimation dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'gt_depth_map': np.ndarray [Y, X]
        }

    Required Keys:

    - seg_depth_path

    Added Keys:

    - gt_depth_map (np.ndarray): Depth map with shape (Y, X) by
        default, and data type is float32 if set to_float32 = True.
    - depth_rescale_factor (float): The rescale factor of depth map, which
        can be used to recover the original value of depth map.

    Args:
        decode_backend (str): The data decoding backend type. Options are
            'numpy', 'nifti', and 'cv2'. Defaults to 'cv2'.
        to_float32 (bool): Whether to convert the loaded depth map to a float32
            numpy array. If set to False, the loaded image is an uint16 array.
            Defaults to True.
        depth_rescale_factor (float): Factor to rescale the depth value to
            limit the range. Defaults to 1.0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See :class:`mmengine.fileio` for details.
            Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 decode_backend: str = 'cv2',
                 to_float32: bool = True,
                 depth_rescale_factor: float = 1.0,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__()
        self.decode_backend = decode_backend
        self.to_float32 = to_float32
        self.depth_rescale_factor = depth_rescale_factor
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load depth map.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded depth map.
        """
        data_bytes = fileio.get(results['depth_map_path'], self.backend_args)
        gt_depth_map = datafrombytes(data_bytes, backend=self.decode_backend)

        if self.to_float32:
            gt_depth_map = gt_depth_map.astype(np.float32)

        gt_depth_map *= self.depth_rescale_factor
        results['gt_depth_map'] = gt_depth_map
        results['seg_fields'].append('gt_depth_map')
        results['depth_rescale_factor'] = self.depth_rescale_factor
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadImageFromNpyFile(LoadImageFromFile):
    """Load an image from ``results['img_path']``.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']

        try:
            if Path(filename).suffix in ['.npy', '.npz']:
                img = np.load(filename)
            else:
                if self.file_client_args is not None:
                    file_client = fileio.FileClient.infer_client(
                        self.file_client_args, filename)
                    img_bytes = file_client.get(filename)
                else:
                    img_bytes = fileio.get(
                        filename, backend_args=self.backend_args)
                img = mmcv.imfrombytes(
                    img_bytes,
                    flag=self.color_type,
                    backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results
