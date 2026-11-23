# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    parser.add_argument(
        '--test_dataset',
        type=str,
        default=None,
        help='Name of test dataset variable in config')

    parser.add_argument(
        '--test_dataset_ty',
        type=str,
        choices=['retinal', 'cxr'],
        default='retinal',
        help='test dataset type')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def remove_none_data_prefix(dataset_cfg):
    """Remove optional path entries set to None before dataset construction.

    Classification-only retinal configs intentionally set ``seg_map_path``
    to ``None``.  MMEngine's BaseDataset requires every value in
    ``data_prefix`` to be a string, even when the segmentation branch is not
    used, so leaving that placeholder in the test config makes dataset
    construction fail before inference starts.
    """
    if isinstance(dataset_cfg, dict) and isinstance(
            dataset_cfg.get('data_prefix'), dict):
        dataset_cfg['data_prefix'] = {
            key: value
            for key, value in dataset_cfg['data_prefix'].items()
            if value is not None
        }
    return dataset_cfg


def main():
    args = parse_args()
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    if args.test_dataset is not None:
        if args.test_dataset not in cfg:
            raise ValueError(
                f'Test dataset "{args.test_dataset}" not found in config.'
            )
        cfg.test_dataloader.dataset = cfg[args.test_dataset]
        cfg.test_dataloader.dataset = remove_none_data_prefix(
            cfg.test_dataloader.dataset)

        if args.test_dataset in ['dataset_RFMiDv2_test', 'dataset_BREST_test']:    #这两个检测任务单独处理
            cfg.test_evaluator = [
                dict(
                    type='NoSegGTClsMetric',
                    target_classes=('Drusen', 'Scar', 'Detachment'),
                    area_threshold=0)
                ]
        else:
            if args.test_dataset_ty == 'retinal':
                # The decoupled retinal configs use a binary DR head.  The
                # legacy metric default treats the last output as five-class
                # grading and applies softmax to a one-column tensor, which
                # always predicts class 0.  Use sigmoid for the binary head.
                cls_classes = cfg.test_dataloader.dataset.get('cls_classes', ())
                binary_dr = bool(cls_classes and cls_classes[-1] == 'DR')
                cfg.test_evaluator = dict(
                    type='SegClsMetric',
                    compute_all_black=True,
                    if_include_multi_class=not binary_dr)
            else:
                cfg.test_evaluator = dict(type='SegClsMetric', compute_all_black=True, if_include_multi_class=False)


    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # Pass the requested output directory to custom metrics.  Some test
    # evaluators are lists (for example RFMiDv2/BREST), so do not index the
    # evaluator as a dict unconditionally.
    if args.out is not None:
        if isinstance(cfg.test_evaluator, (list, tuple)):
            evaluators = []
            for evaluator in cfg.test_evaluator:
                evaluator = dict(evaluator)
                evaluator['output_dir'] = args.out
                evaluator['keep_results'] = True
                evaluators.append(evaluator)
            cfg.test_evaluator = evaluators
        else:
            cfg.test_evaluator = dict(cfg.test_evaluator)
            cfg.test_evaluator['output_dir'] = args.out
            cfg.test_evaluator['keep_results'] = True

    

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
