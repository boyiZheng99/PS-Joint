# PS-Joint

This repository is the official implementation of paper: PS-Joint: Partially Supervised Learning with Uncertainty for Joint
Medical Image Segmentation and Disease Classification.

## Project layout

- `mmseg/`: runnable MMSegmentation framework with PS-Joint already included.
- `configs/ps_joint/ps_joint_retinal.py`: one fully resolved PS-Joint config.
- `tools/`: training, testing, and distributed launch entry points.
- `scripts/`: short local and Slurm launchers.

## Install

Create an environment and install this repository directly:

```bash
pip install -r requirements.txt
pip install -v -e .
```

The bundled DINOv3 source retains its upstream license at
`mmseg/tool_libs/segdino/dinov3/LICENSE.md`.

## Data and pretrained weights

Prepare these files without committing them to GitHub:

```text
pretrained/
├── swin_tiny_patch4_window7_224.pth
├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
└── CLIP-ViT-B-16.pt
```

Set portable paths:

```bash
export PSJOINT_DATA_ROOT=/path/to/FOVCrop-padding
export PSJOINT_PRETRAINED_ROOT=/path/to/pretrained
export PSJOINT_DINO_REPO=$PWD/mmseg/tool_libs/segdino/dinov3
```

The dataset directory must follow the paths explicitly listed in the flat
config. Classification labels use partial multi-label CSV files; absent labels
are encoded as `-1`.

## Train and test

```bash
export GPUS=8
export WORK_DIR=$PWD/work_dirs/ps_joint_retinal
bash scripts/train.sh

bash scripts/test.sh work_dirs/ps_joint_retinal/iter_15000.pth \
  --test_dataset dataset_DDR_test --test_dataset_ty retinal
```
