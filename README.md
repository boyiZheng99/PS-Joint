# PS-Joint

Runnable MMSegmentation project for the final retinal PS-Joint variant. It
contains the MMSegmentation framework and the PS-Joint implementation, but no
baseline experiments, ablation variants, checkpoints, generated results, or
private datasets. The original research workspace is not modified.

## Project layout

- `mmseg/`: runnable MMSegmentation framework with PS-Joint already included.
- `configs/ps_joint/ps_joint_retinal.py`: one fully resolved PS-Joint config.
- `tools/`: training, testing, and distributed launch entry points.
- `scripts/`: short local and Slurm launchers.

The project does not require an overlay installation step. The configuration
is flat and has no `_base_` inheritance.

## Paper-to-code naming

The method is named **PS-Joint** in the paper. Its Python registry class is
`PSJoint` because Python identifiers cannot contain a hyphen:

```text
PS-Joint
├── segmentor: PSJoint                  (models/segmentors/ps_joint.py)
└── decoder:   PSJointDecodeHead        (models/decode_heads/ps_joint_decode_head.py)
```

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

For LUMI, edit `YOUR_PROJECT` in `scripts/slurm_train.sh`, export the paths above,
and submit the script with `sbatch`.

The release definition can be checked without constructing the full model:

```bash
python scripts/verify_release.py
```

## Final method represented here

- Swin-T spatial-prior expert and DINOv3-S adapter expert.
- Prompt-tuned CLIP disease queries.
- Direct key reparameterization with Gaussian noise clipped to ±2.35.
- Lightweight 8× patch embedding for probabilistic query interaction.
- Query-variance Mahalanobis maps computed in FP32 with variance floor `1e-4`.
- Patch-space Mahalanobis fusion and dual-expert pseudo-label selection.
- Deterministic mean evaluation (`eval_strategy='mean'`).
