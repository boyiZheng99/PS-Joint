# Release scope

The release was extracted without changing the original workspace.

Included PS-Joint-specific components:

- the complete MMSegmentation runtime package and packaging metadata required
  to install and run this repository directly;
- final dual-expert segmentor and pseudo-label logic;
- Swin/DINOv3 adapter path and prompt-tuned CLIP text encoder;
- final key-reparameterized probabilistic interaction;
- FP32 query-variance patch-space Mahalanobis segmentation fusion;
- partial-label dataset, losses, transforms and metrics;
- the exact customized training and dataset-selectable testing entry points;
- one resolved retinal config under `configs/ps_joint/`, with no inheritance;
- the complete upstream DINOv3 project, excluding only `.git` metadata and
  generated Python cache files.

Excluded content:

- alternative PS-Joint variants and ablations;
- unrelated baseline/SOTA implementations;
- CXR-only configurations and implementations;
- checkpoints, pretrained weights, datasets, logs, tables and visualizations;
- generated experiment submission directories.
