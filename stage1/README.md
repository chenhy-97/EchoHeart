# EchoHeart Stage 1

Segmentation-guided self-supervised pretraining for echocardiography images, built on [DINOv3](https://github.com/facebookresearch/dinov3).

## Installation

Create the shared environment from the repository root, then enter `stage1`:

```bash
conda env create -f conda.yaml
conda activate structure_dinov3
cd stage1
```

## Data format

Place images and their segmentation maps in matching subdirectories:

Segmentation maps contain integer class IDs (`0` = background). Patches overlapping the selected foreground are masked. `mask_guidance.class_ids` selects labels; `null` uses all foreground labels. If fewer foreground patches are available than requested, only those patches are masked.

```text
data/
├── images/
│   └── view_a/
│       ├── frame001.png
│       └── frame002.png
└── masks/
    └── view_a/
        ├── frame001.png
        └── frame002.npy
```

## Training

The training entry point is [train.py](echoheart/train/train.py). The supplied [configuration](echoheart/configs/train/structure_aware_vits_pretrain.yaml) uses a ViT-S/16 backbone.

```bash
bash train.sh \
  train.image_root=./data/images \
  train.seg_root=./data/masks \
  train.batch_size_per_gpu=16
```

The launcher forwards additional configuration overrides. Select the number of GPUs and output directory through environment variables:

```bash
NPROC_PER_NODE=4 OUTPUT_DIR=./checkpoints/echoheart_stage1 \
  bash train.sh \
    train.image_root=./data/images \
    train.seg_root=./data/masks \
    train.batch_size_per_gpu=16
```

`CONFIG_FILE` selects a different YAML configuration. Training settings are merged with [ssl_default_config.yaml](echoheart/configs/ssl_default_config.yaml).

## Acknowledgments and license

This implementation builds on [DINOv3](https://github.com/facebookresearch/dinov3) by Meta AI. The DINOv3 license is retained in [LICENSE.md](LICENSE.md), together with the attribution in the source files.
