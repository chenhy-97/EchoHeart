# EchoHeart

Official implementation of **A Hierarchical Foundation Model for Comprehensive and Interpretable Transthoracic Echocardiography Interpretation**.

EchoHeart is trained in three stages:

1. **Stage 1** pretrains a ViT-S/16 image encoder on echocardiography frames with segmentation-guided DINOv3 self-supervision.
2. **Stage 2** builds a video model on top of the Stage 1 encoder (spatial then temporal attention pooling, a class-attention decoder, and 48 task heads).
3. **Stage 3** fuses all videos of an examination into study-level predictions with task-specific attention.

## Installation

```bash
conda env create -f conda.yaml
conda activate structure_dinov3
```

For JPEG- or JPEG2000-compressed DICOM inputs, install the additional decoders:

```bash
python -m pip install -r requirements-dicom.txt
```



## Data

Stage 1 uses images with paired segmentation maps; see [Stage 1](stage1/README.md).

Stages 2 and 3 use study labels and a video manifest described in [docs/data.md](docs/data.md). Validate them and compute the regression loss scales from the training split:

```bash
python prepare_data.py \
  --labels data/input/study_labels.csv \
  --videos data/input/videos.csv \
  --output data/prepared
```

Tasks with fewer than 50 labeled training subjects (regression) or fewer than 10 positive and 10 negative subjects (classification) are disabled; see `--min-reg-subjects` and `--min-class-subjects`.

## Train



### Stage 1: foreground-masked pretraining

Use the ViT-S/16 configuration and paired images/masks described in [Stage 1](stage1/README.md):

```bash
bash stage1/train.sh
```



### Stage 2: video model

`--stage1-weights` accepts `stage1.pth` or a Stage 1 `teacher_checkpoint.pth`:

```bash
python train_stage2.py \
  --data data/prepared \
  --stage1-weights weights/stage1.pth \
  --output outputs/stage2 \
  --epochs 100 --device cuda
```



### Stage 3: study model

```bash
python train_stage3.py \
  --data data/prepared \
  --stage2-checkpoint outputs/stage2/best.pt \
  --output outputs/stage3 \
  --epochs 100 --device cuda
```

## Evaluate

Evaluate the final Stage 3 model on the test split, either from a training checkpoint:

```bash
python tests/test_study.py \
  --data data/prepared \
  --checkpoint outputs/stage3/best.pt \
  --output outputs/study_test \
  --device cuda
```

or from released weights:

```bash
python tests/test_study.py \
  --data data/prepared \
  --stage1-weights weights/stage1.pth \
  --stage2-weights weights/stage2.pth \
  --stage3-weights weights/stage3.pth \
  --output outputs/study_test \
  --device cuda
```

## Acknowledgments

EchoHeart builds on [DINOv3](https://github.com/facebookresearch/dinov3) by Meta AI. The DINOv3 code in `stage1/` and `stage2/dinov3/` is distributed under the [DINOv3 License](stage2/dinov3/LICENSE).