"""Video model construction and checkpoint loading."""

from collections.abc import Mapping
from pathlib import Path
import sys

import torch


_RUNTIME_DIRECTORY = str(Path(__file__).resolve().parent)
if _RUNTIME_DIRECTORY not in sys.path:
    sys.path.insert(0, _RUNTIME_DIRECTORY)

from dinov3.models.vision_transformer import DinoVisionTransformer
from .model import ImageEncoders, MultiTaskEchocardiographyModel


BACKBONE_PREFIX = "image_encoder.dinov2."


def build_backbone():
    """ViT-S/16 with the Stage 1 pretraining configuration."""
    backbone = DinoVisionTransformer(
        img_size=224, patch_size=16, in_chans=3,
        pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32",
        embed_dim=384, depth=12, num_heads=6, ffn_ratio=4,
        qkv_bias=True, drop_path_rate=0.0, layerscale_init=1e-5,
        norm_layer="layernormbf16", ffn_layer="mlp", ffn_bias=True,
        proj_bias=True, n_storage_tokens=4, mask_k_bias=True,
    )
    backbone.init_weights()
    return backbone


def build_video_model(device="cpu"):
    """Construct the 19-regression, 29-classification video model."""
    encoder = ImageEncoders(
        build_backbone(), spatial_res_input=(16, 16), spatial_res_reduced=(8, 8),
        dim=384, num_heads=8, dropout=0.0,
    )
    model = MultiTaskEchocardiographyModel(
        encoder, head_input_dim=384, num_heads=8, decoder_depth=3,
        num_reg_heads=19, num_cls_heads=29, reg_outputs=1,
        cls_classes=[2] * 29, dropout=0.0, layerscale=1e-5,
        mlp_ratio=4.0, return_tokens=True,
    )
    return model.to(device)


def read_weights(source):
    if isinstance(source, Mapping):
        return source
    return torch.load(source, map_location="cpu", weights_only=False)


def backbone_state(state):
    """Accept a backbone state dict or a Stage 1 ``teacher_checkpoint.pth``."""
    if "teacher" in state:
        state = state["teacher"]
    prefixed = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
    return prefixed or dict(state)


def load_stage1(model, state):
    model.image_encoder.dinov2.load_state_dict(backbone_state(state), strict=True)


def load_stage2(model, state):
    """Load every video-model weight except the backbone (``stage2.pth``)."""
    expected = {key for key in model.state_dict() if not key.startswith(BACKBONE_PREFIX)}
    actual = set(state)
    if actual != expected:
        raise RuntimeError(
            f"Stage 2 state keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    model.load_state_dict({**model.state_dict(), **state}, strict=True)


def load_video_weights(model, stage1_weights=None, stage2_weights=None, checkpoint=None):
    """Load released ``stage1.pth``/``stage2.pth`` files or a training checkpoint.

    Inputs may be paths or loaded mappings. A training checkpoint stores the
    complete video model under ``video_model``.
    """
    if stage1_weights is not None:
        load_stage1(model, read_weights(stage1_weights))
    if stage2_weights is not None:
        load_stage2(model, read_weights(stage2_weights))
    if checkpoint is not None:
        model.load_state_dict(read_weights(checkpoint)["video_model"], strict=True)


def split_video_state(state):
    """Split a complete video-model state dict into stage1 and stage2 parts."""
    stage1 = {k[len(BACKBONE_PREFIX):]: v for k, v in state.items() if k.startswith(BACKBONE_PREFIX)}
    stage2 = {k: v for k, v in state.items() if not k.startswith(BACKBONE_PREFIX)}
    return stage1, stage2
