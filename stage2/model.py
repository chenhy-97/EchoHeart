"""Video encoder: DINOv3 patch tokens, spatial then temporal attention pooling,
a class-attention decoder, and per-task heads."""

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from einops import rearrange


# ===================== Building blocks =====================

class AttnPoolBlock(nn.Module):
    """Pre-LN multi-head attention pooling: learns a query to aggregate tokens."""

    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        trunc_normal_(self.query, std=0.02)
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden),
            nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, tokens):
        B, N, D = tokens.shape
        q = self.query.expand(B, -1, -1)
        k = v = self.ln(tokens)
        pooled, _ = self.attn(q, k, v)
        pooled = pooled + q
        pooled = pooled + self.ffn(pooled)
        return pooled.squeeze(1)


class SpatialAggregator(nn.Module):
    """Partition each frame's HxW patches into blocks, pool each block via AttnPoolBlock."""

    def __init__(self, dim, num_heads=8, out_mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.block_pool = AttnPoolBlock(
            dim, num_heads=num_heads, mlp_ratio=out_mlp_ratio, dropout=dropout,
        )

    def forward(self, x, resolution_input: Tuple[int, int], resolution_reduced: Tuple[int, int]):
        H, W = resolution_input
        out_h, out_w = resolution_reduced
        Bf, N, C = x.shape
        assert N == H * W, f"Token count {N} != H*W ({H}*{W})"
        assert H % out_h == 0 and W % out_w == 0, "H/W must be divisible by out_h/out_w"

        patch_h, patch_w = H // out_h, W // out_w
        x_img = rearrange(x, "B (H W) C -> B H W C", H=H, W=W)
        blocks = rearrange(
            x_img, "B (oh ph) (ow pw) C -> (B oh ow) (ph pw) C",
            oh=out_h, ow=out_w, ph=patch_h, pw=patch_w,
        )
        pooled = self.block_pool(blocks)
        out = rearrange(pooled, "(B oh ow) C -> B (oh ow) C", B=Bf, oh=out_h, ow=out_w)
        return out


class SliceAggregator(nn.Module):
    """Per-spatial-position temporal pooling via AttnPoolBlock."""

    def __init__(self, dim=768, num_heads=8, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.pool = AttnPoolBlock(
            dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
        )

    def forward(self, x):
        B, T, N, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        fused = self.pool(x_flat)
        fused = fused.view(B, N, D)
        return fused


class ImageEncoders(nn.Module):
    def __init__(self, dinov2_model, spatial_res_input=(16, 16),
                 spatial_res_reduced=(8, 8), dim=384, num_heads=8, dropout=0.0):
        super().__init__()
        self.dinov2 = dinov2_model
        self.spatial_res_input = spatial_res_input
        self.spatial_res_reduced = spatial_res_reduced
        self.spatial_aggregator = SpatialAggregator(dim, num_heads=num_heads, dropout=dropout)
        self.slice_aggregator = SliceAggregator(dim, num_heads=num_heads, dropout=dropout)
        out_h, out_w = spatial_res_reduced
        self.pos_embed = nn.Parameter(torch.zeros(1, out_h * out_w, dim))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        batch, frames, channels, height, width = x.shape
        x_frames = x.view(batch * frames, channels, height, width)
        patch_tokens = self.dinov2.get_intermediate_layers(
            x_frames, n=1, reshape=False, return_class_token=False,
        )[0]
        spatial_out = self.spatial_aggregator(
            patch_tokens,
            resolution_input=self.spatial_res_input,
            resolution_reduced=self.spatial_res_reduced,
        )
        spatial_out = spatial_out.view(batch, frames, -1, patch_tokens.size(-1))
        return self.slice_aggregator(spatial_out), self.pos_embed


# ===================== CAiT-style decoder =====================

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


class ClassAttentionBlock(nn.Module):
    """CAiT-style Class-Attention block with LayerScale.

    - Pre-LN on queries and keys/values.
    - 2D position encoding injected into keys (tokens + pos).
    - Two residual paths, each gated by LayerScale.
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0, layerscale=1e-5):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden),
            nn.GELU(), nn.Linear(hidden, dim),
        )
        self.ls1 = LayerScale(dim, layerscale)
        self.ls2 = LayerScale(dim, layerscale)

    def forward(self, queries, tokens, pos=None):
        if pos is not None and pos.size(0) == 1 and tokens.size(0) > 1:
            pos = pos.expand(tokens.size(0), -1, -1)
        q = self.ln_q(queries)
        k = self.ln_kv(tokens if pos is None else tokens + pos)
        v = self.ln_kv(tokens)
        attn_out, _ = self.attn(q, k, v)
        x = queries + self.ls1(attn_out)
        x = x + self.ls2(self.ffn(x))
        return x


class CaiTDecoder(nn.Module):
    """CAiT decoder: learns per-task CLS tokens, cross-attends to fused tokens."""

    def __init__(
        self, dim, depth=3, num_heads=8,
        num_reg_heads=19, num_cls_heads=19,
        layerscale=1e-5, mlp_ratio=4.0, dropout=0.0,
    ):
        super().__init__()
        self.num_reg = num_reg_heads
        self.num_cls = num_cls_heads
        self.cls_reg = nn.Parameter(torch.randn(num_reg_heads, dim))
        self.cls_cls = nn.Parameter(torch.randn(num_cls_heads, dim))
        trunc_normal_(self.cls_reg, std=0.02)
        trunc_normal_(self.cls_cls, std=0.02)
        self.blocks = nn.ModuleList([
            ClassAttentionBlock(
                dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                dropout=dropout, layerscale=layerscale,
            )
            for _ in range(depth)
        ])

    def forward(self, fused_tokens, pos):
        B, N, D = fused_tokens.shape
        q_reg = self.cls_reg.unsqueeze(0).expand(B, -1, -1)
        q_cls = self.cls_cls.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            q_reg = blk(q_reg, fused_tokens, pos)
        for blk in self.blocks:
            q_cls = blk(q_cls, fused_tokens, pos)
        return q_reg, q_cls


# ===================== Multi-task heads =====================

class MultiTaskHeadTokenDriven(nn.Module):
    """Per-task MLP heads driven by decoded CLS tokens.

    Each regression head: dim -> dim//2 -> 1
    Each classification head: dim -> dim//2 -> n_classes (adaptive per task)
    """

    def __init__(
        self, dim, num_reg_heads=19, num_cls_heads=19,
        reg_outputs=1, cls_classes=None, dropout=0.3,
    ):
        super().__init__()
        self.num_reg_heads = num_reg_heads
        self.num_cls_heads = num_cls_heads
        self.reg_outputs = reg_outputs

        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(dim // 2, reg_outputs),
            ) for _ in range(num_reg_heads)
        ])

        if cls_classes is None:
            cls_classes = [2] * num_cls_heads
        elif isinstance(cls_classes, int):
            cls_classes = [cls_classes] * num_cls_heads
        elif len(cls_classes) != num_cls_heads:
            raise ValueError(
                f"cls_classes length {len(cls_classes)} != num_cls_heads {num_cls_heads}"
            )
        cls_classes = [int(n) for n in cls_classes]

        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(dim // 2, n_classes),
            ) for n_classes in cls_classes
        ])
        self.cls_classes = cls_classes

    def forward(self, q_reg, q_cls):
        regression_outputs = [
            self.reg_heads[i](q_reg[:, i, :]) for i in range(self.num_reg_heads)
        ]
        classification_outputs = [
            self.cls_heads[j](q_cls[:, j, :]) for j in range(self.num_cls_heads)
        ]
        return {"regression": regression_outputs, "classification": classification_outputs}


# ===================== Full model =====================

class MultiTaskEchocardiographyModel(nn.Module):
    """Full multi-task echocardiography video model.

    Pipeline:
      frames -> ImageEncoders (DINOv3 + spatial/temporal aggregation)
             -> CaiTDecoder (Class-Attention + LayerScale + pos)
             -> MultiTaskHeadTokenDriven (per-task predictions)

    Returns (outputs, aux_loss) by default, or (outputs, aux_loss, q_reg, q_cls)
    when return_tokens=True; Stage 3 fuses the q_reg/q_cls task tokens.
    """

    def __init__(
        self, image_encoder, head_input_dim=768, num_heads=8,
        decoder_depth=3, num_reg_heads=19, num_cls_heads=19,
        reg_outputs=1, cls_classes=2, dropout=0.0,
        layerscale=1e-5, mlp_ratio=4.0, return_tokens=False,
    ):
        super().__init__()
        self.return_tokens = return_tokens
        self.image_encoder = image_encoder
        self.decoder = CaiTDecoder(
            dim=head_input_dim, depth=decoder_depth, num_heads=num_heads,
            num_reg_heads=num_reg_heads, num_cls_heads=num_cls_heads,
            layerscale=layerscale, mlp_ratio=mlp_ratio, dropout=dropout,
        )
        self.multi_task_head = MultiTaskHeadTokenDriven(
            dim=head_input_dim, num_reg_heads=num_reg_heads,
            num_cls_heads=num_cls_heads, reg_outputs=reg_outputs,
            cls_classes=cls_classes,
        )

    def forward(self, x):
        fused_tokens, pos = self.image_encoder(x)
        q_reg, q_cls = self.decoder(fused_tokens, pos)
        outputs = self.multi_task_head(q_reg, q_cls)
        aux_loss = torch.zeros((), device=fused_tokens.device)
        if self.return_tokens:
            return outputs, aux_loss, q_reg, q_cls
        else:
            return outputs, aux_loss
