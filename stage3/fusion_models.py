"""Task-specific attention fusion of video predictions into study predictions."""

import math

import torch
from torch import nn


class TaskSpecificMLP(nn.Module):
    """An independent two-layer residual prediction head for each task."""

    def __init__(self, num_tasks, input_dim, hidden_dim):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(num_tasks, input_dim, hidden_dim))
        self.bias1 = nn.Parameter(torch.zeros(num_tasks, hidden_dim))
        self.weight2 = nn.Parameter(torch.empty(num_tasks, hidden_dim))
        self.bias2 = nn.Parameter(torch.zeros(num_tasks))
        nn.init.xavier_uniform_(self.weight1)
        nn.init.zeros_(self.weight2)

    def forward(self, tokens):
        hidden = torch.relu(
            torch.einsum("btd,tdh->bth", tokens, self.weight1)
            + self.bias1.unsqueeze(0)
        )
        return (
            torch.einsum("bth,th->bt", hidden, self.weight2)
            + self.bias2.unsqueeze(0)
        )


def masked_mean(values, video_mask):
    mask = video_mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class PatientFusion(nn.Module):
    """Aggregate task tokens and video predictions into study predictions.

    Token inputs have shape [batch, videos, tasks, token_dim]. Prediction
    inputs have shape [batch, videos, tasks], and video_mask is [batch, videos].
    Classification inputs are probabilities; outputs are study-level logits.
    Attention weights are returned as [batch, videos, all_tasks].
    """

    def __init__(self, token_dim=384, hidden_dim=256, num_reg_tasks=19,
                 num_cls_tasks=29, attention_temperature=0.2):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_reg_tasks = int(num_reg_tasks)
        self.num_cls_tasks = int(num_cls_tasks)
        self.num_tasks = self.num_reg_tasks + self.num_cls_tasks
        self.attention_temperature = float(attention_temperature)
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")

        self.heads = TaskSpecificMLP(self.num_tasks, 2 * self.token_dim, self.hidden_dim)
        self.prediction_weight = nn.Parameter(torch.empty(self.num_tasks, self.token_dim))
        self.prediction_bias = nn.Parameter(torch.zeros(self.num_tasks, self.token_dim))
        self.queries = nn.Parameter(torch.empty(self.num_tasks, 2 * self.token_dim))
        nn.init.trunc_normal_(self.prediction_weight, std=0.02)
        nn.init.normal_(self.queries, std=1.0)
        self.attention_norm = nn.LayerNorm(2 * self.token_dim, elementwise_affine=False)
        self.patient_norm = nn.LayerNorm(2 * self.token_dim, elementwise_affine=False)

    def forward(self, reg_tokens, cls_tokens, video_mask, reg_predictions, cls_probabilities):
        if video_mask.ndim != 2 or not video_mask.bool().any(dim=1).all():
            raise ValueError("Every study requires at least one valid video")
        tokens = torch.cat([reg_tokens, cls_tokens], dim=2).float()
        predictions = torch.cat([reg_predictions, cls_probabilities], dim=2).float()
        if tokens.shape[2] != self.num_tasks or predictions.shape[2] != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} tasks")
        mask = video_mask.bool()

        # Embed each prediction alongside its task-specific visual token.
        cls_logits = torch.logit(cls_probabilities.float().clamp(1e-6, 1.0 - 1e-6))
        attention_predictions = torch.cat([reg_predictions.float(), cls_logits], dim=2)
        prediction_tokens = (
            attention_predictions.unsqueeze(-1) * self.prediction_weight[None, None]
            + self.prediction_bias[None, None]
        )
        fused_tokens = torch.cat([tokens, prediction_tokens], dim=-1)

        # Normalize attention across the valid videos of each study.
        scores = torch.einsum("bmtd,td->bmt", self.attention_norm(fused_tokens), self.queries)
        scores = scores / math.sqrt(2 * self.token_dim)
        scores = scores / self.attention_temperature
        scores = scores.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        attention = scores.softmax(dim=1)
        patient_tokens = torch.einsum("bmt,bmtd->btd", attention, fused_tokens)
        outputs = self.heads(self.patient_norm(patient_tokens))

        # Learn a residual over the mean video prediction.
        averaged = masked_mean(predictions, mask)
        averaged_regression = averaged[:, :self.num_reg_tasks]
        averaged_probability = averaged[:, self.num_reg_tasks:].clamp(1e-6, 1.0 - 1e-6)
        outputs = outputs + torch.cat(
            [averaged_regression, torch.logit(averaged_probability)], dim=1
        )
        return {
            "regression": outputs[:, :self.num_reg_tasks],
            "classification_logits": outputs[:, self.num_reg_tasks:],
            "attention": attention,
        }
