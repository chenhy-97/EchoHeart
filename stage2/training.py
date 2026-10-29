"""Study-level training and evaluation for Stage 2 and Stage 3."""

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
from pydicom.errors import InvalidDicomError
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data import iter_studies, load_studies, read_video
from .metrics import compute_metrics
from .modeling import build_video_model, load_video_weights, read_weights


TASK_CONFIG_KEYS = (
    "regression_tasks", "classification_tasks", "tasks", "active_tasks",
    "regression_loss_scale_native_units",
)
TRAIN_CONFIG_KEYS = (
    "train_videos", "eval_video_cap", "video_batch_size", "lr_stage1",
    "lr_stage2", "lr_stage3", "aux_weight", "attention_temperature",
    "precision", "seed", "train_stage1", "freeze_stage2",
    "early_stop_patience",
)


class EmptyStudyError(RuntimeError):
    """The study has no readable videos."""


def _append_record(path, record):
    if path is not None:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class StudyModel(nn.Module):
    """Video network and optional task-specific patient fusion."""

    def __init__(self, stage, attention_temperature=0.2):
        super().__init__()
        self.stage = stage
        self.video = build_video_model(device="cpu")
        self.fusion = None
        if stage == 3:
            from stage3.fusion_models import PatientFusion

            self.fusion = PatientFusion(
                token_dim=384, hidden_dim=256,
                num_reg_tasks=19, num_cls_tasks=29,
                attention_temperature=attention_temperature,
            )


def task_config(cfg):
    """Retain the task schema and regression loss scales."""
    return {key: cfg[key] for key in TASK_CONFIG_KEYS}


def configure(model, cfg, train_stage1=False, freeze_stage2=False):
    active = set(cfg["active_tasks"])
    reg = [name in active for name in cfg["regression_tasks"]]
    cls = [name in active for name in cfg["classification_tasks"]]
    for parameter in model.video.parameters():
        parameter.requires_grad_(not freeze_stage2)
    for parameter in model.video.image_encoder.dinov2.parameters():
        parameter.requires_grad_(train_stage1)
    for heads, mask in (
        (model.video.multi_task_head.reg_heads, reg),
        (model.video.multi_task_head.cls_heads, cls),
    ):
        for head, enabled in zip(heads, mask):
            for parameter in head.parameters():
                parameter.requires_grad_(enabled and not freeze_stage2)

    packed = {}
    for parameter, mask in (
        (model.video.decoder.cls_reg, reg),
        (model.video.decoder.cls_cls, cls),
    ):
        if parameter.requires_grad:
            packed[parameter] = torch.tensor(mask, device=parameter.device)
    if model.fusion is not None:
        for parameter in model.fusion.parameters():
            parameter.requires_grad_(True)
            packed[parameter] = torch.tensor(reg + cls, device=parameter.device)
    for parameter, mask in packed.items():
        shape = [-1] + [1] * (parameter.ndim - 1)
        parameter.register_hook(lambda grad, mask=mask.reshape(shape): grad * mask)
    return packed


def make_optimizer(model, packed, args):
    backbone = model.video.image_encoder.dinov2
    backbone_ids = {id(parameter) for parameter in backbone.parameters()}
    groups = []
    stage2_parameters = [
        parameter for parameter in model.video.parameters()
        if parameter.requires_grad and parameter not in packed
        and id(parameter) not in backbone_ids
    ]
    if stage2_parameters:
        groups.append({"params": stage2_parameters, "lr": args.lr_stage2,
                       "weight_decay": 0.01})
    queries = [parameter for parameter in
               (model.video.decoder.cls_reg, model.video.decoder.cls_cls)
               if parameter.requires_grad]
    if queries:
        groups.append({"params": queries, "lr": args.lr_stage2, "weight_decay": 0.0})
    if model.fusion is not None:
        groups.append({"params": list(model.fusion.parameters()),
                       "lr": args.lr_stage3, "weight_decay": 0.0})
    if args.train_stage1:
        groups.append({"params": list(backbone.parameters()),
                       "lr": args.lr_stage1, "weight_decay": 0.01})
    return torch.optim.AdamW(groups)


def encode(model, videos, precision="float32"):
    """Encode a [B, 16, 3, 256, 256] batch into task tokens and predictions."""
    enabled = precision == "bfloat16"
    with torch.autocast(device_type=videos.device.type, dtype=torch.bfloat16,
                        enabled=enabled):
        outputs, _, reg_tokens, cls_tokens = model.video(videos)
        reg_pred = torch.stack(
            [value.reshape(-1) for value in outputs["regression"]], dim=1
        ).float()
        cls_prob = torch.stack(outputs["classification"], dim=1).float().softmax(-1)[..., 1]
    return reg_tokens.half().float(), cls_tokens.half().float(), reg_pred, cls_prob


def forward_study(model, paths, training=False, precision="float32", video_batch_size=1,
                  return_counts=False, error_log=None, bad_videos=None):
    device = next(model.parameters()).device
    chunks = []
    if bad_videos is None:
        bad_videos = set()
    readable = 0
    video_trainable = any(parameter.requires_grad for parameter in model.video.parameters())
    for offset in range(0, len(paths), video_batch_size):
        frames = []
        for path in paths[offset:offset + video_batch_size]:
            if str(path) in bad_videos:
                continue
            try:
                frames.append(read_video(path))
            except (OSError, ValueError, RuntimeError, KeyError, EOFError, TypeError,
                    InvalidDicomError) as error:
                bad_videos.add(str(path))
                record = {"filepath": str(path), "error_type": type(error).__name__,
                          "error": str(error)}
                _append_record(error_log, record)
                print("UNREADABLE_VIDEO " + json.dumps(record, ensure_ascii=False), flush=True)
        if not frames:
            continue
        readable += len(frames)
        videos = torch.stack(frames)
        videos = videos.to(device)
        if training and video_trainable:
            output = checkpoint(lambda value: encode(model, value, precision), videos,
                                use_reentrant=False)
        else:
            with torch.no_grad():
                output = encode(model, videos, precision)
        chunks.append(output)
    if not chunks:
        raise EmptyStudyError("No readable videos in study")
    reg_tokens, cls_tokens, reg_pred, cls_prob = [
        torch.cat([chunk[index] for chunk in chunks], dim=0).unsqueeze(0)
        for index in range(4)
    ]
    mean_video = {
        "regression": reg_pred.mean(1),
        "classification_logits": torch.logit(cls_prob.mean(1).clamp(1e-6, 1 - 1e-6)),
    }
    if model.fusion is None:
        result = mean_video, mean_video
        return (*result, readable) if return_counts else result
    mask = torch.ones(1, reg_pred.shape[1], device=device, dtype=torch.bool)
    fused = model.fusion(reg_tokens, cls_tokens, mask, reg_pred, cls_prob)
    result = fused, mean_video
    return (*result, readable) if return_counts else result


def losses(output, labels, cfg):
    """Mean loss within each task type, summed across regression and classification."""
    active = set(cfg["active_tasks"])
    terms = []
    device = output["regression"].device
    for kind, names in (("regression", cfg["regression_tasks"]),
                        ("classification", cfg["classification_tasks"])):
        task_losses = []
        for index, name in enumerate(names):
            value = float(labels[name])
            if name not in active or not np.isfinite(value) or value < 0:
                continue
            target = torch.tensor(value, device=device)
            if kind == "regression":
                scale = float(cfg["regression_loss_scale_native_units"][name])
                term = F.smooth_l1_loss(output["regression"][0, index] * 100 / scale,
                                       target / scale, beta=1.0)
            else:
                term = F.binary_cross_entropy_with_logits(
                    output["classification_logits"][0, index], target)
            task_losses.append(term)
        if task_losses:
            terms.append(torch.stack(task_losses).mean())
    if not terms:
        raise ValueError("Study contains no eligible labels")
    return torch.stack(terms).sum()


def _json_write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate(model, studies, videos, cfg, output, name="validation", cap=0,
             precision="float32", video_batch_size=1, bad_videos=None):
    model.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if bad_videos is None:
        bad_videos = set()
    scores, records, predictions, probabilities, targets, classes = [], [], [], [], [], []
    mean_predictions, mean_probabilities = [], []
    with torch.no_grad():
        for index, (row, paths) in enumerate(iter_studies(studies, videos, cap=cap)):
            try:
                predicted, mean_video, n_readable = forward_study(
                    model, paths, precision=precision, video_batch_size=video_batch_size,
                    return_counts=True, error_log=output / "unreadable_videos.jsonl",
                    bad_videos=bad_videos)
            except EmptyStudyError:
                _append_record(output / "excluded_studies.jsonl",
                               {"phase": name, "study_id": str(row.study_id),
                                "reason": "no_readable_videos"})
                continue
            if any(np.isfinite(row[task]) and row[task] >= 0 for task in cfg["active_tasks"]):
                score = float(losses(predicted, row, cfg))
                if not np.isfinite(score):
                    raise RuntimeError("Nonfinite validation loss")
                scores.append(score)
            reg = (predicted["regression"][0] * 100).cpu().numpy()
            cls = predicted["classification_logits"][0].sigmoid().cpu().numpy()
            mean_reg = (mean_video["regression"][0] * 100).cpu().numpy()
            mean_cls = mean_video["classification_logits"][0].sigmoid().cpu().numpy()
            predictions.append(reg)
            probabilities.append(cls)
            mean_predictions.append(mean_reg)
            mean_probabilities.append(mean_cls)
            targets.append([row[task] for task in cfg["regression_tasks"]])
            classes.append([row[task] for task in cfg["classification_tasks"]])
            attention = predicted.get("attention")
            entropy = None
            if attention is not None and n_readable > 1:
                weights = attention[0]
                entropy = (-(weights * weights.clamp_min(1e-30).log()).sum(0)
                           / np.log(n_readable)).cpu().numpy()
            for task_index, task in enumerate(cfg["tasks"]):
                if task not in cfg["active_tasks"]:
                    continue
                is_regression = task_index < len(cfg["regression_tasks"])
                local_index = task_index if is_regression else task_index - len(cfg["regression_tasks"])
                records.append({
                    "study_id": row.study_id, "task": task, "label": float(row[task]),
                    "prediction": float(reg[local_index] if is_regression else cls[local_index]),
                    "stage2_mean": float(mean_reg[local_index] if is_regression else mean_cls[local_index]),
                    "n_videos": n_readable, "n_requested_videos": len(paths),
                    "entropy": float(entropy[task_index]) if entropy is not None else None,
                })
            print(f"EVAL {name} {index + 1}/{len(studies)} videos={n_readable}/{len(paths)}", flush=True)
    if not predictions or not scores:
        raise ValueError("Evaluation requires studies with at least one active label")
    metrics = compute_metrics(predictions, probabilities, targets, classes, cfg)
    metrics["loss"] = float(np.mean(scores))
    metrics["studies"] = len(predictions)
    metrics["labeled_studies"] = len(scores)
    if model.fusion is not None:
        metrics["stage2_mean"] = compute_metrics(mean_predictions, mean_probabilities,
                                                 targets, classes, cfg)
    _json_write(output / f"{name}_metrics.json", metrics)
    pd.DataFrame(records).to_csv(output / f"{name}_predictions.csv", index=False)
    return metrics


def _read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _check_initialization_tasks(payload, cfg):
    schema = payload.get("task_config")
    if isinstance(schema, Mapping) and any(
        schema[key] != cfg[key] for key in ("regression_tasks", "classification_tasks", "tasks")
    ):
        raise ValueError("Initialization task order differs from the training data")


def _load_fusion(model, payload):
    """Load a training checkpoint (``fusion_model``) or a released ``stage3.pth``."""
    payload = read_weights(payload)
    state = payload["fusion_model"] if "fusion_model" in payload else payload
    model.fusion.load_state_dict(state, strict=True)


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    return {"torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(payload):
    torch.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])


def _validate_runtime(args):
    if args.cpu_threads < 1 or args.video_batch_size < 1 or args.eval_video_cap < 0:
        raise ValueError("cpu-threads/video-batch-size must be positive; eval-video-cap must be nonnegative")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else
                              torch.cuda.current_device())
    if device.type == "cpu" and args.precision != "float32":
        raise ValueError("CPU execution uses --precision float32")
    if args.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bfloat16 requires a GPU with BF16 support")
    return device


def _parser(stage, mode):
    parser = argparse.ArgumentParser(description=f"Stage {stage} {mode}")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["float32", "bfloat16"],
                        default="float32" if mode == "train" else None)
    parser.add_argument("--eval-video-cap", type=int, default=0)
    parser.add_argument("--video-batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--stage1-weights", type=Path)
    parser.add_argument("--stage2-weights", type=Path)
    if stage == 3:
        parser.add_argument("--stage3-weights", type=Path)
    if mode == "validate":
        parser.add_argument("--checkpoint", type=Path)
        parser.add_argument("--split", choices=["train", "val", "test"], default="val")
        parser.add_argument("--limit", type=int, default=0)
        parser.add_argument("--attention-temperature", type=float, default=0.2)
        return parser
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-videos", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--lr-stage1", type=float, default=1e-6)
    parser.add_argument("--lr-stage2", type=float, default=1e-5)
    parser.add_argument("--lr-stage3", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--attention-temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-stage1", action="store_true")
    parser.add_argument("--freeze-stage2", action="store_true")
    if stage == 3:
        parser.add_argument("--stage2-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--skip-initial-eval", action="store_true")
    return parser


def train(stage, args):
    device = _validate_runtime(args)
    if args.epochs < 1 or args.train_videos < 1 or args.early_stop_patience < 0 or args.save_every < 0:
        raise ValueError("epochs/train-videos must be positive; patience/save-every must be nonnegative")
    if args.attention_temperature <= 0 or args.aux_weight < 0:
        raise ValueError("attention-temperature must be positive and aux-weight nonnegative")
    if any(value <= 0 for value in (args.lr_stage1, args.lr_stage2, args.lr_stage3)):
        raise ValueError("Learning rates must be positive")
    if args.freeze_stage2 and (stage == 2 or args.train_stage1):
        raise ValueError("--freeze-stage2 requires Stage 3 with Stage 1 frozen")
    stage2_checkpoint = getattr(args, "stage2_checkpoint", None)
    stage3_weights = getattr(args, "stage3_weights", None)
    initializers = (args.stage1_weights, args.stage2_weights, stage2_checkpoint, stage3_weights)
    if args.resume and (any(initializers) or args.random_init):
        raise ValueError("Use --resume without initialization arguments")
    if stage2_checkpoint and args.stage2_weights:
        raise ValueError("Use either --stage2-checkpoint or --stage2-weights")
    cfg, train_studies, train_videos = load_studies(args.data, "train", args.train_limit,
                                                 seed=args.seed)
    _, val_studies, val_videos = load_studies(args.data, "val", args.val_limit,
                                            seed=args.seed)
    cfg = task_config(cfg)
    labeled = train_studies[cfg["active_tasks"]].ge(0).any(axis=1)
    train_studies = train_studies.loc[labeled].reset_index(drop=True)
    if train_studies.empty:
        raise ValueError("Training requires active labels")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.output.iterdir()):
        raise ValueError("Choose an empty training output directory")
    _set_seed(args.seed)
    model = StudyModel(stage, args.attention_temperature)
    resumed = _read_checkpoint(args.resume) if args.resume else None
    if resumed is not None:
        if resumed.get("stage") != stage:
            raise ValueError("Resume checkpoint has a different training stage")
        load_video_weights(model.video, checkpoint=resumed)
        if stage == 3:
            _load_fusion(model, resumed)
    else:
        source = None
        if stage2_checkpoint:
            source = _read_checkpoint(stage2_checkpoint)
            if source.get("stage") != 2:
                raise ValueError("--stage2-checkpoint expects a Stage 2 training checkpoint")
            _check_initialization_tasks(source, cfg)
        if not args.random_init and not (args.stage1_weights or source):
            raise ValueError("Supply --stage1-weights or --stage2-checkpoint")
        if stage == 3 and not (source or args.stage2_weights):
            raise ValueError("Stage 3 requires --stage2-checkpoint or --stage1-weights with --stage2-weights")
        load_video_weights(model.video, stage1_weights=args.stage1_weights,
                           stage2_weights=args.stage2_weights, checkpoint=source)
        if stage3_weights:
            _load_fusion(model, stage3_weights)
        del source
    model.to(device)
    packed = configure(model, cfg, args.train_stage1, args.freeze_stage2)
    optimizer = make_optimizer(model, packed, args)
    settings = {key: getattr(args, key) for key in TRAIN_CONFIG_KEYS}
    best, best_epoch, stale_epochs = float("inf"), 0, 0
    start_epoch, start_step, prior_loss = 0, 0, []
    history = []
    bad_videos = set()
    if resumed is not None:
        if resumed["task_config"] != cfg or resumed["config"] != settings:
            raise ValueError("Resume task configuration and training options must match the checkpoint")
        if resumed["train_studies"] != len(train_studies):
            raise ValueError("Resume training split size differs from the checkpoint")
        optimizer.load_state_dict(resumed["optimizer"])
        best, best_epoch = resumed["best_validation_loss"], resumed["best_epoch"]
        stale_epochs = resumed["stale_epochs"]
        history = resumed["history"]
        if resumed["phase"] == "validated":
            start_epoch = resumed["epoch"]
        else:
            start_epoch = resumed["epoch"] - 1
            start_step = resumed["completed_steps_in_epoch"]
            prior_loss = resumed["epoch_losses"]
        if start_epoch >= args.epochs:
            raise ValueError("--epochs must exceed the completed checkpoint epoch")
        if start_step > len(train_studies):
            raise ValueError("Checkpoint step exceeds the training split size")
        if np.isfinite(best) and args.resume.parent.resolve() != args.output.resolve():
            previous_best = args.resume.parent / "best.pt"
            if not previous_best.is_file():
                raise FileNotFoundError("Resume checkpoint directory requires best.pt")
            best_payload = _read_checkpoint(previous_best)
            if (best_payload.get("stage") != stage or best_payload.get("task_config") != cfg
                    or best_payload.get("epoch") != best_epoch
                    or best_payload.get("validation_loss") != best):
                raise ValueError("best.pt differs from the model selected in the resume checkpoint")
            del best_payload
            shutil.copyfile(previous_best, args.output / "best.pt")
        _restore_rng(resumed)
        del resumed
    _json_write(args.output / "config.json", {"stage": stage, "task_config": cfg, **settings})
    if args.resume is None and not args.skip_initial_eval:
        evaluate(model, val_studies, val_videos, cfg, args.output, name="initial",
                 cap=args.eval_video_cap, precision=args.precision,
                 video_batch_size=args.video_batch_size, bad_videos=bad_videos)

    def save(filename, epoch, completed_steps, phase, epoch_losses, score=None):
        payload = {
            "stage": stage,
            "video_model": {name: value.detach().cpu() for name, value in model.video.state_dict().items()},
            "task_config": cfg, "config": settings, "optimizer": optimizer.state_dict(),
            "epoch": epoch, "completed_steps_in_epoch": completed_steps,
            "phase": phase, "epoch_losses": epoch_losses, "validation_loss": score,
            "best_validation_loss": best, "best_epoch": best_epoch,
            "stale_epochs": stale_epochs, "history": history,
            "train_studies": len(train_studies), **_rng_state(),
        }
        if model.fusion is not None:
            payload["fusion_model"] = {
                name: value.detach().cpu() for name, value in model.fusion.state_dict().items()
            }
        destination = args.output / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.freeze_stage2:
            model.video.eval()
        model.video.image_encoder.dinov2.train(args.train_stage1)
        epoch_losses = list(prior_loss) if epoch == start_epoch else []
        ordered = train_studies.sample(frac=1, random_state=args.seed + epoch)
        items = iter_studies(ordered, train_videos, training=True, epoch=epoch,
                             seed=args.seed, cap=args.train_videos)
        for step, (row, paths) in enumerate(items):
            if epoch == start_epoch and step < start_step:
                continue
            optimizer.zero_grad(set_to_none=True)
            try:
                output, mean_video = forward_study(
                    model, paths, training=True, precision=args.precision,
                    video_batch_size=args.video_batch_size,
                    error_log=args.output / "unreadable_videos.jsonl", bad_videos=bad_videos)
            except EmptyStudyError:
                _append_record(args.output / "excluded_studies.jsonl",
                               {"phase": "train", "epoch": epoch + 1,
                                "study_id": str(row.study_id),
                                "reason": "no_readable_videos"})
                continue
            loss = losses(output, row, cfg)
            if stage == 3:
                loss = loss + args.aux_weight * losses(mean_video, row, cfg)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0,
                                                      error_if_nonfinite=True)
            optimizer.step()
            value = float(loss.detach())
            epoch_losses.append(value)
            print(f"TRAIN stage={stage} epoch={epoch + 1} step={step + 1}/{len(train_studies)} "
                  f"loss={value:.6f} grad_norm={float(grad_norm):.6f}", flush=True)
            if args.save_every and (step + 1) % args.save_every == 0:
                save("last.pt", epoch + 1, step + 1, "train", epoch_losses)
        if not epoch_losses:
            raise RuntimeError("No training steps completed")
        save("last.pt", epoch + 1, len(train_studies), "before_validation", epoch_losses)
        metrics = evaluate(model, val_studies, val_videos, cfg, args.output,
                           cap=args.eval_video_cap, precision=args.precision,
                           video_batch_size=args.video_batch_size, bad_videos=bad_videos)
        score = metrics["loss"]
        improved = score < best
        if improved:
            best, best_epoch, stale_epochs = score, epoch + 1, 0
        else:
            stale_epochs += 1
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)),
                        "validation_loss": score, "best_validation_loss": best})
        _json_write(args.output / "history.json", history)
        if improved:
            save("best.pt", epoch + 1, len(train_studies), "validated", epoch_losses, score)
        save("last.pt", epoch + 1, len(train_studies), "validated", epoch_losses, score)
        print(f"EPOCH {epoch + 1} train={np.mean(epoch_losses):.6f} "
              f"validation={score:.6f} best={best:.6f}", flush=True)
        if args.early_stop_patience and stale_epochs >= args.early_stop_patience:
            break


def validate(stage, args):
    """Evaluate a training checkpoint or the released stage weight files."""
    released = [args.stage1_weights, args.stage2_weights]
    if stage == 3:
        released.append(args.stage3_weights)
    if bool(args.checkpoint) == any(released) or (not args.checkpoint and not all(released)):
        names = "--stage1-weights, --stage2-weights" + (", --stage3-weights" if stage == 3 else "")
        raise ValueError(f"Supply either --checkpoint or all of {names}")
    payload = _read_checkpoint(args.checkpoint) if args.checkpoint else None
    if payload is not None and (payload.get("stage") != stage or "video_model" not in payload):
        raise ValueError(f"Expected a complete Stage {stage} training checkpoint")
    if args.precision is None:
        cpu = torch.device(args.device).type == "cpu"
        args.precision = "float32" if cpu or payload is None else payload["config"]["precision"]
    device = _validate_runtime(args)
    cfg, studies, videos = load_studies(args.data, args.split, args.limit)
    cfg = task_config(cfg)
    if payload is not None:
        if cfg != payload["task_config"]:
            raise ValueError("Evaluation task configuration differs from the checkpoint")
        temperature = payload["config"]["attention_temperature"]
    else:
        temperature = args.attention_temperature
    model = StudyModel(stage, temperature)
    if payload is not None:
        load_video_weights(model.video, checkpoint=payload)
        if stage == 3:
            _load_fusion(model, payload)
    else:
        load_video_weights(model.video, stage1_weights=args.stage1_weights,
                           stage2_weights=args.stage2_weights)
        if stage == 3:
            _load_fusion(model, args.stage3_weights)
    model.to(device)
    del payload
    metrics = evaluate(model, studies, videos, cfg, args.output,
                       cap=args.eval_video_cap, precision=args.precision,
                       video_batch_size=args.video_batch_size)
    print(json.dumps(metrics, ensure_ascii=False, allow_nan=False), flush=True)


def main(stage, mode):
    if stage not in (2, 3) or mode not in ("train", "validate"):
        raise ValueError("Expected stage=2/3 and mode=train/validate")
    args = _parser(stage, mode).parse_args()
    if mode == "train":
        train(stage, args)
    else:
        validate(stage, args)
