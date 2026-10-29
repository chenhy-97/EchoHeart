"""Task-wise metrics for predictions in native regression units."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def compute_metrics(reg_pred, cls_prob, reg_target, cls_target, cfg):
    """Evaluate active tasks; targets below zero are missing labels."""
    reg_pred, cls_prob, reg_target, cls_target = [
        np.asarray(values, dtype=np.float64)
        for values in (reg_pred, cls_prob, reg_target, cls_target)
    ]
    regression = cfg["regression_tasks"]
    classification = cfg["classification_tasks"]
    if (reg_pred.ndim != 2 or cls_prob.ndim != 2
            or reg_pred.shape != reg_target.shape or cls_prob.shape != cls_target.shape
            or reg_pred.shape[0] != cls_prob.shape[0]
            or reg_pred.shape[1] != len(regression)
            or cls_prob.shape[1] != len(classification)):
        raise ValueError("Expected matching [N, R] regression and [N, C] classification arrays")
    active = set(cfg["active_tasks"])
    per_task = {}
    aucs, mapes = [], []
    for index, task in enumerate(regression):
        valid = np.isfinite(reg_target[:, index]) & (reg_target[:, index] >= 0)
        if task not in active or not valid.any():
            continue
        target, pred = reg_target[valid, index], reg_pred[valid, index]
        if not np.isfinite(pred).all():
            raise ValueError(f"Predictions must be finite for {task}")
        error = pred - target
        nonzero = target != 0
        mape = _number(np.mean(np.abs(error[nonzero] / target[nonzero])) * 100) if nonzero.any() else None
        total = np.sum((target - target.mean()) ** 2)
        r2 = _number(1 - np.sum(error ** 2) / total) if len(target) > 1 and total > 0 else None
        per_task[task] = {
            "n": int(len(target)),
            "mae": _number(np.mean(np.abs(error))),
            "mape": mape,
            "rmse": _number(np.sqrt(np.mean(error ** 2))),
            "r2": r2,
        }
        if mape is not None:
            mapes.append(mape)
    for index, task in enumerate(classification):
        valid = np.isfinite(cls_target[:, index]) & (cls_target[:, index] >= 0)
        if task not in active or not valid.any():
            continue
        target, prob = cls_target[valid, index], cls_prob[valid, index]
        if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
            raise ValueError(f"Probabilities must be finite and between zero and one for {task}")
        if not np.isin(target, [0, 1]).all():
            raise ValueError(f"Classification targets must be zero or one for {task}")
        pred = prob >= 0.5
        auc = _number(roc_auc_score(target, prob)) if len(np.unique(target)) == 2 else None
        per_task[task] = {
            "n": int(len(target)),
            "auc": auc,
            "accuracy": _number(accuracy_score(target, pred)),
            "f1": _number(f1_score(target, pred, zero_division=0)),
            "precision": _number(precision_score(target, pred, zero_division=0)),
            "recall": _number(recall_score(target, pred, zero_division=0)),
            "brier": _number(np.mean((prob - target) ** 2)),
        }
        if auc is not None:
            aucs.append(auc)
    return {
        "per_task": per_task,
        "macro_auc": _number(np.mean(aucs)) if aucs else None,
        "macro_mape": _number(np.mean(mapes)) if mapes else None,
    }
