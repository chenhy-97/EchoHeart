"""Prepare study manifests and training-label regression scales."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(labels_path, videos_path, output, schema_path=None,
            min_reg_subjects=50, min_class_subjects=10):
    schema_path = schema_path or Path(__file__).with_name("task_schema.json")
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    regression = schema["regression_tasks"]
    classification = schema["classification_tasks"]
    tasks = regression + classification
    if len(regression) != 19 or len(classification) != 29 or len(set(tasks)) != 48:
        raise ValueError("The schema must contain 19 regression and 29 classification tasks")
    labels = pd.read_csv(labels_path, dtype={"subject_id": str, "study_id": str, "split": str})
    identifiers = ["subject_id", "study_id", "split"]
    if not set(identifiers).issubset(labels.columns):
        raise ValueError("Labels require subject_id, study_id, and split")
    if labels[identifiers].isna().any().any() or labels.study_id.duplicated().any():
        raise ValueError("Identifiers must be present and study_id must be unique")
    if not labels.split.isin(["train", "val", "test"]).all():
        raise ValueError("split must be train, val, or test")
    if labels.groupby("subject_id").split.nunique().gt(1).any():
        raise ValueError("All studies from one subject must have the same split")
    for task in tasks:
        if task not in labels:
            labels[task] = -1.0
        labels[task] = pd.to_numeric(labels[task], errors="raise").fillna(-1)
        values = labels[task].to_numpy()
        if not np.isfinite(values).all() or ((values < 0) & (values != -1)).any():
            raise ValueError(f"Use nonnegative labels or -1 for {task}")
        if task in classification and not np.isin(values, [-1, 0, 1]).all():
            raise ValueError(f"Use binary labels for {task}")
    train = labels.split.eq("train")
    if not train.any() or not labels.split.eq("val").any():
        raise ValueError("Training and validation studies are required")
    active, scales, coverage = [], {}, []
    for task in tasks:
        observed = train & labels[task].ge(0)
        subject_count = int(labels.loc[observed, "subject_id"].nunique())
        positive = int(labels.loc[train & labels[task].eq(1), "subject_id"].nunique())
        negative = int(labels.loc[train & labels[task].eq(0), "subject_id"].nunique())
        enabled = (positive >= min_class_subjects and negative >= min_class_subjects
                   if task in classification else subject_count >= min_reg_subjects)
        if enabled:
            active.append(task)
            if task in regression:
                scales[task] = max(float(labels.loc[observed, task].std(ddof=0)), 1.0)
        coverage.append({"task": task, "enabled": enabled, "train_subjects": subject_count})
    if not active:
        raise ValueError("No tasks meet the training-label coverage thresholds")
    videos_path = Path(videos_path).resolve()
    videos = pd.read_csv(videos_path, dtype={"study_id": str, "filepath": str})
    if not {"study_id", "filepath"}.issubset(videos.columns):
        raise ValueError("Video manifest requires study_id and filepath")
    if videos[["study_id", "filepath"]].isna().any().any():
        raise ValueError("Each video requires a study_id and filepath")
    resolved = []
    for value in videos.filepath:
        path = Path(value).expanduser()
        path = (videos_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")
        resolved.append(str(path))
    videos["filepath"] = resolved
    if videos.groupby("filepath").study_id.nunique().gt(1).any():
        raise ValueError("A video belongs to more than one study")
    videos = videos[["study_id", "filepath"]].drop_duplicates().sort_values("filepath")
    if not set(labels.study_id).issubset(videos.study_id):
        raise ValueError("Each labeled study requires at least one video")
    if not set(videos.study_id).issubset(labels.study_id):
        raise ValueError("Every video study_id requires a label row")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    labels[identifiers + tasks].to_csv(output / "study_labels.csv", index=False)
    videos.to_csv(output / "videos.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "task_coverage.csv", index=False)
    cfg = {"regression_tasks": regression, "classification_tasks": classification,
           "tasks": tasks, "active_tasks": active,
           "regression_loss_scale_native_units": scales}
    (output / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Prepared {len(labels)} studies, {len(videos)} videos, {len(active)} active tasks")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--min-reg-subjects", type=int, default=50)
    parser.add_argument("--min-class-subjects", type=int, default=10)
    args = parser.parse_args()
    if min(args.min_reg_subjects, args.min_class_subjects) < 1:
        parser.error("Coverage thresholds must be positive")
    prepare(args.labels, args.videos, args.output, args.schema,
            args.min_reg_subjects, args.min_class_subjects)


if __name__ == "__main__":
    main()
