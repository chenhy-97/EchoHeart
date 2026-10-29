"""Study manifests, video sampling, and frame decoding."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from .video_io import read_video as read_video_frames


FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])


def load_studies(data_dir, split, limit=0, seed=None):
    """Load a split from config.json, study_labels.csv, and videos.csv."""
    data_dir = Path(data_dir).resolve()
    cfg = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    regression = cfg["regression_tasks"]
    classification = cfg["classification_tasks"]
    tasks = cfg["tasks"]
    if len(regression) != 19 or len(classification) != 29:
        raise ValueError("Expected 19 regression tasks and 29 classification tasks")
    if tasks != regression + classification or len(set(tasks)) != 48:
        raise ValueError("tasks must contain regression_tasks followed by classification_tasks")
    active = cfg["active_tasks"]
    if not active or len(set(active)) != len(active) or not set(active).issubset(tasks):
        raise ValueError("active_tasks must contain distinct names from tasks")
    scales = cfg["regression_loss_scale_native_units"]
    for task in regression:
        if task in active and (task not in scales or not np.isfinite(float(scales[task]))
                               or float(scales[task]) <= 0):
            raise ValueError(f"Expected a positive regression loss scale for {task}")

    studies = pd.read_csv(data_dir / "study_labels.csv",
                          dtype={"subject_id": str, "study_id": str, "split": str})
    required = ["subject_id", "study_id", "split"] + tasks
    missing = [name for name in required if name not in studies.columns]
    if missing:
        raise ValueError(f"study_labels.csv is missing columns: {missing}")
    if studies[["subject_id", "study_id", "split"]].isna().any().any():
        raise ValueError("subject_id, study_id, and split must be present for every study")
    if studies.study_id.duplicated().any():
        raise ValueError("study_labels.csv must contain one row per study_id")
    if not studies.split.isin(["train", "val", "test"]).all():
        raise ValueError("split must be train, val, or test")
    if studies.groupby("subject_id")["split"].nunique().gt(1).any():
        raise ValueError("A subject_id appears in more than one split")
    for task in tasks:
        values = pd.to_numeric(studies[task], errors="raise").fillna(-1).astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Labels must be finite for {task}")
        if ((values < 0) & (values != -1)).any():
            raise ValueError(f"Missing labels must be -1 for {task}")
        if task in classification and not values.isin([-1, 0, 1]).all():
            raise ValueError(f"Classification labels must be -1, 0, or 1 for {task}")
        studies[task] = values

    manifest = pd.read_csv(data_dir / "videos.csv",
                           dtype={"study_id": str, "filepath": str})
    if not {"study_id", "filepath"}.issubset(manifest.columns):
        raise ValueError("videos.csv requires study_id and filepath columns")
    if manifest[["study_id", "filepath"]].isna().any().any():
        raise ValueError("Every video requires study_id and filepath")
    videos = {}
    owners = {}
    for item in manifest.sort_values("filepath").itertuples(index=False):
        path = Path(item.filepath).expanduser()
        if not path.is_absolute():
            path = data_dir / path
        path = path.resolve()
        if path in owners and owners[path] != item.study_id:
            raise ValueError("A video filepath appears under more than one study_id")
        owners[path] = item.study_id
        videos.setdefault(item.study_id, []).append(str(path))
    if any(study_id not in videos for study_id in studies.study_id):
        raise ValueError("Every study in study_labels.csv requires at least one video")
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    selected = studies.loc[studies.split == split]
    if split in ("train", "val"):
        selected = selected.loc[(selected[active] >= 0).any(axis=1)]
    selected = selected.reset_index(drop=True)
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if limit:
        selected = (selected.iloc[:limit].copy() if seed is None else
                    selected.sample(n=min(limit, len(selected)), random_state=seed))
    if selected.empty:
        raise ValueError(f"No studies in split {split}")
    return cfg, selected, {study_id: videos[study_id] for study_id in selected.study_id}


def iter_studies(studies, videos, training=False, epoch=0, seed=42, cap=8):
    """Yield studies in manifest order with the original per-study sampling."""
    if cap < 0:
        raise ValueError("cap must be nonnegative")
    for _, row in studies.iterrows():
        paths = videos[row.study_id]
        if training and cap:
            try:
                study_seed = int(row.study_id)
            except (TypeError, ValueError):
                study_seed = sum((i + 1) * ord(c) for i, c in enumerate(str(row.study_id)))
            rng = np.random.default_rng(seed + epoch * 100000 + study_seed)
            indices = sorted(rng.choice(len(paths), min(cap, len(paths)), replace=False))
            paths = [paths[i] for i in indices]
        elif not training and cap and len(paths) > cap:
            paths = [paths[i] for i in np.linspace(0, len(paths) - 1, cap, dtype=int)]
        yield row, paths


def read_video(path):
    """Return 16 RGB frames as a [16, 3, 256, 256] float tensor."""
    path = str(path)
    if Path(path).suffix.lower() in {".dcm", ".dicom", ".ima", ""}:
        import pydicom

        ds = pydicom.dcmread(path, stop_before_pixels=True,
                            specific_tags=["NumberOfFrames", "PhotometricInterpretation"])
        n = int(getattr(ds, "NumberOfFrames", 1))
        if n <= 1:
            raise ValueError(f"Not a multiframe video: {path}")
        ids = np.linspace(0, n - 1, 16, dtype=int) if n >= 16 else np.arange(16) % n
        unique = sorted(set(ids.tolist()))
        frames = dict(zip(unique, pydicom.pixels.iter_pixels(path, indices=unique)))
        result = []
        for i in ids:
            x = frames[int(i)]
            if x.dtype != np.uint8:
                raise ValueError(f"Unsupported DICOM dtype {x.dtype}")
            if x.ndim == 2:
                if ds.PhotometricInterpretation == "MONOCHROME1":
                    x = 255 - x
                x = np.repeat(x[..., None], 3, -1)
            if x.shape[-1] != 3:
                raise ValueError("Expected RGB")
            result.append(FRAME_TRANSFORM(Image.fromarray(x)))
        return torch.stack(result)

    return torch.stack([FRAME_TRANSFORM(frame) for frame in read_video_frames(path, num_frames=16)])
