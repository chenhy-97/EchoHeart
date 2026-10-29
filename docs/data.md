# Data format

Organize the input data as two CSV files and a directory of videos:

```text
data/input/
    study_labels.csv
    videos.csv
    videos/
        study_001_a4c.npy
        study_001_plax.avi
        study_002_a4c.npy
        study_003_a4c.npy
        study_004_a4c.npy
```

## Study labels

One row represents one study. Required columns are `subject_id`, `study_id`, and `split`. Label columns use the English task names in [task_schema.json](../task_schema.json).

Example `study_labels.csv`:

```csv
subject_id,study_id,split,lvef,mitral_regurgitation
subject_001,study_001,train,62,0
subject_002,study_002,train,43,1
subject_003,study_003,val,57,0
subject_004,study_004,test,,1
```

| Field | Format |
| --- | --- |
| `subject_id` | Subject identifier; all studies from one subject share the same split |
| `study_id` | Unique study identifier |
| `split` | `train`, `val`, or `test` |
| Regression labels, such as `lvef` | Nonnegative values in native measurement units |
| Classification labels, such as `mitral_regurgitation` | `0` or `1` |
| Missing labels | Empty cells or `-1`; omitted task columns are filled with `-1` |

The model output order is defined by `task_schema.json`: 19 regression tasks followed by 29 classification tasks.

## Video manifest

One row represents one video. Multiple rows can share a `study_id` to form one examination.

Example `videos.csv`:

```csv
study_id,filepath
study_001,videos/study_001_a4c.npy
study_001,videos/study_001_plax.avi
study_002,videos/study_002_a4c.npy
study_003,videos/study_003_a4c.npy
study_004,videos/study_004_a4c.npy
```

`study_id` matches a row in `study_labels.csv`. `filepath` is an absolute path or a path relative to `videos.csv`. Each study has at least one video, and each video belongs to one study.

## Video arrays

Supported inputs are NPY arrays, AVI/MP4 and other OpenCV-supported videos, and multiframe DICOM. An RGB NPY video has shape `[frames, height, width, 3]` and dtype `uint8`, with values from `0` to `255`. For example, a 60-frame, 256-pixel-square clip has shape `[60, 256, 256, 3]`.

The loader samples 16 frames per clip, resizes each frame to `256 x 256`, and scales pixel values to `[0, 1]` without mean/std normalization.
