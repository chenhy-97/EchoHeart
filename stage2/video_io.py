"""Read uniformly sampled frames from a video file."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def read_video(path, num_frames=16):
    path = str(Path(path))
    if path.lower().endswith(".npy"):
        array = np.load(path, mmap_mode="r")
        if array.ndim == 2:
            array = array[None, :, :, None]
        elif array.ndim == 3:
            if array.shape[-1] in (1, 3, 4):
                array = array[None, ...]
            else:
                array = array[..., None]
        if array.ndim != 4:
            raise RuntimeError(f"Unsupported NPY video shape {array.shape}: {path}")
        total = int(array.shape[0])
        if total <= 0:
            raise RuntimeError(f"Video is empty: {path}")
        indices = (np.arange(num_frames) % total if total < num_frames else
                   np.linspace(0, total - 1, num_frames, dtype=int))
        frames = []
        for index in indices:
            frame = np.asarray(array[int(index)])
            if frame.dtype != np.uint8:
                frame = frame.astype(np.float32, copy=False)
                lo, hi = float(frame.min()), float(frame.max())
                if hi > lo:
                    frame = ((frame - lo) / (hi - lo) * 255.0).clip(0, 255)
                else:
                    frame = np.zeros_like(frame)
                frame = frame.astype(np.uint8)
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[..., :3]
            frames.append(Image.fromarray(frame))
        return frames

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        frames = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        capture.release()
        if not frames:
            raise RuntimeError(f"Video is empty: {path}")
        if len(frames) < num_frames:
            return [frames[i % len(frames)] for i in range(num_frames)]
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        return [frames[i] for i in indices]

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = capture.read()
        if success:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        raise RuntimeError(f"No valid frames read from: {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames
