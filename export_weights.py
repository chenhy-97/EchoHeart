"""Export a Stage 2 or Stage 3 training checkpoint to release weight files.

Writes ``stage1.pth`` (backbone), ``stage2.pth`` (video model without the
backbone) and, for Stage 3, ``stage3.pth`` (study fusion). Each file is a
plain state dictionary that loads with ``torch.load(..., weights_only=True)``.
"""

import argparse
from pathlib import Path

import torch

from stage2.modeling import split_video_state


def export(checkpoint, output):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("stage") not in (2, 3) or "video_model" not in payload:
        raise ValueError("Expected a Stage 2 or Stage 3 training checkpoint")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stage1, stage2 = split_video_state(payload["video_model"])
    files = {"stage1.pth": stage1, "stage2.pth": stage2}
    if payload["stage"] == 3:
        files["stage3.pth"] = payload["fusion_model"]
    for name, state in files.items():
        torch.save({key: value.contiguous() for key, value in state.items()}, output / name)
        print(f"Wrote {output / name} ({len(state)} tensors)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
