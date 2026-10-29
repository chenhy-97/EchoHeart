"""Image-folder inputs with optional integer segmentation maps and mask policies."""

import json
import os
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from echoheart.data.datasets.extended import ExtendedVisionDataset


class StructureImageFolder(ExtendedVisionDataset):
    """Load images and masks using matching paths under their respective roots.

    ``transforms`` and ``transform`` both receive a sample dictionary containing
    ``image``, ``seg`` and ``uid``. Segmentation files retain their integer IDs,
    including the indices of palette PNGs. Missing masks produce ``seg=None``
    unless ``require_seg=True``. Targets remain empty for self-supervision unless
    ``target_transform`` is supplied.

    A policy JSON maps image-relative POSIX paths such as ``view/frame.png`` to
    ``{"mask_region": "boundary", "class_ids": [1, 2]}``. Either policy field
    can be omitted to use the configured default. ``ignore_ids``,
    ``min_coverage`` and ``boundary_width`` use the same selector options.
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seg_root: Optional[str] = None,
        seg_exts: Tuple[str, ...] = (".png", ".npy", ".jpg", ".jpeg", ".tif", ".tiff"),
        image_mode: str = "RGB",
        require_seg: bool = False,
        mask_policy_path: Optional[str] = None,
    ):
        if transforms is not None and transform is not None:
            raise ValueError("Provide transforms or transform, not both")
        # torchvision wraps transform in a two-argument StandardTransform;
        # this dataset's established augmentation API takes one sample dict.
        super().__init__(root=root)
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform
        self.root = os.fspath(root)
        self.seg_root = None if seg_root is None else os.fspath(seg_root)
        self.seg_exts = seg_exts
        self.image_mode = image_mode
        self.require_seg = require_seg
        if require_seg and self.seg_root is None:
            raise ValueError("require_seg=True requires seg_root")

        self.image_paths: List[str] = []
        for class_id in sorted(os.listdir(self.root)):
            class_dir = os.path.join(self.root, class_id)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                path = os.path.join(class_dir, fname)
                if os.path.isfile(path) and fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    self.image_paths.append(path)
        self.mask_policies = self._load_mask_policies(mask_policy_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_image_data(self, index: int) -> bytes:
        path = self.image_paths[index]
        try:
            with open(path, "rb") as stream:
                return stream.read()
        except OSError as exc:
            raise RuntimeError(f"Cannot retrieve image for sample {index}: {path}") from exc

    def get_target(self, index: int):
        return 0

    def _relative_path(self, image_path: str) -> str:
        return Path(os.path.relpath(image_path, self.root)).as_posix()

    def _load_mask_policies(self, path: Optional[str]) -> dict:
        if path is None:
            return {}
        from echoheart.data.structure_masking import StructureAwareMaskingGenerator

        with open(path, encoding="utf-8") as stream:
            policies = json.load(stream)
        if not isinstance(policies, dict):
            raise ValueError("Mask policy JSON must map image-relative paths to policy objects")
        known_paths = {self._relative_path(image_path) for image_path in self.image_paths}
        for image_path, policy in policies.items():
            relative = PurePosixPath(image_path)
            if relative.is_absolute() or ".." in relative.parts or "\\" in image_path or relative.as_posix() != image_path:
                raise ValueError(f"Mask policy path must be an image-relative POSIX path: {image_path}")
            if image_path not in known_paths:
                raise ValueError(f"Mask policy refers to an unknown image: {image_path}")
            if not isinstance(policy, dict) or set(policy) - StructureAwareMaskingGenerator._POLICY_KEYS:
                raise ValueError(f"Unsupported mask policy fields for {image_path}")
            defaults = dict(mask_region="foreground", class_ids=None, ignore_ids=[0], min_coverage=0.0, boundary_width=1)
            StructureAwareMaskingGenerator._validate_policy({**defaults, **policy})
        return policies

    def _infer_seg_path(self, img_path: str) -> Optional[str]:
        if self.seg_root is None:
            return None
        relative_stem = os.path.splitext(os.path.relpath(img_path, self.root))[0]
        for suffix in ("", "_mask"):
            for extension in self.seg_exts:
                candidate = os.path.join(self.seg_root, relative_stem + suffix + extension)
                if os.path.isfile(candidate):
                    return candidate
        return None

    def _load_seg(self, seg_path: Optional[str]) -> Optional[np.ndarray]:
        if seg_path is None:
            return None
        if seg_path.lower().endswith(".npy"):
            seg = np.load(seg_path, allow_pickle=False)
        else:
            with Image.open(seg_path) as image:
                seg = np.array(image)
        if seg.ndim != 2 or seg.dtype.kind not in "biu":
            raise ValueError(f"Segmentation must be a HxW integer class-ID map: {seg_path}")
        if not seg.size:
            raise ValueError(f"Segmentation is empty: {seg_path}")
        if seg.dtype.kind == "u" and seg.max() > np.iinfo(np.int64).max:
            raise ValueError(f"Segmentation class IDs must fit in int64: {seg_path}")
        return np.asarray(seg, dtype=np.int64)

    def __getitem__(self, index: int):
        with Image.open(BytesIO(self.get_image_data(index))) as source:
            image = source.convert(self.image_mode)
        image_path = self.image_paths[index]
        seg_path = self._infer_seg_path(image_path)
        if seg_path is None and self.require_seg:
            raise FileNotFoundError(f"Missing segmentation for image: {image_path}")
        seg = self._load_seg(seg_path)
        if seg is not None and seg.shape != (image.height, image.width):
            raise ValueError(
                f"Segmentation shape {seg.shape} differs from image shape "
                f"{(image.height, image.width)}: {image_path}"
            )

        sample = {"image": image, "seg": seg, "uid": image_path}
        policy = self.mask_policies.get(self._relative_path(image_path))
        if policy is not None:
            sample["mask_policy"] = {
                key: list(value) if isinstance(value, list) else value
                for key, value in policy.items()
            }
        transform = self.transforms if self.transforms is not None else self.transform
        output = transform(sample) if transform is not None else sample
        target = self.target_transform(self.get_target(index)) if self.target_transform is not None else ()
        return output, target
