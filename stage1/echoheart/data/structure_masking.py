"""Select DINOv3 pretraining masks inside segmented regions.

The returned boolean array has the original MaskingGenerator convention: True
means replace that patch with the learned mask token. Segmentation labels must
already have the same crop, resize and flip as the corresponding input image.
"""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import torch


class StructureAwareMaskingGenerator:
    """Restrict a random block mask to selected anatomical regions.

    ``foreground`` selects the union of ``class_ids`` (all labels except
    ``ignore_ids`` when class_ids is None). ``boundary`` selects selected-class
    pixels adjacent to a different label in the four-neighbourhood, including
    boundaries between two selected classes. ``boundary_width=1`` selects those
    pixels; larger values expand the band within the selected foreground.
    Width is measured in pixels of the augmented segmentation image.

    A patch is eligible when region coverage is positive and >= min_coverage.
    The masking count is capped by the number of eligible patches. Empty
    foreground produces no masked patches. Missing segmentation and
    mask_region='none' use the original random mask without extra random draws.
    """

    _POLICY_KEYS = frozenset({"mask_region", "class_ids", "ignore_ids", "min_coverage", "boundary_width"})

    def __init__(
        self,
        base_generator,
        patch_size=16,
        mask_region="foreground",
        class_ids=None,
        ignore_ids=(0,),
        min_coverage=0.0,
        boundary_width=1,
    ):
        self.base_generator = base_generator
        self.height, self.width = base_generator.get_shape()
        self.num_patches = self.height * self.width
        if not isinstance(patch_size, Integral) or isinstance(patch_size, bool) or patch_size < 1:
            raise ValueError("patch_size must be a positive integer")
        self.patch_size = int(patch_size)
        self.policy = self._validate_policy({
            "mask_region": mask_region,
            "class_ids": class_ids,
            "ignore_ids": ignore_ids,
            "min_coverage": min_coverage,
            "boundary_width": boundary_width,
        })

    def get_shape(self):
        return self.height, self.width

    @staticmethod
    def _validate_policy(policy):
        if policy["mask_region"] not in {"foreground", "boundary", "none"}:
            raise ValueError("mask_region must be foreground, boundary, or none")
        for name in ("class_ids", "ignore_ids"):
            values = policy[name]
            if values is None and name == "class_ids":
                continue
            if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
                raise ValueError(f"{name} must be a sequence of integer labels")
            values = tuple(values)
            if any(not isinstance(value, Integral) or isinstance(value, bool) for value in values):
                raise ValueError(f"{name} must contain integer labels")
            policy[name] = values
        coverage = policy["min_coverage"]
        if not isinstance(coverage, Real) or not np.isfinite(coverage) or not 0 <= coverage <= 1:
            raise ValueError("min_coverage must be between 0 and 1")
        width = policy["boundary_width"]
        if not isinstance(width, Integral) or isinstance(width, bool) or width < 1:
            raise ValueError("boundary_width must be a positive integer in image pixels")
        return policy

    def _resolve_policy(self, mask_policy):
        policy = dict(self.policy)
        if mask_policy is not None:
            if not isinstance(mask_policy, Mapping):
                raise ValueError("mask_policy must be a dictionary of selection options")
            unknown = set(mask_policy) - self._POLICY_KEYS
            if unknown:
                raise ValueError(f"Unknown mask_policy options: {sorted(unknown)}")
            policy.update(mask_policy)
        return self._validate_policy(policy)

    def get_candidate_patch_mask(self, segmentation_mask, mask_policy=None):
        """Return bool [grid_h, grid_w]; True marks eligible masking locations."""
        policy = self._resolve_policy(mask_policy)
        if isinstance(segmentation_mask, torch.Tensor):
            segmentation_mask = segmentation_mask.detach().cpu().numpy()
        labels = np.asarray(segmentation_mask)
        expected = (self.height * self.patch_size, self.width * self.patch_size)
        if labels.ndim != 2 or labels.shape != expected:
            raise ValueError(
                f"Segmentation must be HxW={expected} after the same crop/resize/flip as the image; "
                f"got {labels.shape}"
            )
        if labels.dtype.kind not in "biu":
            raise ValueError("Segmentation must contain integer class labels, not probabilities or RGB colors")
        if policy["mask_region"] == "none":
            return np.ones(self.get_shape(), dtype=bool)

        valid = ~np.isin(labels, policy["ignore_ids"])
        if policy["class_ids"] is not None:
            valid &= np.isin(labels, policy["class_ids"])
        region = valid
        if policy["mask_region"] == "boundary":
            boundary = np.zeros_like(valid)
            vertical = labels[1:, :] != labels[:-1, :]
            horizontal = labels[:, 1:] != labels[:, :-1]
            boundary[1:, :] |= vertical & valid[1:, :]
            boundary[:-1, :] |= vertical & valid[:-1, :]
            boundary[:, 1:] |= horizontal & valid[:, 1:]
            boundary[:, :-1] |= horizontal & valid[:, :-1]
            region = boundary
            for _ in range(policy["boundary_width"] - 1):
                padded = np.pad(region, 1)
                region = np.logical_or.reduce([
                    padded[y:y + labels.shape[0], x:x + labels.shape[1]]
                    for y in range(3) for x in range(3)
                ])
            region &= valid

        coverage = region.reshape(self.height, self.patch_size, self.width, self.patch_size).mean(axis=(1, 3))
        return (coverage > 0) & (coverage >= policy["min_coverage"])

    def __call__(self, num_masking_patches=0, segmentation_mask=None, mask_policy=None):
        if segmentation_mask is None:
            return self.base_generator(num_masking_patches)
        if (
            not isinstance(num_masking_patches, Integral)
            or isinstance(num_masking_patches, bool)
            or not 0 <= num_masking_patches <= self.num_patches
        ):
            raise ValueError(f"num_masking_patches must be an integer between 0 and {self.num_patches}")
        candidates = self.get_candidate_patch_mask(segmentation_mask, mask_policy).ravel()
        count = min(num_masking_patches, int(candidates.sum()))
        mask = np.asarray(self.base_generator(count), dtype=bool)
        if mask.shape != self.get_shape() or int(mask.sum()) != count:
            raise ValueError("The base generator must return the requested number of masked patches on its grid")
        masked = mask.ravel() & candidates
        remaining = count - int(masked.sum())
        if remaining:
            destinations = np.flatnonzero(~masked & candidates)
            destinations = np.random.choice(destinations, size=remaining, replace=False)
            masked[destinations] = True
        return masked.reshape(self.get_shape())
