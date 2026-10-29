"""DINOv3 augmentation with spatially aligned segmentation input."""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from .augmentations import DataAugmentation
from .structure_augmentations import validate_segmentation_mask


class _AlignedGlobalGeometry:
    """Replay each sampled image geometry on its integer label map."""

    def __init__(self, geometry, segmentation, output_size):
        self.crop, self.flip = geometry.transforms
        self.segmentation = torch.from_numpy(np.array(segmentation, copy=True))
        self.output_size = output_size
        self.masks = []

    def __call__(self, image):
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            image, self.crop.scale, self.crop.ratio,
        )
        image = TF.resized_crop(
            image, top, left, height, width, self.crop.size,
            self.crop.interpolation, antialias=self.crop.antialias,
        )
        labels = self.segmentation[top:top + height, left:left + width]
        # Integer indexing preserves IDs, including IDs not exactly representable
        # in float32. Match the same nearest-neighbor geometry as a label image.
        out_h, out_w = self.crop.size
        yy = (2 * torch.arange(out_h) + 1) * height // (2 * out_h)
        xx = (2 * torch.arange(out_w) + 1) * width // (2 * out_w)
        labels = labels[yy[:, None], xx[None, :]]
        if torch.rand(1) < self.flip.p:
            image = TF.hflip(image)
            labels = labels.flip(-1)
        size = self.output_size
        yy = (2 * torch.arange(size) + 1) * out_h // (2 * size)
        xx = (2 * torch.arange(size) + 1) * out_w // (2 * size)
        self.masks.append(labels[yy[:, None], xx[None, :]].contiguous())
        return image


class SegmentationDataAugmentation(DataAugmentation):
    """The original DINOv3 crops and appearance transforms, plus label maps.

    Accept ``augment(image, segmentation_mask=labels, mask_policy=policy)`` or
    ``augment({'image': image, 'seg': labels, 'mask_policy': policy})``.
    Teacher, Gram-teacher, local-crop and offsets outputs follow DINOv3.
    """

    def __call__(self, image, segmentation_mask=None, mask_policy=None):
        if isinstance(image, dict):
            if segmentation_mask is not None:
                raise ValueError("Pass segmentation_mask either in the sample or as an argument")
            sample = image
            image = sample["image"]
            segmentation_mask = sample.get("segmentation_mask", sample.get("seg"))
            if mask_policy is None:
                mask_policy = sample.get("mask_policy")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")

        if segmentation_mask is None:
            output = super().__call__(image)
            output["global_segmentation_masks"] = [None] * len(output["global_crops"])
        else:
            labels = validate_segmentation_mask(segmentation_mask, (image.height, image.width))
            geometry = self.geometric_augmentation_global
            paired = _AlignedGlobalGeometry(geometry, labels, self.global_crops_size)
            self.geometric_augmentation_global = paired
            try:
                output = super().__call__(image)
            finally:
                self.geometric_augmentation_global = geometry
            output["global_segmentation_masks"] = paired.masks
        if mask_policy is not None:
            output["mask_policy"] = dict(mask_policy)
        return output
