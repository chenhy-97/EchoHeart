"""Build the EchoHeart segmentation input and DINOv3 mask selection interface."""

from .structure_augmentations import StructureDataAugmentation
from .datasets.structure import StructureImageFolder
from .mask_augmentations import SegmentationDataAugmentation
from .structure_masking import StructureAwareMaskingGenerator


def build_data_pipeline(cfg, mask_generator):
    """Return (dataset, optional mask_selector) for the standard DINOv3 collator."""
    data = getattr(cfg, "data", None)
    guidance = getattr(cfg, "mask_guidance", None)
    enabled = getattr(guidance, "enabled", False)
    crops = cfg.crops
    patch_size = cfg.student.patch_size
    if crops.global_crops_size % patch_size:
        raise ValueError("global_crops_size must be a multiple of student.patch_size")

    selector = None
    if enabled:
        if cfg.multidistillation.enabled or cfg.distillation.enabled:
            raise ValueError("mask_guidance uses DINOv3 self-supervised pretraining; set distillation.enabled and multidistillation.enabled to false")
        if cfg.ibot.mask_random_circular_shift:
            raise ValueError("Set ibot.mask_random_circular_shift=false for aligned segmentation masks")
        selector = StructureAwareMaskingGenerator(
            mask_generator,
            patch_size=patch_size,
            mask_region=getattr(guidance, "mask_region", "foreground"),
            class_ids=getattr(guidance, "class_ids", None),
            ignore_ids=getattr(guidance, "ignore_ids", (0,)),
            min_coverage=getattr(guidance, "min_coverage", 0.0),
            boundary_width=getattr(guidance, "boundary_width", 1),
        )

    strategy = getattr(data, "crop_strategy", "standard")
    if getattr(crops, "teacher_no_color_jitter", False):
        raise ValueError("This SSL trainer uses shared student/teacher global crops; set crops.teacher_no_color_jitter=false")
    if strategy == "standard":
        if crops.localcrops_subset_of_globalcrops:
            if crops.local_crops_number % 2 or crops.global_crops_size - crops.local_crops_size < patch_size:
                raise ValueError("Local-crop subsets need an even crop count and global_crops_size >= local_crops_size + patch_size")
        augmentation = SegmentationDataAugmentation(
            crops.global_crops_scale,
            crops.local_crops_scale,
            crops.local_crops_number,
            global_crops_size=crops.global_crops_size,
            local_crops_size=crops.local_crops_size,
            gram_teacher_crops_size=crops.gram_teacher_crops_size,
            gram_teacher_no_distortions=crops.gram_teacher_no_distortions,
            local_crops_subset_of_global_crops=crops.localcrops_subset_of_globalcrops,
            patch_size=patch_size,
            share_color_jitter=crops.share_color_jitter,
            horizontal_flips=crops.horizontal_flips,
            mean=crops.rgb_mean,
            std=crops.rgb_std,
        )
    elif strategy == "structure_aware":
        seg_aug = getattr(cfg, "seg_aug", None)
        if crops.gram_teacher_crops_size is not None or crops.localcrops_subset_of_globalcrops:
            raise ValueError("Use data.crop_strategy=standard for Gram-teacher crops or local-crop subsets")
        augmentation = StructureDataAugmentation(
            global_size=crops.global_crops_size,
            local_size=crops.local_crops_size,
            local_num=crops.local_crops_number,
            sector_mask_path=getattr(data, "sector_mask_path", None),
            class_whitelist=getattr(seg_aug, "class_whitelist", None),
            ignore_ids=getattr(seg_aug, "ignore_ids", (0,)),
            min_class_pixels=getattr(seg_aug, "min_class_pixels", 64),
            margin=getattr(seg_aug, "margin", 0.10),
            global_scale=getattr(data, "global_crops_scale", crops.global_crops_scale),
            return_segmentation_masks=enabled,
        )
    else:
        raise ValueError("data.crop_strategy must be standard or structure_aware")

    seg_root = getattr(cfg.train, "seg_root", None)
    if seg_root is None:
        seg_root = getattr(data, "seg_path", None)
    if not getattr(data, "use_seg", True):
        seg_root = None
    dataset = StructureImageFolder(
        root=cfg.train.image_root,
        seg_root=seg_root,
        transforms=augmentation,
        require_seg=getattr(guidance, "require_seg", False) if enabled else False,
        mask_policy_path=getattr(guidance, "policy_path", None) if enabled else None,
    )
    return dataset, selector
