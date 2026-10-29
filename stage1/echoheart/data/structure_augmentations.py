"""Structure-aware crops: global views around segmented anatomical structures,
local views on structure contours, and a sector-mask fallback for unlabeled frames."""

import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


# =========================
# Helpers
# =========================

def ensure_rgb(img_np: np.ndarray) -> np.ndarray:
    if img_np.ndim == 2:
        return np.stack([img_np]*3, axis=-1)
    if img_np.shape[2] == 1:
        return np.repeat(img_np, 3, axis=2)
    return img_np

def validate_segmentation_mask(seg, image_shape: Tuple[int, int]) -> np.ndarray:
    """Keep integer class IDs and require pixel alignment with the input image."""
    if isinstance(seg, torch.Tensor):
        seg = seg.detach().cpu().numpy()
    seg = np.asarray(seg)
    if seg.ndim != 2 or tuple(seg.shape) != tuple(image_shape):
        raise ValueError(f"Segmentation must have image shape {image_shape}, got {seg.shape}")
    if seg.dtype.kind not in "biu":
        raise ValueError("Segmentation must contain integer class IDs")
    if seg.dtype.kind == "u" and seg.size and seg.max() > np.iinfo(np.int64).max:
        raise ValueError("Segmentation class IDs must fit in int64")
    return np.asarray(seg, dtype=np.int64)


def resize_segmentation_crop(seg: np.ndarray, rect: Tuple[int, int, int, int], size: int):
    """Apply the image crop rectangle and nearest-neighbour resize to labels."""
    x, y, width, height = rect
    crop = seg[y:y + height, x:x + width]
    if not crop.size:
        raise ValueError("Cannot resize an empty segmentation crop")
    # Pixel-centre nearest sampling keeps labels integral throughout resizing.
    rows = np.minimum(((np.arange(size) + 0.5) * crop.shape[0] / size).astype(int), crop.shape[0] - 1)
    cols = np.minimum(((np.arange(size) + 0.5) * crop.shape[1] / size).astype(int), crop.shape[1] - 1)
    return torch.from_numpy(np.ascontiguousarray(crop[np.ix_(rows, cols)], dtype=np.int64))


def load_binary_mask(path: str, H: int, W: int) -> np.ndarray:
    """Read the ultrasound sector mask resized to (H, W); True marks pixels inside the sector."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sector mask not found: {path}")
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = (m > 0).astype(np.uint8)
    if (m.shape[0] != H) or (m.shape[1] != W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)

def get_present_classes(seg: np.ndarray,
                        class_whitelist: Optional[Sequence[int]],
                        ignore_ids: Sequence[int],
                        min_class_pixels: int) -> List[int]:
    """Return class IDs present in the frame, excluding ignored, non-whitelisted and tiny regions."""
    vals, cnts = np.unique(seg, return_counts=True)
    present = []
    ignore_set = set(ignore_ids)
    whitelist_set = set(class_whitelist) if class_whitelist is not None else None
    for v, c in zip(vals, cnts):
        if v in ignore_set:
            continue
        if whitelist_set is not None and v not in whitelist_set:
            continue
        if c < min_class_pixels:
            continue
        present.append(int(v))
    return present

def bbox_of_class(seg: np.ndarray, cls_id: int, margin: float = 0.10
                  ) -> Optional[Tuple[int,int,int,int]]:
    """Square bounding box (x, y, w, h) of one class with a relative margin, or None if absent."""
    mask = (seg == cls_id).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    H, W = seg.shape[:2]
    x0 = max(0, int(x0 - margin * w)); y0 = max(0, int(y0 - margin * h))
    x1 = min(W, int(x1 + margin * w)); y1 = min(H, int(y1 + margin * h))
    ww, hh = x1 - x0, y1 - y0
    side = max(ww, hh)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0 = max(0, cx - side // 2); y0 = max(0, cy - side // 2)
    x1 = min(W, x0 + side);      y1 = min(H, y0 + side)
    return (x0, y0, x1 - x0, y1 - y0)

def contour_points_of_class(seg: np.ndarray, cls_id: int) -> Optional[np.ndarray]:
    """Return all external contour points of one class as an (N, 2) array of (x, y), or None."""
    mask = (seg == cls_id).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts = np.vstack([c.reshape(-1,2) for c in cnts])
    return pts  # (N,2)

def crop_square(img_np: np.ndarray, center: Tuple[int,int], size: int) -> np.ndarray:
    """Crop a size x size square around center, reflect-padding outside the image."""
    H, W = img_np.shape[:2]; x, y = center; half = size // 2
    x0, y0 = x - half, y - half; x1, y1 = x0 + size, y0 + size
    padL, padT = max(0, -x0), max(0, -y0)
    padR, padB = max(0, x1 - W), max(0, y1 - H)
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    patch = img_np[y0:y1, x0:x1]
    if any([padL, padT, padR, padB]):
        patch = cv2.copyMakeBorder(patch, padT, padB, padL, padR, cv2.BORDER_REFLECT_101)
    return patch

def rect_inside_mask(mask: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """Return True when all four rectangle corners lie inside the mask."""
    H, W = mask.shape
    x1, y1 = min(W-1, x+w-1), min(H-1, y+h-1)
    for (cx,cy) in [(x,y),(x1,y),(x,y1),(x1,y1)]:
        if cx<0 or cy<0 or cx>=W or cy>=H: return False
        if not mask[cy,cx]: return False
    return True

def random_resized_rect_in_mask(mask: np.ndarray,
                                scale: Tuple[float,float]=(0.32,1.0),
                                ratio: Tuple[float,float]=(3/4,4/3),
                                trials: int = 50) -> Tuple[int,int,int,int]:
    """RandomResizedCrop-style rectangle (x, y, w, h) whose corners lie inside the mask."""
    H,W = mask.shape; area = H*W
    ys, xs = np.where(mask>0)
    if len(xs)==0: return (0,0,W,H)
    x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
    for _ in range(trials):
        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        a = math.exp(random.uniform(*log_ratio))  # w/h
        w = int(round(math.sqrt(target_area * a)))
        h = int(round(math.sqrt(target_area / a)))
        if w<=0 or h<=0: continue
        if w > (x1-x0+1) or h > (y1-y0+1): continue
        for _ in range(20):
            x = random.randint(x0, max(x0, x1-w+1))
            y = random.randint(y0, max(y0, y1-h+1))
            if rect_inside_mask(mask, x, y, w, h):
                return (x, y, w, h)
    # Fallback: the largest square centered in the mask bounding box.
    side = min(x1-x0+1, y1-y0+1)
    cx, cy = (x0+x1)//2, (y0+y1)//2
    x = max(0, cx - side//2); y = max(0, cy - side//2)
    return (x, y, side, side)


# =========================
# Augmentation
# =========================

class StructureDataAugmentation(object):
    """Crop around anatomical structures instead of random image regions.

    With segmentation, each of the two global crops is the bounding box of one
    randomly chosen class (classes are not merged), and the K local crops are
    centered on contour points sampled across classes, weighted by contour
    length. Without segmentation, global and local crops are sampled inside
    the ultrasound sector given by ``sector_mask_path``.
    """
    def __init__(self,
                 global_size: int = 224,
                 local_size: int = 96,
                 local_num: int = 8,
                 # Frames contain different subsets of classes; these select usable ones.
                 class_whitelist: Optional[Sequence[int]] = None,  # None allows every class
                 ignore_ids: Sequence[int] = (0,),                  # e.g. 0 = background
                 min_class_pixels: int = 64,                        # drop regions smaller than this
                 margin: float = 0.10,
                 global_scale: Tuple[float,float] = (0.32, 1.0),
                 global_ratio: Tuple[float,float] = (3/4, 4/3),
                 sector_mask_path: Optional[str] = None,
                 return_segmentation_masks: bool = False):
        self.global_size = global_size
        self.local_size  = local_size
        self.local_num   = local_num
        self.class_whitelist = tuple(class_whitelist) if class_whitelist is not None else None
        self.ignore_ids = tuple(ignore_ids)
        self.min_class_pixels = int(min_class_pixels)
        self.margin = margin
        self.global_scale = global_scale
        self.global_ratio = global_ratio
        self.sector_mask_path = sector_mask_path
        self.return_segmentation_masks = return_segmentation_masks
        self._sector_cache: Dict[Tuple[int,int], np.ndarray] = {}

        # Appearance transforms
        norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        jitter = T.ColorJitter(0.2,0.2,0.1,0.05)
        blur_g = T.GaussianBlur(23, sigma=(0.1, 2.0))
        blur_l = T.GaussianBlur(13, sigma=(0.1, 2.0))
        self.global_transfo1 = T.Compose([jitter, T.RandomGrayscale(p=0.2), blur_g, T.ToTensor(), norm])
        self.global_transfo2 = T.Compose([jitter, T.RandomGrayscale(p=0.2), blur_g, T.ToTensor(), norm])
        self.local_transfo   = T.Compose([jitter, T.RandomGrayscale(p=0.2), blur_l, T.ToTensor(), norm])

    # ---------- Without segmentation: random crops inside the sector ----------
    def _get_sector_mask(self, H: int, W: int) -> np.ndarray:
        key = (H,W)
        if key not in self._sector_cache:
            if self.sector_mask_path is None:
                raise RuntimeError("sector_mask_path is required to crop frames without segmentation")
            self._sector_cache[key] = load_binary_mask(self.sector_mask_path, H, W)
        return self._sector_cache[key]

    def _without_seg(self, img_np: np.ndarray, seg: Optional[np.ndarray] = None):
        H,W = img_np.shape[:2]
        mask = self._get_sector_mask(H, W)

        # Two global crops inside the sector
        rects = [
            random_resized_rect_in_mask(mask, self.global_scale, self.global_ratio),
            random_resized_rect_in_mask(mask, self.global_scale, self.global_ratio),
        ]
        glist = []
        for i,(x,y,w,h) in enumerate(rects):
            crop = img_np[y:y+h, x:x+w]
            pil  = Image.fromarray(crop).resize((self.global_size, self.global_size), Image.BICUBIC)
            glist.append(self.global_transfo1(pil) if i==0 else self.global_transfo2(pil))

        # K local crops centered at random points inside the sector
        centers = []
        for _ in range(self.local_num):
            x,y,w,h = random_resized_rect_in_mask(mask, (0.05,0.05), (1.0,1.0))  # small rectangle used as a random center
            cx, cy = x + w//2, y + h//2
            centers.append((cx,cy))
        locals_list = []
        for (cx,cy) in centers:
            patch = crop_square(img_np, (cx,cy), self.local_size)
            locals_list.append(self.local_transfo(Image.fromarray(patch)))

        if self.return_segmentation_masks:
            segmentation_masks = [
                None if seg is None else resize_segmentation_crop(seg, rect, self.global_size)
                for rect in rects
            ]
            return glist, locals_list, segmentation_masks
        return glist, locals_list

    # ---------- With segmentation: crops per anatomical class ----------
    def _with_seg(self, img_np: np.ndarray, seg: np.ndarray):
        # 1) Classes present in this frame
        present = get_present_classes(seg,
                                      class_whitelist=self.class_whitelist,
                                      ignore_ids=self.ignore_ids,
                                      min_class_pixels=self.min_class_pixels)
        if len(present) == 0:
            return None  # fall back to sector cropping

        # 2) Global crops: bounding boxes of two sampled classes (repeated if only one exists)
        if len(present) >= 2:
            global_classes = random.sample(present, 2)
        else:
            global_classes = [present[0], present[0]]
        glist = []
        segmentation_masks = []
        for i, cid in enumerate(global_classes):
            roi = bbox_of_class(seg, cid, self.margin)
            if roi is None:
                # Class vanished after filtering; use the full frame
                x,y,w,h = 0,0,img_np.shape[1], img_np.shape[0]
            else:
                x,y,w,h = roi
            crop = img_np[y:y+h, x:x+w]
            pil  = Image.fromarray(crop).resize((self.global_size, self.global_size), Image.BICUBIC)
            glist.append(self.global_transfo1(pil) if i==0 else self.global_transfo2(pil))

            if self.return_segmentation_masks:
                segmentation_masks.append(resize_segmentation_crop(seg, (x, y, w, h), self.global_size))

        # 3) Local crops: K contour points, classes weighted by contour length
        per_class_pts: Dict[int, np.ndarray] = {}
        weights = []
        ids_for_sampling = []
        for cid in present:
            pts = contour_points_of_class(seg, cid)
            if pts is not None and len(pts) > 0:
                per_class_pts[cid] = pts
                ids_for_sampling.append(cid)
                weights.append(len(pts))
        locals_list = []
        centers: List[Tuple[int,int]] = []

        if len(ids_for_sampling) == 0:
            # No contours: center every local crop on the frame center
            H,W = seg.shape[:2]
            grid = [(W//2, H//2)] * self.local_num
            centers = grid[:self.local_num]
        else:
            weights = np.array(weights, dtype=np.float64)
            probs = (weights / weights.sum()).tolist()
            for _ in range(self.local_num):
                cid = random.choices(ids_for_sampling, weights=probs, k=1)[0]
                pts = per_class_pts[cid]
                px, py = pts[np.random.randint(0, len(pts))]  # (x,y)
                centers.append((int(px), int(py)))

        for (cx,cy) in centers:
            patch = crop_square(img_np, (cx,cy), self.local_size)
            locals_list.append(self.local_transfo(Image.fromarray(patch)))

        if self.return_segmentation_masks:
            return glist, locals_list, segmentation_masks
        return glist, locals_list

    # ---------- Entry point ----------
    def __call__(self, image, segmentation_mask=None, mask_policy=None):
        """Accept a PIL image with optional ``segmentation_mask``/``mask_policy``,
        or a dict ``{"image": PIL.Image, "seg": integer label map, "mask_policy": ...}``.
        With ``return_segmentation_masks=True`` the output also holds labels aligned
        with the two global crops.
        """
        if isinstance(image, Image.Image):
            image_pil = image
        elif isinstance(image, dict):
            image_pil = image["image"]
            if segmentation_mask is None:
                segmentation_mask = image.get("seg", image.get("segmentation_mask"))
            if mask_policy is None:
                mask_policy = image.get("mask_policy")
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")
        if not isinstance(image_pil, Image.Image):
            raise TypeError("image must be a PIL.Image")

        img_np = ensure_rgb(np.array(image_pil))
        seg = (None if segmentation_mask is None else
               validate_segmentation_mask(segmentation_mask, img_np.shape[:2]))

        out = None
        if seg is not None:
            out = self._with_seg(img_np, seg)
        if out is None:
            out = self._without_seg(img_np, seg)

        g_list, locals_list = out[:2]
        result = {
            "global_crops": g_list,
            "global_crops_teacher": g_list,
            "local_crops": locals_list[: self.local_num],
        }
        if self.return_segmentation_masks:
            result["global_segmentation_masks"] = out[2]
        if mask_policy is not None:
            result["mask_policy"] = mask_policy
        return result
