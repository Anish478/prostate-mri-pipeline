from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import SimpleITK as sitk

__all__ = [
    "dice",
    "jaccard",
    "mse",
    "compute_all"
]


def _load(img_or_path):
    if isinstance(img_or_path, (str, Path)):
        return sitk.ReadImage(str(img_or_path))
    return img_or_path


def dice(pred, gt) -> float:
    """Dice similarity coefficient between two binary masks."""
    pred = sitk.GetArrayFromImage(_load(pred)) > 0
    gt = sitk.GetArrayFromImage(_load(gt)) > 0

    inter = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    return 2.0 * inter / union if union > 0 else np.nan


def jaccard(pred, gt) -> float:
    """Jaccard index between two binary masks."""
    pred = sitk.GetArrayFromImage(_load(pred)) > 0
    gt = sitk.GetArrayFromImage(_load(gt)) > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else np.nan


def mse(img1, img2) -> float:
    """Mean squared error between two images (not necessarily masks)."""
    a = sitk.GetArrayFromImage(_load(img1)).astype(np.float32)
    b = sitk.GetArrayFromImage(_load(img2)).astype(np.float32)
    return float(np.mean((a - b) ** 2))


def compute_all(pred_mask: Path,
                gt_mask: Optional[Path] = None,
                fixed_img: Optional[Path] = None,
                moved_img: Optional[Path] = None) -> dict:
    """Utility to compute whatever is available.

    Returns dictionary of metrics that were possible to compute.
    """
    metrics = {}
    if gt_mask and gt_mask.exists():
        metrics["dice"] = dice(pred_mask, gt_mask)
        metrics["jaccard"] = jaccard(pred_mask, gt_mask)
    if fixed_img and moved_img:
        metrics["mse"] = mse(fixed_img, moved_img)
    return metrics 