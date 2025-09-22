from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

__all__ = ["overlay_slices"]


def overlay_slices(fixed_img: Path, moved_img: Path, alpha: float = 0.5, slice_index: int = None):
    """Quick matplotlib overlay of one axial slice to visually inspect registration."""
    fixed = sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_img)))
    moved = sitk.GetArrayFromImage(sitk.ReadImage(str(moved_img)))

    if fixed.shape != moved.shape:
        raise ValueError("Images have different shapes; cannot overlay directly.")

    z = slice_index if slice_index is not None else fixed.shape[0] // 2
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(fixed[z, :, :], cmap='gray'); plt.title('Fixed')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(moved[z, :, :], cmap='gray'); plt.title('Moved')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(fixed[z, :, :], cmap='gray')
    plt.imshow(moved[z, :, :], cmap='hot', alpha=alpha)
    plt.title('Overlay')
    plt.axis('off')
    plt.tight_layout()
    plt.show() 