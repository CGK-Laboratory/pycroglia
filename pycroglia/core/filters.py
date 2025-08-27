import cv2
import skimage
import numpy as np

from numpy.typing import NDArray

from pycroglia.core.enums import SkimageCellConnectivity


def calculate_otsu_threshold(img: NDArray, adjust: float) -> NDArray:
    """Calculates a binary mask for each slice of a 3D image using Otsu's method and a threshold adjustment factor.

    Args:
        img (NDArray): 3D image array with shape (zs, height, width), where zs is the number of slices.
        adjust (float): Adjustment factor to modify the threshold computed by Otsu's method.

    Returns:
        NDArray: Boolean 3D array (same shape as input) representing the binary thresholded mask.
    """
    zs, height, width = img.shape
    binary_stack = np.zeros((zs, height, width), dtype=np.uint8)

    for i in range(zs):
        z_slice = img[i, :, :].astype(np.uint8)
        # Otsu method for obtaining the threshold
        level, _ = cv2.threshold(z_slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adjusted_level = min(255.0, level * adjust)

        # Apply the adjusted level
        _, obtained_slice = cv2.threshold(
            z_slice, adjusted_level, 255, cv2.THRESH_BINARY
        )
        binary_stack[i, :, :] = obtained_slice > 0

    return binary_stack


def remove_small_objects(
    img: NDArray,
    min_size: int,
    connectivity: SkimageCellConnectivity = SkimageCellConnectivity.EDGES,
) -> NDArray:
    """Removes connected components smaller than a given size from a 3D binary mask.

    Args:
        img (NDArray): 3D binary array (dtype=bool or uint8) with shape (zs, height, width).
        min_size (int): Minimum number of pixels required to keep a component.
        connectivity (SkimageCellConnectivity): Connectivity used by skimage (4 or 8). Defaults to SkimageCellConnectivity.EDGES.

    Returns:
        NDArray: 3D binary array with small objects removed.
    """
    img_bool = img.astype(bool)
    labeled_img = skimage.morphology.label(img_bool, connectivity=connectivity.value)
    filtered = skimage.morphology.remove_small_objects(
        labeled_img, min_size=min_size, connectivity=connectivity.value
    )
    return filtered.astype(img.dtype)
