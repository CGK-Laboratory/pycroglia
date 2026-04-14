import warnings
import skimage
import numpy as np

from numpy.typing import NDArray

from pycroglia.core.enums import SkimageCellConnectivity
from skimage.exposure import histogram


def _threshold_otsu(image):
    """Return otsu threshold, using matlab's criteria about repeated values.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray.
        Input image.

    Returns
    -------
    threshold : float
        Pixels higher than this are assumed to be foreground.

    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method

    """

    unique_value = image.reshape(-1)[0]
    
    if np.all(image == unique_value):
        return unique_value

    counts, bin_centers = histogram(
                image.reshape(-1), source_range='image', normalize=False
            )
    counts = counts.astype('float32', copy=False)

    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]

    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = max(np.where(variance12 == max(variance12))[0])  # In case of multiple max, take the last one

    return bin_centers[idx]


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
        z_slice = img[i, :, :]
        # Otsu method for obtaining the threshold
        level = _threshold_otsu(z_slice)
        adjusted_level = level * adjust

        # Apply the adjusted level
        binary_stack[i, :, :] = z_slice > adjusted_level

    return binary_stack


def remove_small_objects(
    img: NDArray,
    min_size: int,
    connectivity: SkimageCellConnectivity = SkimageCellConnectivity.CORNERS,
) -> NDArray:
    """Removes connected components smaller than a given size from a 3D binary mask.

    Args:
        img (NDArray): 3D binary array (dtype=bool or uint8) with shape (zs, height, width).
        min_size (int): Minimum number of pixels required to keep a component.
        connectivity (SkimageCellConnectivity): Connectivity used by skimage (4 or 8). Defaults to SkimageCellConnectivity.CORNERS.

    Returns:
        NDArray: 3D binary array with small objects removed.
    """
    img_bool = img.astype(bool)
    labeled_img = skimage.morphology.label(img_bool, connectivity=connectivity.value)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Only one label was provided to `remove_small_objects`"
        )
        filtered = skimage.morphology.remove_small_objects(
            labeled_img, max_size=min_size-1, connectivity=connectivity.value
        )
    result = filtered > 0
    return result.astype(img.dtype)
