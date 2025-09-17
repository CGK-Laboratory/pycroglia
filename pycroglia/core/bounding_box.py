from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class ComputeResult:
    bounded_img: NDArray
    right: int
    left: int
    top: int
    bottom: int


def compute(input_img: NDArray) -> ComputeResult:
    """Compute the tight bounding box of a 3D binary image along the Z and Y axes.

    The function finds the minimum and maximum foreground voxel indices
    (value == 1 / True) along the Z (rows) and Y (columns) dimensions,
    crops the input volume accordingly, and keeps the full X (slices) range.

    Args:
        input_img (NDArray):
            A 3D boolean or uint8 array with shape (Z, Y, X).

    Returns:
        ComputeResult:
            A dataclass with the following fields:
            - bounded_img (NDArray): Cropped sub-volume
              [left:right+1, bottom:top+1, :].
            - right (int): Max Z index (0-based).
            - left (int): Min Z index (0-based).
            - top (int): Max Y index (0-based).
            - bottom (int): Min Y index (0-based).
    """
    assert input_img.ndim == 3, f"Expected a 3D array, got shape {input_img.shape}"

    # Find foreground voxel coordinates
    coords = np.argwhere(input_img)
    if coords.size == 0:
        raise ValueError("No foreground voxels found (all zeros).")

    # coords columns correspond to (x, y, z) == (row, col, slice) in NumPy
    z = coords[:, 0]
    y = coords[:, 1]

    z_min, z_max = int(z.min()), int(z.max())
    y_min, y_max = int(y.min()), int(y.max())

    bounded_img = input_img[z_min : z_max + 1, y_min : y_max + 1, :]

    return ComputeResult(
        bounded_img=bounded_img, right=z_max, left=z_min, top=y_max, bottom=y_min
    )
