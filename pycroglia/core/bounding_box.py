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
    """Compute the tight bounding box of a 3D binary image along the X and Y axes.

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
            - right (int): Max X index (0-based).
            - left (int): Min X index (0-based).
            - top (int): Max Y index (0-based).
            - bottom (int): Min Y index (0-based).
    """
    assert input_img.ndim == 3, f"Expected 3D array, got {input_img.shape}"

    coords = np.argwhere(input_img)
    assert coords.size > 0, "No foreground voxels found."

    y = coords[:, 1]
    x = coords[:, 2]

    bottom, top = int(y.min()), int(y.max())
    left, right = int(x.min()), int(x.max())

    bounded_img = input_img[:, bottom : top + 1, left : right + 1]

    return ComputeResult(
        bounded_img=bounded_img,
        right=right,
        left=left,
        top=top,
        bottom=bottom,
    )
