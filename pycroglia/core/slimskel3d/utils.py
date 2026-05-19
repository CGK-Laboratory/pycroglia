import numpy as np
from numpy.typing import NDArray


def _get_neighbourhood(img: NDArray, indices: NDArray) -> NDArray:
    """Return the 3x3x3 neighborhood of given voxels in a 3D binary image.

    This function collects the values of all 27 neighbors (including the voxel
    itself) around each input voxel index.  The input ``img`` must be padded such
    that none of the ``indices`` lie on the real boundary, ensuring safe index
    offsets without bounds checking.

    Uses vectorised offset broadcasting for performance.

    Args:
        img (NDArray):
            A padded 3D binary image (bool or int), shape ``(Z, Y, X)``.
        indices (NDArray):
            1-D array of linear indices (0-based, flattened order,
            i.e. as returned by ``np.flatnonzero`` or ``img.ravel()``).

    Returns:
        NDArray:
            A boolean array of shape ``(len(indices), 27)``, where each row
            contains the 27-neighborhood of a voxel in row-major (z, y, x) order.
    """
    Z, Y, X = img.shape
    dz, dy, dx = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")
    offsets = dz.ravel() * (Y * X) + dy.ravel() * X + dx.ravel()
    return img.ravel()[indices[:, np.newaxis] + offsets[np.newaxis, :]].astype(bool)
