from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.ndimage import convolve
import numpy as np


@dataclass
class Point:
    """A 3D voxel coordinate in (x, y, z)."""

    x: int
    y: int
    z: int


# 26-neighbourhood offsets (all possible moves except staying in place)
CUBE = np.ones((3, 3, 3))


def connect_points_along_path(img: NDArray, start: Point, end: Point) -> NDArray:
    """
    Compute the shortest 3D path connecting two voxels within a binary skeleton mask.

    This function performs a two-front **breadth-first search (BFS)** expansion
    from both `start` and `end` voxels simultaneously through a 3D binary volume.
    Expansion proceeds only through foreground voxels (`True`), and stops once
    both fronts meet, marking all voxels that belong to the minimal connecting
    region.

    The method mirrors MATLAB’s `ConnectPointsAlongPath` behavior: it iteratively
    dilates both fronts (start and end) through a 26-connected neighborhood
    until they overlap, at which point the minimal connecting path is extracted.

    Args:
        img (NDArray):
            3D binary array of shape `(Z, Y, X)`. Foreground voxels (path-eligible)
            must have value `True`.
        start (Point):
            Starting voxel coordinates as `(x, y, z)`. Must lie inside the foreground.
        end (Point):
            Ending voxel coordinates as `(x, y, z)`. Must lie inside the foreground.

    Returns:
        NDArray:
            A 3D binary mask of the same shape as `img`, where `1` (True) marks
            voxels belonging to the **shortest connection** between `start` and `end`.
            If no connection exists, raises a `ValueError`.

    Raises:
        ValueError:
            If no continuous path of foreground voxels connects `start` and `end`.
    """
    assert img.ndim == 3, "Input must be 3D"
    assert img[start.z, start.y, start.x], "Start point must be inside skeleton"
    assert img[end.z, end.y, end.x], "End point must be inside skeleton"

    # Initialize D: duplicate input image into 4D [z, y, x, 2]
    D_layer = np.where(img, np.inf, np.nan)
    D = np.stack([D_layer.copy(), D_layer.copy()], axis=-1)

    # Set start and end points to 0
    D[start.z, start.y, start.x, 0] = 0
    D[end.z, end.y, end.x, 1] = 0

    mask = D == 0
    n = 0

    # Iteratively expand mask until connection found or no more reachable voxels
    while np.isinf(D[end.z, end.y, end.x, 0]) and np.count_nonzero(mask):
        n += 1
        # Convolve mask to find neighboring voxels still at infinity
        for k in range(2):
            layer = mask[..., k].astype(float)
            layer = convolve(layer, CUBE, mode="constant", cval=0) > 0
            layer &= np.isinf(D[..., k])
            D[..., k][layer] = n
            mask[..., k] = layer

    # If endpoint still infinite, no path was found
    if np.isinf(D[end.z, end.y, end.x, 0]):
        raise ValueError("No path found between points.")
    else:
        # Combine both layers and keep only voxels where sum == n
        mask = np.sum(D, axis=-1) == n

    return mask.astype(np.uint8)
