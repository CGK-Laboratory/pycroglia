from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.ndimage import convolve
import numpy as np
from skimage.morphology import star

@dataclass
class Point:
    """A 3D voxel coordinate in (x, y, z)."""

    x: int
    y: int
    z: int


# 26-neighbourhood offsets (all possible moves except staying in place)
CUBE = np.ones((3,3,3))

def connect_points_along_path(img: NDArray, start: Point, end: Point) -> NDArray:
    """
    Find the shortest path between two voxels constrained to foreground (True) voxels.

    This function performs a **breadth-first search (BFS)** on a 3D binary volume to
    trace the shortest path from `start` to `end`, moving only through connected
    voxels that have value `True`.

    Args:
        img (NDArray): 3D binary array of shape (Z, Y, X).
                       Foreground voxels must have value True.
        start (Point): Starting voxel, given as (x, y, z).
        end (Point): Ending voxel, given as (x, y, z).

    Returns:
        NDArray: Array of shape (N, 3) with the sequence of voxel coordinates
                 forming the path, ordered as (z, y, x).
                 If no path is found, returns an empty array.

    Notes:
        - Neighbourhood is 26-connected (includes diagonals).
        - BFS guarantees the path found is the shortest in terms of voxel steps.
        - The coordinate system is **0-based, z-y-x order** (NumPy convention).
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

    mask = (D == 0)
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
