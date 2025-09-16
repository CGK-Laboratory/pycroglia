from dataclasses import dataclass
from numpy.typing import NDArray
from collections import deque
import numpy as np


@dataclass
class Point:
    """A 3D voxel coordinate in (x, y, z)."""

    x: int
    y: int
    z: int


# 26-neighbourhood offsets (all possible moves except staying in place)
NEIGHBOURS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == dy == dx == 0)
]


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
    assert img.ndim == 3, "the image should be 3D"

    shape = img.shape
    visited = np.zeros(shape, dtype=bool)
    prev = {}

    # BFS queue
    q = deque()
    q.append((start.z, start.y, start.x))
    visited[start.z, start.y, start.x] = True

    found = False
    while q:
        z, y, x = q.popleft()
        if (z, y, x) == (end.z, end.y, end.x):
            found = True
            break

        for dz, dy, dx in NEIGHBOURS:
            zn, yn, xn = z + dz, y + dy, x + dx
            if (
                0 <= zn < shape[0]
                and 0 <= yn < shape[1]
                and 0 <= xn < shape[2]
                and img[zn, yn, xn]
                and not visited[zn, yn, xn]
            ):
                visited[zn, yn, xn] = True
                prev[(zn, yn, xn)] = (z, y, x)
                q.append((zn, yn, xn))

    if not found:
        return np.array([])

    path = []
    curr = (end.z, end.y, end.x)
    while curr != (start.z, start.y, start.x):
        path.append(curr)
        curr = prev[curr]
    path.append((start.z, start.y, start.x))
    path.reverse()

    return np.array(path, dtype=int)  # shape (N, 3), order (z, y, x)
