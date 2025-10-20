import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class Coordinates:
    """Container for 3D voxel coordinates."""

    x: int
    y: int
    z: int


def compute(
    img: NDArray, starting_pixel: tuple[int, int, int], scale: float
) -> Coordinates:
    """
    Find the nearest foreground voxel (value=1) in a 3D binary image.

    Args:
        img (NDArray):
            3D binary image of shape (z, y, x). Foreground voxels are 1 or True.
        starting_pixel (tuple[int,int,int]):
            Starting voxel coordinates in (z, y, x) order.
        scale (float):
            Anisotropy scaling factor applied equally to the z and y distances
            (mimicking the MATLAB implementation).

    Returns:
        Coordinates:
            Dataclass containing (x, y, z) of the closest foreground voxel.

    Raises:
        ValueError: If the image contains no foreground voxels.
    """
    # Foreground voxel coordinates
    coords = np.argwhere(img == 1)

    if coords.size == 0:
        raise ValueError("No foreground voxels found.")

    z0, y0, x0 = starting_pixel

    # Apply anisotropic distance
    diffs = coords - np.array([z0, y0, x0])
    dists = np.sqrt(
        (diffs[:, 0]) ** 2  # z difference scaled
        + (diffs[:, 1] * scale) ** 2  # y difference scaled
        + (diffs[:, 2] * scale) ** 2
    )  # x difference unchanged

    idx = np.argmin(dists)
    z, y, x = coords[idx]

    return Coordinates(x=int(x), y=int(y), z=int(z))
