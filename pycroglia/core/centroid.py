from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist


class Centroids:
    """Computes centroids of 3D cell masks and their average pairwise distance.

    This class takes a list of 3D binary masks, extracts the centroid of each
    non-empty mask, and provides a method to compute the average pairwise
    distance between all centroids in physical units.

    Attributes:
        centroids (NDArray[np.float64]): Array of centroids with shape (N, 3),
            where each row is (z, y, x) in voxel coordinates.
    """

    def __init__(self, masks: list[NDArray], scale: float, zscale: float) -> None:
        """Initializes the Centroids object from binary masks.

        Args:
            masks (list[NDArray]): List of 3D binary masks (boolean or 0/1 arrays).
                Each mask represents a segmented cell. The shape is (Z, Y, X).
        """
        centroids = []
        for mask in masks:
            coords = np.argwhere(mask)  # voxel coords as (z,y,x)
            if coords.size == 0:
                continue
            centroid = coords.mean(axis=0)
            centroids.append(centroid)
        self.centroids = np.array(centroids, dtype=np.float64)
        self.scale = scale
        self.zscale = zscale

    def compute(self) -> dict[str, Any]:
        """Computes the average pairwise centroid distance in physical units.

        Args:
            scale (float): Scaling factor for X and Y dimensions (microns per pixel).
            zscale (float): Scaling factor for Z dimension (microns per slice).

        Returns:
            float: The average Euclidean distance between centroids in microns.
        """
        scaled = self.centroids.copy()
        scaled[:, 1] *= self.scale  # y
        scaled[:, 2] *= self.scale  # x
        scaled[:, 0] *= self.zscale  # z

        dists = pdist(scaled)
        avg_dist = dists.mean()

        return {"average_distance": avg_dist}
