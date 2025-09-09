import numpy as np
from  numpy.typing  import NDArray
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
    def __init__(self, masks: list[NDArray]) -> None:
        """Initializes the Centroids object from binary masks.

        Args:
            masks (list[NDArray]): List of 3D binary masks (boolean or 0/1 arrays).
                Each mask represents a segmented cell. The shape is (Z, Y, X).
        """
        centroids = []
        for mask in masks:
            coords = np.argwhere(mask) # voxel coords as (z,y,x)
            if coords.size == 0:
                continue
            centroid = coords.mean(axis=0)
            centroids.append(centroid)
        self.centroids = np.array(centroids, dtype=np.float64)

    def compute_average_distance(self, scale: float, zscale: float) -> float:
        """Computes the average pairwise centroid distance in physical units.

        Args:
            scale (float): Scaling factor for X and Y dimensions (microns per pixel).
            zscale (float): Scaling factor for Z dimension (microns per slice).

        Returns:
            float: The average Euclidean distance between centroids in microns.
        """
        self.centroids[:, 1] *= scale   # y
        self.centroids[:, 2] *= scale   # x
        self.centroids[:, 0] *= zscale  # z

        dists = pdist(self.centroids)
        avg_dist = dists.mean()

        return avg_dist
