from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

KEY_CENTROIDS = "centroids"
KEY_AVG_CENTROIDS_DISTANCE = "average_distance"


class Centroids:
    """Compute centroids of 3D cell masks and their mean spatial separation.

    This class extracts the geometric centroids of multiple 3D binary masks,
    representing individual segmented cells, and computes the average pairwise
    Euclidean distance between all centroids in physical units (micrometers).

    The computation accounts for anisotropic voxel scaling along the Z and XY
    axes, allowing accurate physical distance measurements in microscopy data.

    Attributes:
        centroids (NDArray[np.float64]):
            Array of shape (N, 3) containing the centroids of non-empty masks,
            where each row represents a centroid in (z, y, x) voxel coordinates.
        scale (float):
            Pixel size in the XY plane, expressed in micrometers per voxel.
        zscale (float):
            Z-step size (slice spacing) in micrometers per voxel.
    """

    def __init__(self, masks: list[NDArray], scale: float, zscale: float) -> None:
        """Initialize the Centroids object from a list of binary 3D masks.

        Args:
            masks (list[NDArray]):
                List of 3D binary arrays of shape (Z, Y, X), where True (or 1)
                indicates voxels belonging to a segmented cell.
            scale (float):
                Pixel size in the XY plane, in micrometers per voxel.
            zscale (float):
                Z-step size (slice spacing), in micrometers per voxel.

        Notes:
            - Empty masks (with no True voxels) are ignored.
            - Centroids are computed as the mean of voxel coordinates
              (`np.argwhere(mask).mean(axis=0)`), in voxel units.
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
        """Compute the mean pairwise centroid distance in micrometers.

        The method scales voxel coordinates to physical space using the
        provided pixel and z-step sizes, then computes all pairwise
        Euclidean distances between centroids and reports the mean.

        Returns:
            dict[str, Any]: Dictionary containing:
                - **average_distance (float)**: Mean pairwise distance between
                  all centroids, expressed in micrometers.

        Raises:
            ValueError: If fewer than two centroids are available to compute distances.
        """
        scaled = self.centroids.copy()
        scaled[:, 1] *= self.scale  # y
        scaled[:, 2] *= self.scale  # x
        scaled[:, 0] *= self.zscale  # z

        dists = pdist(scaled)
        avg_dist = dists.mean()

        return {KEY_AVG_CENTROIDS_DISTANCE: avg_dist, KEY_CENTROIDS: self.centroids}
