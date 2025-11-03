from typing import Any
import numpy as np
from scipy.spatial import ConvexHull


KEY_CELL_VOLUMES = "cell_volumes"
KEY_CONVEX_SIMPLICES = "convex_simplices"
KEY_CONVEX_VERTICES = "convex_vertices"
KEY_CONVEX_VOLUMES = "convex_volumes"
KEY_CONVEX_MAX_CELL_VOLUME = "max_cell_volume"
KEY_CELL_COMPLEXITIES = "cell_complexities"


class FullCellAnalysis:
    """Compute convex hulls, volumes, and complexity metrics for segmented cells.

    Given a list of 3D binary masks (segmented cells), this class computes:

        1. Raw voxel-based cell volume (scaled to physical units by ``voxscale``).
        2. Convex hull of voxel coordinates and its volume.
        3. Cell complexity: ratio of convex hull volume to cell volume.
        4. Maximum cell volume across all masks.
        5. Convex hull vertices and simplices for visualization.

    Attributes:
        masks (list[np.ndarray]):
            List of 3D binary arrays where ``True`` or ``1`` indicates
            cell voxels.
        voxscale (float):
            Scaling factor to convert voxel counts/volumes into
            physical units (e.g., µm³ per voxel).
    """

    def __init__(self, masks: list[np.ndarray], voxscale: float) -> None:
        """
        Args:
            masks (list[np.ndarray]): List of 3D binary masks, one per cell.
            voxscale (float): Factor to convert voxel volume to physical units.
        """
        self.masks = masks
        self.voxscale = voxscale

    def compute(self) -> dict[str, Any]:
        """Perform convex hull and complexity analysis for all cells.

        For each cell:
            - Counts voxels and converts to volume using ``voxscale``.
            - Builds a convex hull from voxel coordinates.
            - Computes convex hull volume and stores vertices/simplices.
            - Computes complexity as ``convex_volume / cell_volume``.

        Returns:
            Dict[str, Any]:
                Dict containing convex hulls, volumes, complexities, cell volumes
                and maximum cell volume.
        """
        voxel_counts = np.array([mask.sum() for mask in self.masks])
        cell_volumes = voxel_counts * self.voxscale
        max_cell_volume = cell_volumes.max()

        convex_volumes = np.zeros(len(self.masks), dtype=np.float64)
        convex_vertices = []
        convex_simplices = []
        for i, mask in enumerate(self.masks):
            coords = np.argwhere(mask)  # (z, y, x)
            hull = ConvexHull(coords.astype(np.float64))
            convex_volumes[i] = hull.volume * self.voxscale
            convex_vertices.append(hull.vertices)
            convex_simplices.append(hull.simplices)

        complexities = np.zeros_like(cell_volumes, dtype=np.float64)
        valid = cell_volumes > 0
        complexities[valid] = convex_volumes[valid] / cell_volumes[valid]

        return {
            KEY_CELL_VOLUMES: cell_volumes,
            KEY_CONVEX_SIMPLICES: convex_simplices,
            KEY_CONVEX_VERTICES: convex_vertices,
            KEY_CONVEX_VOLUMES: convex_volumes,
            KEY_CONVEX_MAX_CELL_VOLUME: max_cell_volume,
            KEY_CELL_COMPLEXITIES: complexities,
        }
