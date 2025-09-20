from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull


@dataclass
class AnalysisResult:
    """Results of full cell analysis including convex hulls and derived metrics.

    Attributes:
        convex_simplices (list[NDArray]):
            List of arrays of simplices (faces) for each convex hull,
            useful for 3D visualization or mesh reconstruction.
        convex_vertices (list[NDArray]):
            List of arrays containing vertex indices of convex hulls.
            Each entry corresponds to one cell.
        convex_volumes (NDArray):
            Array of convex hull volumes for each cell, scaled by ``voxscale``.
        cell_volumes: NDArray:
            Array of with the approximated volume of each cell.
        cell_complexities (NDArray):
            Array of complexity values for each cell, computed as
            ``convex_volume / cell_volume``.
        max_cell_volume (np.float64):
            Maximum voxel-based cell volume across all cells (scaled).
    """

    convex_simplices: list[NDArray]
    convex_vertices: list[NDArray]
    cell_volumes: NDArray
    convex_volumes: NDArray
    cell_complexities: NDArray
    max_cell_volume: np.float64


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

    def compute(self) -> AnalysisResult:
        """Perform convex hull and complexity analysis for all cells.

        For each cell:
            - Counts voxels and converts to volume using ``voxscale``.
            - Builds a convex hull from voxel coordinates.
            - Computes convex hull volume and stores vertices/simplices.
            - Computes complexity as ``convex_volume / cell_volume``.

        Returns:
            AnalysisResult:
                Dataclass containing convex hulls, volumes, complexities, cell volumes
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

        return AnalysisResult(
            cell_volumes=cell_volumes,
            convex_simplices=convex_simplices,
            convex_vertices=convex_vertices,
            convex_volumes=convex_volumes,
            max_cell_volume=max_cell_volume,
            cell_complexities=complexities,
        )
