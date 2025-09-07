from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull


@dataclass
class AnalysisResult:
    """Holds results of full cell analysis including convex hulls and derived metrics.

    Attributes:
        convex_vertices (list[np.ndarray]): List of arrays containing indices of
            convex hull vertices for each cell. Each array corresponds to one cell.
        convex_volumes (np.ndarray): Array of convex hull volumes for each cell
            in physical units (scaled by `voxscale`).
        cell_complexities (np.ndarray): Array of complexity measures for each cell,
            defined as `convex_volume / cell_volume`.
        max_cell_volume (np.float64): Maximum cell volume among all cells, in physical units.
    """

    convex_vertices: list[NDArray]
    convex_volumes: NDArray
    cell_complexities: NDArray
    max_cell_volume: np.float64


class FullCellAnalysis:
    """Compute full cell volumes, convex hulls, and complexity metrics.

    This class takes a list of 3D binary masks representing segmented cells
    and calculates for each cell:
        1. The raw voxel-based volume scaled to physical units.
        2. The convex hull volume using `scipy.spatial.ConvexHull`.
        3. The complexity of the cell defined as `convex_volume / cell_volume`.
        4. Maximum cell volume across all cells.
        5. The indices of convex hull vertices for further visualization or analysis.

    Attributes:
        masks (list[np.ndarray]): List of 3D binary masks for each segmented cell.
        voxscale (float): Scaling factor to convert voxel counts/volume into physical units (e.g., µm³ per voxel).
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
        """Compute convex hulls, cell volumes, complexities, and maximum cell volume.

        For each cell mask:
            1. Count the number of voxels and scale by `voxscale` to get `cell_volume`.
            2. Compute the convex hull of voxel coordinates.
            3. Compute convex hull volume scaled by `voxscale`.
            4. Store convex hull vertex indices for visualization.
            5. Calculate cell complexity as `convex_volume / cell_volume`.

        Returns:
            AnalysisResult: Dataclass containing:
                - convex_vertices: vertex indices for each cell's convex hull
                - convex_volumes: array of convex hull volumes
                - max_cell_volume: maximum voxel volume among all cells
                - cell_complexities: array of complexity values
        """
        voxel_counts = np.array([mask.sum() for mask in self.masks])
        cell_volumes = voxel_counts * self.voxscale
        max_cell_volume = cell_volumes.max()

        convex_volumes = np.zeros(len(self.masks), dtype=np.float64)
        convex_vertices = []
        for i, mask in enumerate(self.masks):
            coords = np.argwhere(mask)  # (z, y, x)
            hull = ConvexHull(coords.astype(np.float64))
            convex_volumes[i] = hull.volume * self.voxscale
            convex_vertices.append(hull.vertices)

        complexities = np.zeros_like(cell_volumes, dtype=np.float64)
        valid = cell_volumes > 0
        complexities[valid] = convex_volumes[valid] / cell_volumes[valid]

        return AnalysisResult(
            convex_vertices=convex_vertices,
            convex_volumes=convex_volumes,
            max_cell_volume=max_cell_volume,
            cell_complexities=complexities,
        )
