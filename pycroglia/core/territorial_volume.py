from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from pycroglia.core.labeled_cells import LabeledCells


class TerritorialVolume:
    """Computes convex hull–based territorial volumes for labeled cells.

    This class calculates the convex volume of each labeled cell in a 3D
    image by computing the convex hull of its voxel coordinates. The
    volume is scaled by the voxel size to obtain physical units.

    Attributes:
        img (NDArray): The input 3D image or volume where cells are labeled.
        cells (LabeledCells): Labeled cells object that provides access
            to voxel indices of each cell.
        voxscale (float): Scaling factor that converts voxel units into
            physical volume units (e.g., µm³).
    """

    def __init__(self, img: NDArray, cells: LabeledCells, voxscale: float) -> None:
        """Initializes the TerritorialVolume object.

        Args:
            img (NDArray): 3D input image/volume with labeled cells.
            cells (LabeledCells): Object providing access to individual
                labeled cells.
            voxscale (float): Conversion factor from voxel volume to
                real-world volume units.
        """
        self.cells = cells
        self.img = img
        self.voxscale = voxscale

    def compute(self) -> NDArray:
        num_of_cells = self.cells.len()
        convex_volume = np.zeros((num_of_cells, 1))
        for i in range(0, num_of_cells):
            # CHECK(jab227): is the index order correct?
            z, y, x = np.unravel_index(self.cells.get_cell(i), self.img.shape)
            obj = np.array([y, x, z])
            hull = ConvexHull(obj)
            convex_volume[i, :] = hull.volume * self.voxscale
        return convex_volume


@dataclass
class TerritorialVolumeMetrics:
    """Holds summary metrics of territorial volume analysis.

    Attributes:
        total_volume_covered (np.float64): Total convex volume occupied
            by all labeled cells.
        image_cube_volume (np.float64): Volume of the entire image cube
            in physical units.
        empty_volume (np.float64): Remaining unoccupied volume.
        covered_percentage (np.float64): Percentage of image cube
            occupied by labeled cells.
    """

    total_volume_covered: np.float64
    image_cube_volume: np.float64
    empty_volume: np.float64
    covered_percentage: np.float64


def compute_metrics(
    convex_volume: NDArray, voxscale: float, img_size: tuple[int, int], zplanes: int
) -> TerritorialVolumeMetrics:
    """Computes global volume coverage metrics from convex hull volumes.

    Args:
        convex_volume (NDArray): Array of per-cell convex hull volumes
            in physical units (output of :meth:`TerritorialVolume.compute`).
        voxscale (float): Scaling factor for voxel volumes (µm³ per voxel).
        img_size (tuple[int, int]): Image dimensions in (x, y).
        zplanes (int): Number of planes along the z-dimension.

    Returns:
        TerritorialVolumeMetrics: A dataclass containing total covered
        volume, image cube volume, empty volume, and percentage coverage.
    """
    x, y = img_size
    total_volume_covered = np.sum(convex_volume)
    image_cube_volume: float = np.float64((x * y * zplanes) * voxscale)
    empty_volume = image_cube_volume - total_volume_covered
    covered_percentage = (total_volume_covered / image_cube_volume) * 100
    return TerritorialVolumeMetrics(
        total_volume_covered=total_volume_covered,
        image_cube_volume=image_cube_volume,
        empty_volume=empty_volume,
        covered_percentage=covered_percentage,
    )
