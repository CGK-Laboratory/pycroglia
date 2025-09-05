from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull


class TerritorialVolume:
    """Compute convex hull volumes for segmented 3D cells.

    This class takes a list of binary masks, where each mask corresponds
    to a segmented cell within a 3D image. For each mask, it extracts voxel
    coordinates, computes the convex hull enclosing those voxels, and returns
    the hull volume scaled into physical units.

    Attributes:
        masks (list[np.ndarray]): List of 3D boolean arrays. Each array
            represents the voxel mask of one segmented cell. All masks must
            share the same shape corresponding to the original image volume.
        voxscale (float): Scaling factor for converting voxel-based volumes
            into physical units (e.g., µm³ per voxel).
    """
    def __init__(self, masks: list[NDArray], voxscale: float) -> None:
        """Initializes a TerritorialVolume instance.

        Args:
            masks (list[np.ndarray]): List of binary 3D masks, one per cell.
                Each mask should have the same dimensions (Z, Y, X) as the
                source image.
            voxscale (float): Scaling factor for converting voxel volumes into
                physical volume units (e.g., µm³ per voxel).
        """
        self.masks = masks
        self.voxscale = voxscale

    def compute(self) -> NDArray:
        """Computes convex hull volumes for all segmented cells.

        For each mask:
            1. Extract voxel coordinates using `np.argwhere`.
               - Returns indices in (z, y, x) order.
            2. Convert coordinates to float64 for numerical stability.
            3. Construct a convex hull enclosing the voxel cloud with
               `scipy.spatial.ConvexHull`.
            4. Multiply the hull volume by `voxscale` to convert into
               physical volume.

        Returns:
            np.ndarray: A 1D array of shape (n_cells,) containing the convex
            hull volume (float64) of each cell in physical units.
        """
        num_of_cells = len(self.masks)
        convex_volume = np.zeros(num_of_cells)

        for i, mask in enumerate(self.masks):
            # Get coordinates of all voxels in the mask
            coords = np.argwhere(mask)  # shape (n_voxels, 3), each row (z,y,x)

            obj = coords.astype(np.float64)

            # Compute convex hull volume
            hull = ConvexHull(obj)
            convex_volume[i] = hull.volume * self.voxscale

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
    convex_volume: NDArray, voxscale: float, img_size: tuple[int, int, int], zplanes: int
) -> TerritorialVolumeMetrics:
    """Computes global volume coverage metrics from convex hull volumes.

    Args:
        convex_volume (NDArray): Array of per-cell convex hull volumes
            in physical units (output of :meth:`TerritorialVolume.compute`).
        voxscale (float): Scaling factor for voxel volumes (µm³ per voxel).
        img_size (tuple[int, int, int]): Image dimensions in (x, y, z).
        zplanes (int): Number of planes along the z-dimension.

    Returns:
        TerritorialVolumeMetrics: A dataclass containing total covered
        volume, image cube volume, empty volume, and percentage coverage.
    """
    x, y, _ = img_size
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
