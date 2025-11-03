from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

KEY_CONVEX_VOLUME = "cells_convex_volume"
KEY_TOTAL_VOLUME_COVERED = "total_volume_covered"
KEY_IMAGE_CUBE_VOLUME = "image_cube_volume"
KEY_EMPTY_VOLUME = "empty_volume"
KEY_COVERED_PERCENTAGE = "covered_percentage"


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

    def __init__(self, masks: list[NDArray], voxscale: float, zplanes: int) -> None:
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
        self.zplanes = zplanes

    def compute(self) -> dict[str, Any]:
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
        _, y, x = self.masks[0].shape
        total_volume_covered = np.sum(convex_volume)
        image_cube_volume: float = np.float64((x * y * self.zplanes) * self.voxscale)
        empty_volume = image_cube_volume - total_volume_covered
        covered_percentage = (total_volume_covered / image_cube_volume) * 100
        return {
            KEY_CONVEX_VOLUME: convex_volume,
            KEY_TOTAL_VOLUME_COVERED: total_volume_covered,
            KEY_IMAGE_CUBE_VOLUME: image_cube_volume,
            KEY_EMPTY_VOLUME: empty_volume,
            KEY_COVERED_PERCENTAGE: covered_percentage,
        }
