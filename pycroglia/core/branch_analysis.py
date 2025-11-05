from typing import Any
from numpy.typing import NDArray
from pycroglia.core.slimskel3d.slimskel3d import slimskel3d
from scipy.ndimage import convolve
from pycroglia.core.reorder import reorder_pixel_list
from pycroglia.core.arclength import arclength
import pycroglia.core.bounding_box as bounding_box
import pycroglia.core.nearest_pixel as nearest_pixel
import pycroglia.core.connection as connection
import numpy as np

KEY_ENDPOINTS = "endpoints"
KEY_NUM_BRANCHPOINTS = "num_branchpoints"
KEY_MAX_BRANCH_LENGTH = "max_branch_length"
KEY_MIN_BRANCH_LENGTH = "min_branch_length"
KEY_AVG_BRANCH_LENGTH = "avg_branch_length"
KEY_BRANCH_POINTS = "branch_points"
KEY_ALLBRANCH = "allbranch"
KEY_FULLMASKS = "fullmasks"


class EmptySkeleton(Exception):
    """Raised when a skeleton or image has no nonzero voxels (i.e., is empty)."""

    def __init__(self):
        super().__init__("Tried to create an empty skeleton")


def init_kernel() -> NDArray:
    """Initialize a 3×3×3 26-connected neighborhood kernel.

    This function creates a 3D binary convolution kernel used for
    neighborhood analysis in volumetric skeletonization and branch
    detection. The kernel represents a **26-connected** neighborhood,
    where all voxels in a 3×3×3 cube are considered neighbors except
    the central voxel itself.

    Returns:
        NDArray: A 3×3×3 NumPy array of type ``int32`` with ones in all
        positions except the center, which is zero.
    """
    kernel = np.ones((3, 3, 3), dtype=np.int32)
    kernel[1, 1, 1] = 0
    return kernel


# 26-connected 3×3×3 kernel used for neighborhood analysis
KERNEL = init_kernel()


def get_empty_branch_analysis() -> dict[str, Any]:
    return {
        "endpoints": [],
        "num_branchpoints": 0,
        "max_branch_length": 0.0,
        "min_branch_length": 0.0,
        "avg_branch_length": 0.0,
        "branch_points": 0.0,
    }


class BranchAnalysis:
    """Analyze the morphological properties of 3D skeleton branches.

    This class performs quantitative analysis on a 3D skeletonized cell mask,
    identifying endpoints, branch points, and computing arc lengths for each
    branch. It replicates the behavior of the MATLAB `BranchAnalysis` routine,
    using connected-path reconstruction and distance-based reordering of voxels
    from endpoints toward the cell centroid.

    Attributes:
        cell (NDArray):
            3D binary mask of the segmented cell volume (Z, Y, X).
        centroid (NDArray):
            (3,) array with the centroid coordinates `(z, y, x)` in voxel units.
        scale (float):
            XY pixel size (µm per voxel).
        zscale (float):
            Z pixel size (µm per slice).
        zslices (int):
            Number of Z-slices in the image stack.
    """

    def __init__(
        self,
        cell: NDArray,
        centroid: NDArray,
        scale: float,
        zscale: float,
        zslices: int,
    ) -> None:
        """Initialize the BranchAnalysis instance.

        Args:
            cell (NDArray): 3D binary cell mask of shape `(Z, Y, X)`.
            centroid (NDArray): (3,) array with the centroid `(z, y, x)`.
            scale (float): XY pixel size in microns.
            zscale (float): Z pixel size in microns.
            zslices (int): Total number of Z slices in the volume.
        """
        self.cell = cell
        self.centroid = centroid
        self.scale = scale
        self.zscale = zscale
        self.zslices = zslices

    def compute(self) -> dict[str, Any]:
        """Perform 3D branch analysis on the input cell skeleton.

        This method:
          1. Skeletonizes the cell volume using `slimskel3d`.
          2. Computes a bounding box around the skeleton.
          3. Identifies all endpoints via 3×3×3 convolution.
          4. Connects each endpoint to the centroid using
             `connect_points_along_path` (26-connected BFS).
          5. Reorders the resulting voxel list using
             `reorder_pixel_list` for spatial continuity.
          6. Calculates each branch’s physical arc length in microns.
          7. Aggregates results to find branch point counts and
             connectivity classifications (primary to quaternary).

        Returns:
            dict[str, Any]: A dictionary containing:
                - **endpoints (NDArray)**: 3D boolean mask of detected endpoints.
                - **num_branchpoints (int)**: Number of detected branch points.
                - **max_branch_length (float)**: Maximum branch length (µm).
                - **min_branch_length (float)**: Minimum branch length (µm).
                - **avg_branch_length (float)**: Mean branch length (µm).
                - **branch_points (NDArray)**: (N, 3) array of branch-point coordinates `(z, y, x)`.
        """
        # TODO - Check hardcoded value
        whole_skel = slimskel3d(self.cell, 100)
        if np.all(whole_skel == 0):
            raise EmptySkeleton
        bounding_box_result = bounding_box.compute(whole_skel)
        bounded_skel = bounding_box_result.bounded_img
        left, bottom = bounding_box_result.left, bounding_box_result.bottom
        i2 = np.floor(self.centroid).astype(int)
        closest_point = nearest_pixel.compute(
            whole_skel, (i2[0], i2[1], i2[2]), self.scale
        )
        i2 = np.array([closest_point.z, closest_point.y, closest_point.x])
        i2_local = i2 - np.array([0, bottom, left])

        endpoints = (
            convolve(bounded_skel, KERNEL, mode="constant") == 1
        ) & bounded_skel
        endpoints_list = np.argwhere(endpoints == 1)
        n_endpoints = endpoints_list.shape[0]

        masklist = np.zeros((*bounded_skel.shape, n_endpoints), dtype=bool)
        arclength_of_each_branch = np.zeros(n_endpoints, dtype=float)
        for j, i1 in enumerate(endpoints_list):
            # Connect current endpoint to centroid
            start = connection.Point(z=i1[0], y=i1[1], x=i1[2])
            end = connection.Point(z=i2_local[0], y=i2_local[1], x=i2_local[2])
            path_coords = connection.connect_points_along_path(bounded_skel, start, end)
            masklist[..., j] = path_coords

            # Reorder pixels by connectivity (stub; implement graph-ordering if needed)
            pxlist = np.flatnonzero(masklist[..., j] == 1)
            distpoint = reorder_pixel_list(pxlist, bounded_skel.shape, i1, i2_local)

            # Convert voxel coordinates to microns
            distpoint = distpoint.astype(float)
            distpoint[:, 0] *= self.zscale  # z
            distpoint[:, 1] *= self.scale  # y
            distpoint[:, 2] *= self.scale  # x

            # Compute arc length (microns)
            arclen_result = arclength(distpoint)
            arclength_of_each_branch[j] = arclen_result.arclength

        arclength_of_each_branch = arclength_of_each_branch[
            arclength_of_each_branch > 0.0
        ]
        # Summary statistics
        if len(arclength_of_each_branch) > 0:
            max_branch_length = float(np.max(arclength_of_each_branch))
            min_branch_length = float(np.min(arclength_of_each_branch))
            avg_branch_length = float(np.mean(arclength_of_each_branch))
        else:
            max_branch_length = min_branch_length = avg_branch_length = 0.0

        # Combine all branch masks
        fullmask = np.sum(masklist.astype(int), axis=3)
        fullmask[fullmask > 3] = 4  # cap at quaternary connectivity
        quaternary = fullmask == 1

        branch_points = np.zeros((*bounded_skel.shape, 4), dtype=bool)
        for kk in range(1, 4):  # 1:3 inclusive
            temp = fullmask > kk
            temp_endpoints = (
                convolve(temp.astype(int), KERNEL, mode="constant") == 1
            ) & temp
            branch_points[..., kk] = temp_endpoints
        quat_endpts = (
            convolve(quaternary.astype(int), KERNEL, mode="constant") == 1
        ) & quaternary
        quat_brpts = quat_endpts - endpoints
        fullrep = fullmask.copy()
        fullrep[fullrep < 4] = 0
        qbpts = fullrep + quat_brpts.astype(int)
        qbpts1 = convolve(qbpts, np.ones((3, 3, 3), dtype=int), mode="constant")
        branch_points[..., 0] = quat_brpts & (qbpts1 >= 5)
        allbranch = np.sum(branch_points, axis=3)
        branch_points = np.argwhere(allbranch == 1)
        num_branchpoints = branch_points.shape[0]

        return {
            KEY_ALLBRANCH: allbranch,
            KEY_FULLMASKS: [fullmask],
            KEY_ENDPOINTS: endpoints,
            KEY_NUM_BRANCHPOINTS: num_branchpoints,
            KEY_MAX_BRANCH_LENGTH: max_branch_length,
            KEY_MIN_BRANCH_LENGTH: min_branch_length,
            KEY_AVG_BRANCH_LENGTH: avg_branch_length,
            KEY_BRANCH_POINTS: branch_points,
        }
