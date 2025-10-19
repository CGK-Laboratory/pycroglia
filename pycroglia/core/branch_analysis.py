from typing import Any
from numpy.typing import NDArray
from pycroglia.core.arclength import arclength
from pycroglia.core.slimskel3d.slimskel3d import slimskel3d
from scipy.ndimage import convolve
from pycroglia.core.reorder import reorder_pixel_list
from pycroglia.core.arclength import arclength
import pycroglia.core.bounding_box as bounding_box
import pycroglia.core.nearest_pixel as nearest_pixel
import pycroglia.core.connection as connection
import numpy as np

def init_kernel() -> NDArray:
    kernel = np.ones((3, 3, 3), dtype=np.int32)
    kernel[1, 1, 1] = 0
    return kernel

KERNEL = init_kernel()

from scipy.io import loadmat
def indices_to_mask(
    cell_indices: np.ndarray, img_shape: tuple[int, int, int]
) -> np.ndarray:
    mask = np.zeros(img_shape, dtype=bool)
    mask.ravel()[cell_indices] = True
    return mask.astype(np.uint8)
    
class BranchAnalysis:
    def __init__(self, cell: NDArray, centroid: NDArray, scale: float, zscale: float, zslices: int)->None:
        self.cell = cell
        self.centroid = centroid
        self.scale = scale
        self.zscale = zscale
        self.zslices = zslices
        

    def compute(self) -> dict[str, Any]:
        whole_skel = slimskel3d(self.cell, 100)
        mat = loadmat("whole_skel.mat")
        whole_skel_matlab = mat["WholeSkel"]
        whole_skel_matlab = np.transpose(whole_skel_matlab, (2, 1, 0))
        # --- Basic sanity checks
        assert whole_skel.shape == whole_skel_matlab.shape, (
            f"Shape mismatch: Python {whole_skel.shape} vs MATLAB {whole_skel_matlab.shape}"
        )

        # --- Normalize dtype and binary range
        whole_skel_py = whole_skel.astype(bool)
        whole_skel_mat = whole_skel_matlab.astype(bool)
        diff = whole_skel_py ^ whole_skel_mat
        print("Mismatched voxels:", np.count_nonzero(diff))

        coords = np.argwhere(diff)
        print("Differing coordinates:\n", coords)

        zmax, ymax, xmax = whole_skel_py.shape
        for coord in coords:
            z,y,x = coord[0], coord[1], coord[2]
            print(f"Voxel ({z},{y},{x}) -> "
                  f"Python={whole_skel_py[z,y,x]}, MATLAB={whole_skel_mat[z,y,x]}")
        coords = np.array([[6,603,751],[6,604,752],[36,648,765],[37,648,766],[37,649,764],[38,650,765]])
        for c in coords:
            z,y,x = c
            print(f"\nVoxel {tuple(c)}")
            print("MATLAB:")
            print(whole_skel_mat[z-1:z+2,y-1:y+2,x-1:x+2].astype(int))
            print("PYTHON:")
            print(whole_skel_py[z-1:z+2,y-1:y+2,x-1:x+2].astype(int))
                
        # --- Numeric equivalence check
        np.testing.assert_array_equal(
            whole_skel_py,
            whole_skel_mat,
            err_msg="Python and MATLAB skeletons differ voxel-by-voxel."
        )
        print("Skeletons are identical voxel-for-voxel.")
        print(f"whole_skel.dtype={whole_skel.dtype}")
        bounding_box_result = bounding_box.compute(whole_skel)
        bounded_skel = bounding_box_result.bounded_img
        print(f"bounding box: {bounded_skel.shape}")
        print(f"bounding box: {bounded_skel.dtype}")        
        left, right, bottom, top = bounding_box_result.left, bounding_box_result.right, bounding_box_result.bottom, bounding_box_result.top        
        i2 = np.floor(self.centroid).astype(int)
        print(f"i2: {i2}")
        print(f"i2 tuple: {(i2[0], i2[1], i2[2])}")
        closest_point = nearest_pixel.compute(whole_skel, (i2[0], i2[1], i2[2]), self.scale)
        print(f"closest: {closest_point}")
        i2 = np.array([closest_point.z  , closest_point.y, closest_point.x])
        i2_local = i2 - np.array([0, bottom, left])
        
        endpoints = (convolve(bounded_skel, KERNEL, mode="constant") == 1) & bounded_skel
        endpoints_list = np.argwhere(endpoints==1)
        n_endpoints = endpoints_list.shape[0]
        print(f'n_endpoints: {n_endpoints}')

        
        masklist = np.zeros((*bounded_skel.shape, n_endpoints), dtype=bool)
        arclength_of_each_branch = np.zeros(n_endpoints, dtype=float)

        for j, i1 in enumerate(endpoints_list):
            # Connect current endpoint to centroid
            start = connection.Point(z=i1[0], y=i1[1], x=i1[2])
            print(f"start={start}")
            end = connection.Point(z=i2_local[0], y=i2_local[1], x=i2_local[2])
            print(f"end={end}")            
            path_coords = connection.connect_points_along_path(bounded_skel, start, end)
            masklist[..., j] = path_coords
            print(f"masklist: {path_coords.shape}")

            # Reorder pixels by connectivity (stub; implement graph-ordering if needed)
            pxlist = np.flatnonzero(masklist[..., j] == 1)
            distpoint = reorder_pixel_list(pxlist, bounded_skel.shape, i1, i2_local)
            
            # Convert voxel coordinates to microns
            distpoint = distpoint.astype(float)
            distpoint[:, 0] *= self.zscale       # z
            distpoint[:, 1] *= self.scale # y
            distpoint[:, 2] *= self.scale # x

            # Compute arc length (microns)
            print(distpoint.shape)
            arclen_result = arclength(distpoint)
            arclength_of_each_branch[j] = arclen_result.arclength

        arclength_of_each_branch = arclength_of_each_branch[arclength_of_each_branch > 0.0]
        # Summary statistics
        if n_endpoints > 0:
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
            temp_endpoints = (convolve(temp.astype(int), KERNEL, mode="constant") == 1) & temp
            branch_points[..., kk] = temp_endpoints
        quat_endpts = (convolve(quaternary.astype(int), KERNEL, mode="constant") == 1) & quaternary
        quat_brpts = quat_endpts - endpoints
        fullrep = fullmask.copy()
        fullrep[fullrep < 4] = 0
        qbpts = fullrep + quat_brpts.astype(int)
        qbpts1 = convolve(qbpts, np.ones((3, 3, 3), dtype=int), mode="constant")
        branch_points[..., 0] = (quat_brpts & (qbpts1 >= 5))
        allbranch = np.sum(branch_points, axis=3)
        branch_points = np.argwhere(allbranch == 1)
        num_branchpoints = branch_points.shape[0]

        return {
            "endpoints": endpoints,
            "num_branchpoints": num_branchpoints,
            "max_branch_length": max_branch_length,
            "min_branch_length": min_branch_length,
            "avg_branch_length": avg_branch_length,
            "branch_points": branch_points,
        }
