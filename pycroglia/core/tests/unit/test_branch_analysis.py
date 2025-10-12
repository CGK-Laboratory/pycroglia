import numpy as np
from pycroglia.core.branch_analysis import BranchAnalysis
from pathlib import Path
from scipy.io import loadmat

TEST_DIR = Path(__file__).parent  # folder where this test lives
FILES_DIR = TEST_DIR / "files"  # adjust if files/ is elsewhere

def indices_to_mask(
    cell_indices: np.ndarray, img_shape: tuple[int, int, int]
) -> np.ndarray:
    mask = np.zeros(img_shape, dtype=bool)
    mask.ravel()[cell_indices] = True
    return mask.astype(np.uint8)

    
def test_branch_analysis_equivalence():
    """Compare BranchAnalysis results numerically to MATLAB output expectations."""
    mat = loadmat(FILES_DIR /"branch_analysis_results.mat", squeeze_me=True, struct_as_record=False)
    mat_result = mat["result"]

    # Extract MATLAB results
    matlab_num_branchpoints = int(mat_result.num_branchpoints)
    matlab_max_branch_length = float(mat_result.max_branch_length)
    matlab_min_branch_length = float(mat_result.min_branch_length)
    matlab_avg_branch_length = float(mat_result.avg_branch_length)

    matlab_branch_points = np.array(mat_result.branch_points)
    matlab_endpoints = np.array(mat_result.endpoints, dtype=np.uint8)
    matlab_endpoints = np.transpose(matlab_endpoints, (2, 1, 0))  # Z, Y, X
    zslices = 39    
    data = loadmat(FILES_DIR /"cell_test.mat", squeeze_me=True)
    indices = data["data"].ravel().astype(int) - 1    
    mask = indices_to_mask(indices, (zslices, 1024, 1024)) 
    cell = mask
    centroid = np.array([31.2319,  787.6710, 637.9670], dtype=float) - 1.0
    scale = 1.0
    zscale = 1.0

    analyzer = BranchAnalysis(
        cell=cell,
        centroid=centroid,
        scale=scale,
        zscale=zscale,
        zslices=zslices,
    )
    py_result = analyzer.compute()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def plot_endpoints_3d(matlab_endpoints, python_endpoints):
        """Compare MATLAB and Python endpoint voxel positions in 3D."""
        
        # Convert to binary arrays
        m = matlab_endpoints.astype(bool)
        p = python_endpoints.astype(bool)

        # Extract coordinates
        mz, my, mx = np.nonzero(m)
        pz, py, px = np.nonzero(p)

        # Intersection & differences
        match = m & p
        diff_mat = m & ~p
        diff_py = p & ~m

        z_match, y_match, x_match = np.nonzero(match)
        z_mat, y_mat, x_mat = np.nonzero(diff_mat)
        z_py, y_py, x_py = np.nonzero(diff_py)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(x_match, y_match, z_match, c="magenta", s=20, label="Match (both)")
        ax.scatter(x_mat, y_mat, z_mat, c="red", s=40, label="MATLAB-only")
        ax.scatter(x_py, y_py, z_py, c="blue", s=40, label="Python-only")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.set_title("3D Endpoint Comparison (MATLAB vs Python)")
        plt.show()
    plot_endpoints_3d(matlab_endpoints, py_result["endpoints"])
    diff_coords = np.argwhere(matlab_endpoints != py_result["endpoints"])
    print("Different voxels:", diff_coords)
    print("Python True voxels:", np.argwhere(py_result["endpoints"]))
    print("MATLAB True voxels:", np.argwhere(matlab_endpoints))
    print("Shapes:", matlab_endpoints.shape, py_result["endpoints"].shape)
    print("Equal voxels:", np.sum(matlab_endpoints == py_result["endpoints"]), "/", matlab_endpoints.size)
    print("Non-zero in Python:", np.count_nonzero(matlab_endpoints))
    print("Non-zero in MATLAB:", np.count_nonzero(py_result["endpoints"]))
    print("Intersection:", np.count_nonzero((matlab_endpoints > 0) & (py_result["endpoints"] > 0)))
    print("Symmetric difference:", np.count_nonzero(matlab_endpoints != py_result["endpoints"]))
    assert np.array_equal(py_result["endpoints"], matlab_endpoints), "Endpoints mask mismatch"
    assert py_result["num_branchpoints"] == matlab_num_branchpoints, "Branchpoint count mismatch"
    assert np.isclose(py_result["max_branch_length"], matlab_max_branch_length, atol=1e-6), "Max branch length mismatch"
    assert np.isclose(py_result["min_branch_length"], matlab_min_branch_length, atol=1e-6), "Min branch length mismatch"
    assert np.isclose(py_result["avg_branch_length"], matlab_avg_branch_length, atol=1e-6), "Average branch length mismatch"

    assert np.array_equal(py_result["branch_points"], matlab_branch_points), "Branch points mismatch"

