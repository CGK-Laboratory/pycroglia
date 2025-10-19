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
    mat = loadmat(FILES_DIR / "branch_analysis_results.mat", squeeze_me=True, struct_as_record=False)
    mat_result = mat["result"]

    # Extract MATLAB results
    matlab_num_branchpoints = int(mat_result.num_branchpoints)
    matlab_max_branch_length = float(mat_result.max_branch_length)
    matlab_min_branch_length = float(mat_result.min_branch_length)
    matlab_avg_branch_length = float(mat_result.avg_branch_length)

    matlab_branch_points = np.array(mat_result.branch_points)
    matlab_branch_points = matlab_branch_points[:, [2, 1, 0]] - 1
    matlab_endpoints = np.array(mat_result.endpoints, dtype=np.uint8)
    matlab_endpoints = np.transpose(matlab_endpoints, (2, 1, 0))  # Z, Y, X

    zslices = 39
    data = loadmat(FILES_DIR / "cell_test.mat", squeeze_me=True)
    indices = data["data"].ravel().astype(int) - 1
    mask = indices_to_mask(indices, (zslices, 1024, 1024))
    cell = mask

    centroid = np.array([31.2319, 787.6710, 637.9670], dtype=float) - 1.0
    analyzer = BranchAnalysis(cell=cell, centroid=centroid, scale=1.0, zscale=1.0, zslices=zslices)
    py_result = analyzer.compute()
    python_endpoints = py_result["endpoints"]

    np.testing.assert_array_equal(
        python_endpoints,
        matlab_endpoints,
        err_msg="Endpoint voxel masks differ voxel-by-voxel.",
    )
    assert py_result["num_branchpoints"] == matlab_num_branchpoints, "Branchpoint count mismatch"   
    assert np.array_equal(py_result["branch_points"], matlab_branch_points), "Branch points mismatch"    
    assert np.isclose(py_result["max_branch_length"], matlab_max_branch_length, atol=1e-6), "Max branch length mismatch"
    assert np.isclose(py_result["min_branch_length"], matlab_min_branch_length, atol=1e-6), "Min branch length mismatch"
    assert np.isclose(py_result["avg_branch_length"], matlab_avg_branch_length, atol=1e-6), "Average branch length mismatch"
