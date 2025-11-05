import numpy as np
from pycroglia.core.full_cell_analysis import FullCellAnalysis


def test_full_cell_analysis():
    """Test FullCellAnalysis computes convex volumes and complexities.

    This test builds two synthetic 3×3×3 masks, each containing four voxels
    arranged in an L-shape. These small shapes have simple convex hulls, making
    it straightforward to verify correctness of the analysis.

    Verifications performed:
        - Convex hull simplices and vertex indices are consistent with the voxel geometry.
        - Convex hull volumes are computed in physical units by scaling with `voxscale`.
        - Cell complexities are correctly calculated as `convex_volume / cell_volume`.
        - The maximum cell volume is determined correctly from voxel counts.

    Asserts:
        - `cell_volumes` match expected floating-point values.
        - `convex_vertices` match the expected vertex indices for each mask.
        - `convex_volumes` match expected floating-point values.
        - `cell_complexities` match expected floating-point values.
        - `max_cell_volume` matches the expected maximum cell volume.
    """
    mask1 = np.zeros((3, 3, 3), dtype=np.uint8)
    mask1[0, 0, 0] = 1
    mask1[0, 1, 0] = 1
    mask1[1, 0, 0] = 1
    mask1[0, 0, 1] = 1

    mask2 = np.zeros((3, 3, 3), dtype=np.uint8)
    mask2[2, 2, 2] = 1
    mask2[2, 1, 2] = 1
    mask2[1, 2, 2] = 1
    mask2[2, 2, 1] = 1

    masks = [mask1, mask2]

    voxscale = 0.1
    fca = FullCellAnalysis(masks, voxscale)
    result = fca.compute()
    expected_cell_volumes = np.array([0.4, 0.4])
    expected_convex_vertices = [
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([0, 1, 2, 3], dtype=np.int32),
    ]
    expected_convex_volumes = np.array([0.01666667, 0.01666667])
    expected_cell_complexities = np.array([0.04166667, 0.04166667])
    expected_max_cell_volume = np.float64(0.4)

    np.testing.assert_allclose(result["cell_volumes"], expected_cell_volumes)
    np.testing.assert_allclose(result["convex_vertices"], expected_convex_vertices)
    assert np.allclose(result["convex_volumes"], expected_convex_volumes)
    np.testing.assert_allclose(result["cell_complexities"], expected_cell_complexities)
    assert np.isclose(expected_max_cell_volume, result["max_cell_volume"])
