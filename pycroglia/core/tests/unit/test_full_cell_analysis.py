import numpy as np

from pycroglia.core.full_cell_analysis import FullCellAnalysis, AnalysisResult


def test_full_cell_analysis():
    """Test FullCellAnalysis computes convex volumes and complexities.

    This test uses two simple 3x3x3 masks with four voxels each to verify that:
        - The convex hull vertices are correctly identified.
        - The convex hull volumes are computed in physical units using voxscale.
        - Cell complexities (convex volume / cell volume) are correctly calculated.
        - Maximum cell volume is correctly determined from the voxel counts.

    Asserts:
        - Convex hull vertices match expected indices for each cell.
        - Convex hull volumes match expected floating-point values.
        - Cell complexities match expected floating-point values.
        - Maximum cell volume matches expected value.
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
    expected = AnalysisResult(
        convex_simplices = [
            np.array([[2, 3, 0],
                      [1, 3, 0],
                      [1, 2, 0],
                      [1, 2, 3]], dtype=np.int32),
            np.array([[2, 1, 0],
                      [3, 1, 0],
                      [3, 2, 0],
                      [3, 2, 1]], dtype=np.int32)
        ],
        convex_vertices=[
            np.array([0, 1, 2, 3], dtype=np.int32),
            np.array([0, 1, 2, 3], dtype=np.int32),
        ],
        convex_volumes=np.array([0.01666667, 0.01666667]),
        cell_complexities=np.array([0.04166667, 0.04166667]),
        max_cell_volume=np.float64(0.4),
    )
    np.testing.assert_allclose(result.convex_vertices, expected.convex_vertices)
    assert np.allclose(result.convex_volumes, expected.convex_volumes)    
    np.testing.assert_allclose(result.cell_complexities, expected.cell_complexities)
    assert np.isclose(expected.max_cell_volume, result.max_cell_volume)
