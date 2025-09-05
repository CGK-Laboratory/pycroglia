import numpy as np
from pycroglia.core.territorial_volume import TerritorialVolume, compute_metrics, TerritorialVolumeMetrics

def test_compute_territorial_volume():
    """Test TerritorialVolume.compute returns correct convex volumes for small masks.

    Asserts:
        The computed convex hull volumes match expected reference values.
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
    tv = TerritorialVolume(masks, voxscale)
    result = tv.compute()
    assert np.allclose(np.array([0.01666667, 0.01666667]), result)


def test_compute_metrics():
    """Test compute_metrics returns correct global volume metrics.

    Asserts:
        All TerritorialVolumeMetrics fields (total_volume_covered, image_cube_volume,
        empty_volume, covered_percentage) match expected reference values.
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
    zplanes = 3
    tv = TerritorialVolume(masks, voxscale)
    convex_vol = tv.compute()
    result = compute_metrics(convex_vol, voxscale, (3,3,3), zplanes)
    expected = TerritorialVolumeMetrics(total_volume_covered=np.float64(0.03333333333333333), image_cube_volume=np.float64(2.7), empty_volume=np.float64(2.666666666666667), covered_percentage=np.float64(1.2345679012345678))
    np.testing.assert_allclose(result.total_volume_covered, expected.total_volume_covered, rtol=1e-9)
    np.testing.assert_allclose(result.image_cube_volume, expected.image_cube_volume, rtol=1e-5)
    np.testing.assert_allclose(result.empty_volume, expected.empty_volume, rtol=1e-5)
    np.testing.assert_allclose(result.covered_percentage, expected.covered_percentage, rtol=1e-5)
    
