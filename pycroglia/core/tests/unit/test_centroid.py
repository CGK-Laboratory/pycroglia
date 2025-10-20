import numpy as np

from pycroglia.core.centroid import Centroids


def test_get_centroids():
    """Test Centroids correctly computes centroids from binary masks.

    Asserts:
        The computed centroids match the expected voxel coordinates
        for two simple 3D masks.
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
    zscale = 0.2
    scale = 0.1
    c = Centroids(masks, scale, zscale)
    expected = np.array([np.array([0.25, 0.25, 0.25]), np.array([1.75, 1.75, 1.75])])
    np.testing.assert_allclose(expected, c.centroids)


def test_compute_average_centroid_distance():
    """Test compute_average_distance returns correct scaled distance.

    Asserts:
        The average pairwise distance between two centroids is correctly
        computed in physical units, given voxel scales for XY and Z.
    """

    zscale = 0.2
    scale = 0.1
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

    c = Centroids(masks, scale, zscale)
    result = c.compute()

    np.testing.assert_approx_equal(result["average_distance"], 0.3674234614174768)
