import numpy as np
from pycroglia.core.reorder import reorder_pixel_list


def test_simple_line_path():
    """Test reorder_pixel_list correctly orders a straight vertical line.

    Asserts:
        - Voxels are reordered from the endpoint at z=0 to the centroid at z=4.
        - The resulting order matches [[0,2,2], [1,2,2], ..., [4,2,2]].
    """
    shape = (5, 5, 5)
    mask = np.zeros(shape, dtype=bool)
    for i in range(5):
        mask[i, 2, 2] = True

    pixel_indices = np.flatnonzero(mask)
    endpoint = np.array([0, 2, 2])
    centroid = np.array([4, 2, 2])

    ordered = reorder_pixel_list(pixel_indices, shape, endpoint, centroid)

    expected = np.array([[i, 2, 2] for i in range(5)])
    assert np.array_equal(ordered, expected), f"Expected:\n{expected}\nGot:\n{ordered}"


def test_diagonal_path():
    """Test reorder_pixel_list correctly orders a 3D diagonal line.

    Asserts:
        - Voxels are reordered from endpoint (0,0,0) to centroid (4,4,4).
        - The resulting path matches [[0,0,0], [1,1,1], ..., [4,4,4]].
    """
    shape = (5, 5, 5)
    mask = np.zeros(shape, dtype=bool)
    for i in range(5):
        mask[i, i, i] = True

    pixel_indices = np.flatnonzero(mask)
    endpoint = np.array([0, 0, 0])
    centroid = np.array([4, 4, 4])

    ordered = reorder_pixel_list(pixel_indices, shape, endpoint, centroid)

    expected = np.array([[i, i, i] for i in range(5)])
    assert np.array_equal(ordered, expected)
