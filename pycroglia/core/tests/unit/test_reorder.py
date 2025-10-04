import numpy as np
from pycroglia.core.reorder import reorder_pixel_list


def test_simple_line_path():
    """Reorder a straight line from endpoint to centroid."""
    shape = (5, 5, 5)
    mask = np.zeros(shape, dtype=bool)
    for i in range(5):
        mask[i, 2, 2] = True

    pixel_indices = np.flatnonzero(mask)
    endpoint = np.array([0, 2, 2])
    centroid = np.array([4, 2, 2])

    ordered = reorder_pixel_list(pixel_indices, shape, endpoint, centroid)

    expected = np.array([[i, 2, 2] for i in range(5)])
    assert np.array_equal(ordered, expected), (
        f"Expected:\n{expected}\nGot:\n{ordered}"
    )


def test_diagonal_path():
    """Reorder a diagonal line in 3D."""
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
