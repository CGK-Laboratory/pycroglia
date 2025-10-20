import numpy as np
import pytest
from pycroglia.core.connection import connect_points_along_path, Point


def test_path_not_found():
    """Test that ValueError is raised when no path exists between isolated voxels.

    Scenario:
        - A 5x5x5 volume has two True voxels: (2,1,1) and (2,3,3).
        - They are disconnected, so BFS cannot find a valid path.

    Asserts:
        - Function raises ValueError ("No path found between points.")
    """
    vol = np.zeros((5, 5, 5), dtype=bool)
    vol[2, 1, 1] = True
    vol[2, 3, 3] = True

    start = Point(x=1, y=1, z=2)
    end = Point(x=3, y=3, z=2)

    with pytest.raises(ValueError, match="No path found between points"):
        connect_points_along_path(vol, start, end)


def test_simple_diagonal_path():
    """Test that a diagonal chain of True voxels is correctly traversed.

    Scenario:
        - A 5x5x5 diagonal line of True voxels from (0,0,0) to (4,4,4).
        - Start = (0,0,0), End = (4,4,4).

    Asserts:
        - Returned array is 3D mask.
        - Path connects start and end voxels.
    """
    vol = np.zeros((5, 5, 5), dtype=bool)
    for i in range(5):
        vol[i, i, i] = True

    start = Point(x=0, y=0, z=0)
    end = Point(x=4, y=4, z=4)

    mask = connect_points_along_path(vol, start, end)

    # Mask should contain a diagonal of 5 True voxels
    coords = np.argwhere(mask)
    expected = np.array([[i, i, i] for i in range(5)])

    np.testing.assert_array_equal(
        coords, expected, err_msg="Path does not follow diagonal."
    )


def test_complex_path():
    """Test that an L-shaped path is reconstructed correctly.

    Scenario:
        - A 5x5x5 volume with path:
          (2,1,1) → (2,1,2) → (2,1,3) → (2,2,3) → (2,3,3)

       - Start = (2,1,1), End = (2,3,3).

    Asserts:
        - Path mask corresponds to expected coordinates.
    """
    vol = np.zeros((5, 5, 5), dtype=bool)
    vol[2, 1, 1] = True
    vol[2, 1, 2] = True
    vol[2, 1, 3] = True
    vol[2, 2, 3] = True
    vol[2, 3, 3] = True

    start = Point(x=1, y=1, z=2)
    end = Point(x=3, y=3, z=2)

    mask = connect_points_along_path(vol, start, end)
    coords = np.argwhere(mask)

    # BFS may take a direct 26-connected path skipping (2,1,3)
    expected_minimal = np.array(
        [
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 3],
            [2, 3, 3],
        ]
    )

    np.testing.assert_array_equal(
        coords,
        expected_minimal,
        err_msg="Path does not match expected minimal L-shape.",
    )
