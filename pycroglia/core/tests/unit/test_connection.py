import numpy as np
from pycroglia.core.connection import connect_points_along_path, Point


def test_path_not_found():
    """Test that no path is found when start and end voxels are isolated.

    Scenario:
        - A 5x5x5 volume is created with only two voxels set to True,
          at (2,1,1) and (2,3,3).
        - Since there is no connecting chain of True voxels, the BFS
          should fail to find a path.

    Asserts:
        - Returned path has size 0, indicating no connection exists.
    """
    vol = np.zeros((5, 5, 5), dtype=bool)
    vol[2, 1, 1] = True
    vol[2, 3, 3] = True

    start = Point(x=1, y=1, z=2)
    end = Point(x=3, y=3, z=2)

    path = connect_points_along_path(vol, start, end)

    assert path.size == 0, f"Expected empty array, got shape {path.shape}"


def test_simple_diagonal_path():
    """Test that a diagonal chain of voxels is traversed correctly.

    Scenario:
        - A 5x5x5 volume with a diagonal line of True voxels from
          (0,0,0) to (4,4,4).
        - Start = (0,0,0), End = (4,4,4).

    Asserts:
        - Path contains exactly 5 points.
        - Path equals the diagonal sequence [[0,0,0], [1,1,1], ..., [4,4,4]].
    """
    vol = np.zeros((5, 5, 5), dtype=bool)
    # diagonal from (0,0,0) to (4,4,4)
    for i in range(5):
        vol[i, i, i] = True

    start = Point(x=0, y=0, z=0)
    end = Point(x=4, y=4, z=4)

    path = connect_points_along_path(vol, start, end)

    # Path should cover 5 voxels along the diagonal
    assert path.shape == (5, 3)
    np.testing.assert_array_equal(path, np.array([[i, i, i] for i in range(5)]))


def test_complex_path():
    """Test that an L-shaped path is reconstructed correctly.

    Scenario:
        - A 5x5x5 volume with a path at z=2 forming:
            (2,1,1) → (2,1,2) → (2,1,3) → (2,2,3) → (2,3,3).
        - Start = (2,1,1), End = (2,3,3).

    Asserts:
        - Path follows the expected L-shape.
        - Path equals [[2,1,1], [2,1,2], [2,2,3], [2,3,3]].
    """
    # Create a 5x5x5 volume
    img = np.zeros((5, 5, 5), dtype=bool)

    # Build an L-shaped path at z=2:
    # from (2,1,1) → (2,1,2) → (2,1,3) → (2,2,3) → (2,3,3)
    img[2, 1, 1] = True
    img[2, 1, 2] = True
    img[2, 1, 3] = True
    img[2, 2, 3] = True
    img[2, 3, 3] = True

    start = Point(x=1, y=1, z=2)
    end = Point(x=3, y=3, z=2)

    path = connect_points_along_path(img, start, end)

    # Expected path in order (z,y,x)
    expected = np.array(
        [
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 3],
            [2, 3, 3],
        ],
        dtype=int,
    )

    np.testing.assert_array_equal(path, expected)
