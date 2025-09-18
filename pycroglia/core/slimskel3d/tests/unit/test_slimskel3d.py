import numpy as np
from pycroglia.core.slimskel3d import slimskel3d
from pycroglia.core.slimskel3d.skeleton3D import skeleton3D


def test_slimskel3d():
    """Test that slimskel3d prunes short spurs while preserving main branches.

    This test constructs a synthetic 10×10×10 volume containing:
      - A main vertical skeleton branch along the z-axis at coordinates (y=5, x=5).
      - A short 2-voxel spur branching in the +x direction at z=5.

    Assertions:
        - The number of voxels after slimming is strictly less than before,
          ensuring spur pruning occurred.
        - The set of remaining voxel coordinates matches the expected slimmed
          skeleton:
            [[2, 5, 5],
             [3, 5, 5],
             [4, 5, 5],
             [6, 5, 5],
             [7, 5, 5],
    """
    # Create a 10x10x10 volume
    vol = np.zeros((10, 10, 10), dtype=bool)

    # Main vertical branch along z at (y=5, x=5) in Python (0-based)
    vol[2:8, 5, 5] = True

    # Add a 2-voxel spur branching in +x direction at z=5
    vol[5, 5, 6] = True  # spur voxel 1
    vol[5, 5, 7] = True  # spur voxel 2

    THR = 2

    # Skeletonize normally
    skel = skeleton3D(vol)
    nvox_before = skel.sum()

    # Run slimming
    slim = slimskel3d.slimskel3d(vol, threshold=THR)
    nvox_after = slim.sum()

    assert nvox_after < nvox_before, "Slim skeleton should prune spurs"
    expected = np.array(
        [
            [2, 5, 5],
            [3, 5, 5],
            [4, 5, 5],
            [6, 5, 5],
            [7, 5, 5],
            [5, 5, 6],
        ]
    )
    slim_coords = np.argwhere(slim)
    assert set(map(tuple, slim_coords)) == set(map(tuple, expected))
