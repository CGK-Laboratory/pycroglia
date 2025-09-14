import numpy as np
from pycroglia.core.nearest_pixel import compute, Coordinates


def test_nearest_pixel():
    """Test NearestPixel correctly finds the closest foreground voxel.

    Scenario:
        - A 5×5×5 binary volume is created with a single voxel set to True
          at coordinates (z=2, y=3, x=4).
        - The search starts at (0,0,0) with isotropic scale=1.0.

    Asserts:
        - The nearest voxel coordinates returned by `compute` match the
          expected location (x=4, y=3, z=2).
        - The returned result is equal to a NearestPixelCoordinates instance
          with these values.
    """

    # Create a simple 5x5x5 volume
    vol = np.zeros((5, 5, 5), dtype=bool)
    vol[2, 3, 4] = True  # foreground voxel at (z=2, y=3, x=4)

    result = compute(vol, (0, 0, 0), scale=1.0)

    # Expected nearest voxel
    expected = Coordinates(x=4, y=3, z=2)

    assert result == expected, f"Expected {expected}, got {result}"
