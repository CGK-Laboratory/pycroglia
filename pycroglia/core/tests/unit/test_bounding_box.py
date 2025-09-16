import numpy as np
from pycroglia.core import bounding_box


def test_bounding_box_of_cell():
    """
    Test Bounding Box computation on a simple 3D volume.

    Scenario:
        A single foreground voxel is placed at (z=2, y=3, x=4) in a
        (5,6,7) volume. The bounding box should tightly crop in z and y,
        while keeping the full x dimension.

    Asserts:
        - The cropped volume has shape (1, 1, 7), shrinking only in z and y.
        - The bounding box z-bounds (left/right) are both 2.
        - The bounding box y-bounds (bottom/top) are both 3.
        - The voxel at the expected cropped coordinate is True.
    """
    # Create a 3D volume (z=5, y=6, x=7)
    vol = np.zeros((5, 6, 7), dtype=bool)
    vol[2, 3, 4] = True  # foreground voxel at (z=2, y=3, x=4)

    result = bounding_box.compute(vol)

    assert result.bounded_img.shape == (1, 1, 7), (
        "Cropped volume should keep all x and shrink z,y"
    )
    assert result.left == 2 and result.right == 2, "Z bounds should match voxel z=2"
    assert result.bottom == 3 and result.top == 3, "Y bounds should match voxel y=3"

    assert result.bounded_img[0, 0, 4]
