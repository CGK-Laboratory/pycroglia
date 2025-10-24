import numpy as np
from pycroglia.core import bounding_box


def test_bounding_box_of_cell():
    """
    Test Bounding Box computation on a simple 3D volume.

    Scenario:
        A single foreground voxel is placed at (z=2, y=3, x=4)
        in a (5,6,7) volume. The bounding box should crop
        along Y and X to tightly include only that voxel.

    Asserts:
        - The cropped volume has shape (5, 1, 1), shrinking only in y and x.
        - The bounding box left/right (x bounds) are both 4.
        - The bounding box bottom/top (y bounds) are both 3.
        - The voxel at the expected cropped coordinate is True.
    """
    # Create a 3D volume (z=5, y=6, x=7)
    vol = np.zeros((5, 6, 7), dtype=bool)
    vol[2, 3, 4] = True  # foreground voxel at (z=2, y=3, x=4)

    result = bounding_box.compute(vol)

    # Since cropping happens along Y and X
    assert result.bounded_img.shape == (5, 1, 1), (
        f"Cropped volume should shrink Y,X but keep full Z; got {result.bounded_img.shape}"
    )

    assert result.left == 4 and result.right == 4, "X bounds should match voxel x=4"
    assert result.bottom == 3 and result.top == 3, "Y bounds should match voxel y=3"

    # Check voxel presence at corresponding position
    assert result.bounded_img[2, 0, 0], (
        "Foreground voxel should remain in cropped volume"
    )
