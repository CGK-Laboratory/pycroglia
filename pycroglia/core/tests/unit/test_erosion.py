import numpy as np
import pytest

from numpy.typing import NDArray
from pycroglia.core.erosion import (
    DiamondFootprint,
    RectangleFootprint,
    DiskFootprint,
    apply_binary_erosion,
)


def simple_binary_img() -> NDArray:
    """Returns a simple 2D binary image with a central block.

    Returns:
        NDArray: 2D binary image.
    """
    img = np.zeros((5, 5), dtype=np.uint8)
    img[1:4, 1:4] = 1
    return img


def test_diamond_footprint_shape():
    """Test DiamondFootprint returns correct structuring element shape.

    Asserts:
        The returned shape matches skimage's diamond.
    """
    fp = DiamondFootprint(r=1)
    shape = fp.get_shape()
    assert shape.shape == (3, 3)
    assert shape[1, 1] == 1
    assert np.sum(shape) == 5


def test_rectangle_footprint_shape():
    """Test RectangleFootprint returns correct structuring element shape.

    Asserts:
        The returned shape matches the requested rectangle.
    """
    fp = RectangleFootprint(x=2, y=3)
    shape = fp.get_shape()
    assert shape.shape == (3, 2)
    assert np.all(shape == 1)


def test_disk_footprint_shape():
    """Test DiskFootprint returns correct structuring element shape.

    Asserts:
        The returned shape matches skimage's disk.
    """
    fp = DiskFootprint(r=1)
    shape = fp.get_shape()
    assert shape.shape == (3, 3)
    assert shape[1, 1] == 1
    assert np.sum(shape) == 5


@pytest.mark.parametrize(
    "footprint_cls, fp_args, expected_sum",
    [
        (DiamondFootprint, {"r": 1}, 1),
        (RectangleFootprint, {"x": 3, "y": 3}, 1),
        (DiskFootprint, {"r": 1}, 1),
    ],
)
def test_apply_binary_erosion(footprint_cls, fp_args, expected_sum):
    """Test apply_binary_erosion erodes the image as expected.

    Args:
        footprint_cls (type): Footprint class to use.
        fp_args (dict): Arguments for the footprint.
        expected_sum (int): Expected sum of the eroded image.

    Asserts:
        The sum of the eroded image matches the expected value.
    """
    img = simple_binary_img()
    fp = footprint_cls(**fp_args)
    eroded = apply_binary_erosion(img, fp)
    assert np.sum(eroded) == expected_sum
    assert eroded.shape == img.shape
