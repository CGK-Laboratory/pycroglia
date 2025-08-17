import numpy as np
import pytest

from numpy.typing import NDArray
from pycroglia.core.erosion import (
    Diamond2DFootprint,
    Rectangle2DFootprint,
    Disk2DFootprint,
    Rectangle3DFootprint,
    Ball3DFootprint,
    Octahedron3DFootprint,
    apply_binary_erosion
)


def simple_binary_img() -> NDArray:
    """Returns a simple 2D binary image with a central block.

    Returns:
        NDArray: 2D binary image.
    """
    img = np.zeros((5, 5), dtype=np.uint8)
    img[1:4, 1:4] = 1
    return img


def simple_3d_binary_img() -> NDArray:
    """Returns a simple 3D binary image with a central block.

    Returns:
        NDArray: 3D binary image.
    """
    img = np.zeros((5, 5, 5), dtype=np.uint8)
    img[1:4, 1:4, 1:4] = 1
    return img


def test_diamond_footprint_shape():
    """Test DiamondFootprint returns correct structuring element shape.

    Asserts:
        The returned shape matches skimage's diamond.
    """
    fp = Diamond2DFootprint(r=1)
    shape = fp.get_shape()
    assert shape.shape == (3, 3)
    assert shape[1, 1] == 1
    assert np.sum(shape) == 5


def test_rectangle_footprint_shape():
    """Test RectangleFootprint returns correct structuring element shape.

    Asserts:
        The returned shape matches the requested rectangle.
    """
    fp = Rectangle2DFootprint(x=2, y=3)
    shape = fp.get_shape()
    assert shape.shape == (3, 2)
    assert np.all(shape == 1)


def test_disk_footprint_shape():
    """Test DiskFootprint returns correct structuring element shape.

    Asserts:
        The returned shape matches skimage's disk.
    """
    fp = Disk2DFootprint(r=1)
    shape = fp.get_shape()
    assert shape.shape == (3, 3)
    assert shape[1, 1] == 1
    assert np.sum(shape) == 5


def test_rectangle3d_footprint_shape():
    """Test Rectangle3DFootprint returns correct 3D structuring element shape.

    Asserts:
        The returned shape matches the requested rectangle.
    """
    fp = Rectangle3DFootprint(x=2, y=3, z=1)
    shape = fp.get_shape()
    assert shape.shape == (3, 7, 5)
    assert np.all(shape == 1)


def test_ball3d_footprint_shape():
    """Test Ball3DFootprint returns correct 3D ball structuring element.

    Asserts:
        The returned shape is cubic and center is True.
    """
    fp = Ball3DFootprint(r=1)
    shape = fp.get_shape()
    assert shape.shape == (3, 3, 3)
    assert shape[1, 1, 1] == 1
    assert np.sum(shape) > 1  # Should be more than just the center


def test_octahedron3d_footprint_shape():
    """Test Octahedron3DFootprint returns correct 3D octahedron structuring element.

    Asserts:
        The returned shape is cubic and center is True.
    """
    fp = Octahedron3DFootprint(r=1)
    shape = fp.get_shape()
    assert shape.shape == (3, 3, 3)
    assert shape[1, 1, 1] == 1
    assert np.sum(shape) > 1  # Should be more than just the center


@pytest.mark.parametrize(
    "footprint_cls, fp_args, expected_sum",
    [
        (Diamond2DFootprint, {"r": 1}, 1),
        (Rectangle2DFootprint, {"x": 3, "y": 3}, 1),
        (Disk2DFootprint, {"r": 1}, 1),
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


@pytest.mark.parametrize(
    "footprint_cls, fp_args, expected_sum",
    [
        (Rectangle3DFootprint, {"x": 1, "y": 1, "z": 1}, 1),
        (Ball3DFootprint, {"r": 1}, 1),
        (Octahedron3DFootprint, {"r": 1}, 1),
    ],
)
def test_apply_binary_erosion_3d(footprint_cls, fp_args, expected_sum):
    """Test apply_binary_erosion erodes a 3D image as expected.

    Args:
        footprint_cls (type): Footprint class to use.
        fp_args (dict): Arguments for the footprint.
        expected_sum (int): Expected sum of the eroded image.

    Asserts:
        The sum of the eroded image matches the expected value and shape.
    """
    img = simple_3d_binary_img()
    fp = footprint_cls(**fp_args)
    eroded = apply_binary_erosion(img, fp)
    assert np.sum(eroded) == expected_sum
    assert eroded.shape == img.shape
