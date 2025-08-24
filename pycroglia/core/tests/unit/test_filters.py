import numpy as np
import pytest
from pycroglia.core.filters import calculate_otsu_threshold, remove_small_objects
from pycroglia.core.enums import SkimageCellConnectivity


def test_otsu_mask_all_black():
    """Test that an image with all pixels set to 0 produces an all-zero binary mask.

    Asserts:
        The mask shape matches the input and all values are zero.
    """
    img = np.zeros((1, 10, 10), dtype=np.uint8)

    mask = calculate_otsu_threshold(img, adjust=1.0)

    assert mask.shape == img.shape
    assert np.all(mask == 0)


def test_otsu_mask_all_white():
    """Test that an image with all pixels set to 255 produces an all-one binary mask.

    Asserts:
        The mask contains only ones and has the correct shape.
    """
    img = np.ones((1, 10, 10), dtype=np.uint8) * 255

    mask = calculate_otsu_threshold(img, adjust=1.0)

    assert mask.shape == img.shape
    assert np.all(mask == 1)


def test_otsu_mask_mixed_values():
    """Test Otsu's method on an image with half 0s and half 255s.

    Asserts:
        The mask contains both foreground and background values and has the correct shape.
    """
    # Matrix with bottom half white
    img = np.zeros((1, 10, 10), dtype=np.uint8)
    img[0, 5:, :] = 255

    mask = calculate_otsu_threshold(img, adjust=1.0)

    assert mask.shape == img.shape
    assert np.sum(mask) > 0


def test_remove_small_object_filter_object():
    """Test that a small object (single pixel) is removed if below the min_size threshold.

    Asserts:
        The result contains no objects (all zeros) and has the correct shape.
    """
    img = np.zeros((1, 10, 10), dtype=bool)
    img[0, 0, 0] = 1

    result = remove_small_objects(img, min_size=2)

    assert result.shape == img.shape
    assert result.sum() == 0


def test_keep_large_object_dont_filter_object():
    """Test that a sufficiently large object is preserved after filtering.

    Asserts:
        The result contains all original object pixels and has the correct shape.
    """
    img = np.zeros((1, 10, 10), dtype=bool)
    img[0, 0:3, 0:3] = 1

    result = remove_small_objects(img, min_size=5)

    assert result.shape == img.shape
    assert result.sum() == 9


def test_remove_small_objects_multiple_sizes():
    """Test that only objects above min_size are kept when multiple objects of different sizes are present.

    Asserts:
        Only the large object remains and the shape is correct.
    """
    # Matrix with small object in (0, 0, 0)
    # Matrix with large object (25 pixels)
    img = np.zeros((1, 10, 10), dtype=bool)
    img[0, 0, 0] = 1
    img[0, 5:10, 5:10] = 1

    result = remove_small_objects(img, min_size=10)

    assert result.shape == img.shape
    assert np.sum(result) == 25
    assert np.all(result[0, 5:10, 5:10] == 1)
    assert result[0, 0, 0] == 0


def test_remove_small_objects_3d():
    """Test remove_small_objects on a 3D image with objects in different slices.

    Asserts:
        Only objects above min_size remain and the shape is correct.
    """

    # Matrix with small object in slice 0
    # Matrix with large object in slice 1 (4 pixels)
    # Matrix with small object in slice 2
    img = np.zeros((3, 5, 5), dtype=bool)
    img[0, 0, 0] = 1
    img[1, 1:3, 1:3] = 1
    img[2, 4, 4] = 1

    result = remove_small_objects(img, min_size=4)

    assert result.shape == img.shape
    assert np.sum(result) == 4
    assert np.all(result[1, 1:3, 1:3] == 1)
    assert result[0, 0, 0] == 0
    assert result[2, 4, 4] == 0


@pytest.mark.parametrize(
    "connectivity,expected_sum",
    [
        (SkimageCellConnectivity.EDGES, 3),
        (SkimageCellConnectivity.CORNERS, 3),
    ],
)
def test_remove_small_objects_connectivity(connectivity, expected_sum):
    """Test remove_small_objects with different connectivity settings using the same image.

    Args:
        connectivity (SkimageCellConnectivity): Connectivity to use.
        expected_sum (int): Expected sum of the result.

    Asserts:
        The result sum and shape are as expected for each connectivity.
    """
    # Object with three pixels in a line
    img = np.zeros((1, 5, 5), dtype=bool)
    img[0, 2, 1] = 1
    img[0, 2, 2] = 1
    img[0, 2, 3] = 1

    result = remove_small_objects(img, min_size=3, connectivity=connectivity)
    assert result.shape == img.shape
    assert np.sum(result) == expected_sum


def test_remove_small_objects_dtype_uint8():
    """Test remove_small_objects with uint8 input and verify output dtype and shape.

    Asserts:
        Output dtype matches input and shape is correct.
    """
    img = np.zeros((1, 5, 5), dtype=np.uint8)
    img[0, 1:4, 1:4] = 1

    result = remove_small_objects(img, min_size=5)

    assert result.shape == img.shape
    assert result.dtype == img.dtype
    assert np.sum(result) == 9


def test_remove_small_objects_object_on_border():
    """Test that an object touching the border is handled correctly.

    Asserts:
        The object is kept or removed based on min_size and shape is correct.
    """
    # Object on border (3 pixels)
    img = np.zeros((1, 5, 5), dtype=bool)
    img[0, 0, 0:3] = 1

    result = remove_small_objects(img, min_size=2)

    assert result.shape == img.shape
    assert np.sum(result) == 3
    assert np.all(result[0, 0, 0:3] == 1)
