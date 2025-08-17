import numpy as np
from pycroglia.core.filters import calculate_otsu_threshold, remove_small_objects


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
    img = np.zeros((1, 10, 10), dtype=np.uint8)
    img[0, 5:, :] = 255  # bottom half white
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
