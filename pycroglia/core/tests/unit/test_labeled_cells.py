import numpy as np
import pytest

from numpy.typing import NDArray
from pycroglia.core.labeled_cells import (
    LabeledCells,
    SkimageImgLabeling,
    MaskListLabeling,
)
from pycroglia.core.enums import SkimageCellConnectivity
from pycroglia.core.errors.errors import PycrogliaException

DEFAULT_TEST_CONNECTIVITY = SkimageCellConnectivity.FACES


def simple_3d_img() -> NDArray:
    """Returns a simple 3D binary image with two separate voxels.

    Returns:
        NDArray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[0, 0, 0] = 1
    img[2, 2, 2] = 1
    return img


def touching_voxels_img() -> NDArray:
    """Returns a 3D image with two voxels touching by edge or corner only.

    Returns:
        NDArray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[1, 1, 1] = 1
    img[1, 2, 2] = 1
    return img


def diagonal_voxels_img() -> NDArray:
    """Returns a 3D image with two voxels touching only by corner.

    Returns:
        NDArray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[0, 0, 0] = 1
    img[1, 1, 1] = 1
    return img


def stacked_voxels_image() -> NDArray:
    """Returns a 3D image with three voxels stacked at the same (x, y) position.

    Returns:
        NDArray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[0, 1, 1] = 1
    img[1, 1, 1] = 1
    img[2, 1, 1] = 1
    return img


def separate_voxels_image() -> NDArray:
    """Returns a 3D image with three separate voxels at different (x, y) positions.

    Returns:
        NDArray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[0, 0, 0] = 1
    img[1, 1, 1] = 1
    img[2, 2, 2] = 1
    return img


def test_skimage_img_labeling_label():
    """Test SkimageImgLabeling labels connected components as expected.

    Asserts:
        The output matches skimage.measure.label for the same connectivity.
    """
    img = simple_3d_img()
    strategy = SkimageImgLabeling(SkimageCellConnectivity.FACES)
    labels = strategy.label(img)
    # Should be 2 components plus background
    assert labels.shape == img.shape
    assert labels.max() == 2
    assert set(np.unique(labels)) == {0, 1, 2}


def test_mask_list_labeling_label():
    """Test MaskListLabeling labels a list of binary masks as expected.

    Asserts:
        Each mask is assigned a unique label in the output.
    """
    mask1 = np.zeros((3, 3, 3), dtype=np.uint8)
    mask2 = np.zeros((3, 3, 3), dtype=np.uint8)
    mask1[0, 0, 0] = 1
    mask2[2, 2, 2] = 1
    masks = [mask1, mask2]
    shape = (3, 3, 3)
    strategy = MaskListLabeling(masks, shape)
    # Pass any array (e.g., zeros) as img, since it's not used
    dummy_img = np.zeros(shape, dtype=np.uint8)
    labels = strategy.label(dummy_img)
    assert labels.shape == shape
    assert labels[0, 0, 0] == 1
    assert labels[2, 2, 2] == 2
    assert np.sum(labels == 1) == 1
    assert np.sum(labels == 2) == 1
    assert np.sum(labels == 0) == (3 * 3 * 3 - 2)


@pytest.mark.parametrize(
    "index, expected", [(-1, False), (0, False), (1, True), (2, True), (3, False)]
)
def test_labeled_cells_is_valid_index(index, expected):
    """Test _is_valid_index for various indices.

    Args:
        index (int): Index to test.
        expected (bool): Expected result.

    Asserts:
        The result of _is_valid_index matches the expected value.
    """
    img = simple_3d_img()
    lc = LabeledCells(img, SkimageImgLabeling(DEFAULT_TEST_CONNECTIVITY))
    assert lc._is_valid_index(index) == expected


@pytest.mark.parametrize(
    "connectivity, expected",
    [
        (SkimageCellConnectivity.FACES, 2),
        (SkimageCellConnectivity.EDGES, 2),
        (SkimageCellConnectivity.CORNERS, 2),
    ],
)
def test_labeled_cells_len(connectivity, expected):
    """Test LabeledCells.len() functionality.

    Args:
        connectivity (SkimageCellConnectivity): Connectivity type.
        expected (int): Expected number of components.

    Asserts:
        The number of components matches the expected value.
    """
    img = simple_3d_img()
    lc = LabeledCells(img, SkimageImgLabeling(connectivity))
    assert lc.len() == expected


def test_labeled_cells_get_cell():
    """Test get_cell returns correct mask for stacked voxels.

    Asserts:
        The mask shape and values are as expected.
    """
    img = stacked_voxels_image()
    lc = LabeledCells(img, SkimageImgLabeling(DEFAULT_TEST_CONNECTIVITY))
    mask = lc.get_cell(1)

    assert mask.shape == (3, 3, 3)
    assert np.sum(mask) == 3
    assert mask[0, 1, 1] == 1
    assert mask[1, 1, 1] == 1
    assert mask[2, 1, 1] == 1


@pytest.mark.parametrize("index", [-1, 0, 2])
def test_labeled_cells_get_cell_invalid_index(index):
    """Test get_cell raises exception for invalid indices.

    Args:
        index (int): Invalid index to test.

    Asserts:
        PycrogliaException is raised with error_code 2000.
    """
    img = stacked_voxels_image()
    lc = LabeledCells(img, SkimageImgLabeling(DEFAULT_TEST_CONNECTIVITY))

    with pytest.raises(PycrogliaException) as err:
        lc.get_cell(index)
    assert err.value.error_code == 2000


def test_labeled_cells_get_cell_size():
    """Test get_cell_size returns correct voxel count for a cell.

    Asserts:
        The size matches the number of voxels in the cell.
    """
    img = stacked_voxels_image()
    lc = LabeledCells(img, SkimageImgLabeling(DEFAULT_TEST_CONNECTIVITY))
    size = lc.get_cell_size(1)
    assert size == 3


@pytest.mark.parametrize("index", [-1, 0, 2])
def test_labeled_cells_get_cell_size_invalid_index(index):
    """Test get_cell_size raises exception for invalid indices.

    Args:
        index (int): Invalid index to test.

    Asserts:
        PycrogliaException is raised with error_code 2000.
    """
    img = stacked_voxels_image()
    lc = LabeledCells(img, SkimageImgLabeling(DEFAULT_TEST_CONNECTIVITY))
    with pytest.raises(PycrogliaException) as err:
        lc.get_cell_size(index)
    assert err.value.error_code == 2000


def test_labeled_cells_cell_to_2d():
    """Test cell_to_2d for stacked voxels.

    Asserts:
        The 2D projection sums the stacked voxels correctly.
    """
    img = stacked_voxels_image()
    lc = LabeledCells(img, SkimageImgLabeling(DEFAULT_TEST_CONNECTIVITY))

    assert lc.len() == 1
    got = lc.cell_to_2d(1)

    assert got.shape == (3, 3)
    assert got[1, 1] == 3
    assert np.sum(got) == 3


@pytest.mark.parametrize(
    "img_fn, connectivity, index",
    [
        (stacked_voxels_image, SkimageCellConnectivity.FACES, -1),  # Invalid value
        (
            stacked_voxels_image,
            SkimageCellConnectivity.FACES,
            3,
        ),  # Bigger than number of cells
    ],
)
def test_labeled_cells_cell_to_2d_nok(img_fn, connectivity, index):
    """Test cell_to_2d raises exception for invalid index.

    Args:
        img_fn (Callable): Function that returns a 3D image.
        connectivity (SkimageCellConnectivity): Connectivity type.
        index (int): Invalid component index.

    Asserts:
        PycrogliaException is raised for invalid index.
    """
    img = img_fn()
    lc = LabeledCells(img, SkimageImgLabeling(connectivity))

    with pytest.raises(PycrogliaException) as err:
        lc.cell_to_2d(index)

    assert err.value.error_code == 2000


def test_labeled_cells_all_cells_to_2d():
    """Test all_cells_to_2d for separate voxels.

    Asserts:
        The shape and values of the 3D array are as expected for separate voxels.
    """
    img = separate_voxels_image()
    lc = LabeledCells(img, SkimageImgLabeling(DEFAULT_TEST_CONNECTIVITY))

    all_2d = lc.all_cells_to_2d()

    assert all_2d.shape == (3, 3, 3)
    assert all_2d[1, 1, 1] == 1
    assert all_2d[0, 0, 0] == 1
    assert all_2d[2, 2, 2] == 1
    assert np.sum(all_2d) == 3
