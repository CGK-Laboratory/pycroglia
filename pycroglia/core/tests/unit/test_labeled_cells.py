import numpy as np
import pytest

from numpy.typing import NDArray
from pycroglia.core.labeled_cells import LabeledCells, label_cells, CellConnectivity
from pycroglia.core.errors.errors import PycrogliaException

DEFAULT_TEST_CONNECTIVITY = CellConnectivity.FACES


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


@pytest.mark.parametrize(
    "img_fn, connectivity, expected_components",
    [
        (simple_3d_img, CellConnectivity.FACES, 2),
        (simple_3d_img, CellConnectivity.EDGES, 2),
        (simple_3d_img, CellConnectivity.CORNERS, 2),
        (touching_voxels_img, CellConnectivity.FACES, 2),
        (touching_voxels_img, CellConnectivity.EDGES, 1),
        (touching_voxels_img, CellConnectivity.CORNERS, 1),
        (diagonal_voxels_img, CellConnectivity.FACES, 2),
        (diagonal_voxels_img, CellConnectivity.EDGES, 2),
        (diagonal_voxels_img, CellConnectivity.CORNERS, 1),
    ],
)
def test_label_cells(img_fn, connectivity, expected_components):
    """Test label_cells for all connectivity and image cases.

    Args:
        img_fn (Callable): Function that returns a 3D image.
        connectivity (CellConnectivity): Connectivity type.
        expected_components (int): Expected number of components.

    Asserts:
        The number of unique labels matches the expected number of components.
    """
    img = img_fn()
    labels = label_cells(img, connectivity)
    assert labels.max() == expected_components
    assert set(np.unique(labels)) == set(range(expected_components + 1))


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
    lc = LabeledCells(img, DEFAULT_TEST_CONNECTIVITY)
    assert lc._is_valid_index(index) == expected


@pytest.mark.parametrize(
    "connectivity, expected",
    [
        (CellConnectivity.FACES, 2),
        (CellConnectivity.EDGES, 2),
        (CellConnectivity.CORNERS, 2),
    ],
)
def test_labeled_cells_len(connectivity, expected):
    """Test LabeledCells.len() functionality.

    Args:
        connectivity (CellConnectivity): Connectivity type.
        expected (int): Expected number of components.

    Asserts:
        The number of components matches the expected value.
    """
    img = simple_3d_img()
    lc = LabeledCells(img, connectivity)
    assert lc.len() == expected


def test_labeled_cells_get_cell():
    """Test get_cell returns correct mask for stacked voxels.

    Asserts:
        The mask shape and values are as expected.
    """
    img = stacked_voxels_image()
    lc = LabeledCells(img, DEFAULT_TEST_CONNECTIVITY)
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
    lc = LabeledCells(img, DEFAULT_TEST_CONNECTIVITY)

    with pytest.raises(PycrogliaException) as err:
        lc.get_cell(index)
    assert err.value.error_code == 2000


def test_labeled_cells_cell_to_2d():
    """Test cell_to_2d for stacked voxels.

    Asserts:
        The 2D projection sums the stacked voxels correctly.
    """
    img = stacked_voxels_image()
    lc = LabeledCells(img, DEFAULT_TEST_CONNECTIVITY)

    assert lc.len() == 1
    got = lc.cell_to_2d(1)

    assert got.shape == (3, 3)
    assert got[1, 1] == 3
    assert np.sum(got) == 3


@pytest.mark.parametrize(
    "img_fn, connectivity, index",
    [
        (stacked_voxels_image, CellConnectivity.FACES, -1),  # Invalid value
        (
            stacked_voxels_image,
            CellConnectivity.FACES,
            3,
        ),  # Bigger than number of cells
    ],
)
def test_labeled_cells_cell_to_2d_nok(img_fn, connectivity, index):
    """Test cell_to_2d raises exception for invalid index.

    Args:
        img_fn (Callable): Function that returns a 3D image.
        connectivity (CellConnectivity): Connectivity type.
        index (int): Invalid component index.

    Asserts:
        PycrogliaException is raised for invalid index.
    """
    img = img_fn()
    lc = LabeledCells(img, connectivity)

    with pytest.raises(PycrogliaException) as err:
        lc.cell_to_2d(index)

    assert err.value.error_code == 2000


def test_labeled_cells_all_cells_to_2d():
    """Test all_cells_to_2d for separate voxels.

    Asserts:
        The shape and values of the 3D array are as expected for separate voxels.
    """
    img = separate_voxels_image()
    lc = LabeledCells(img, DEFAULT_TEST_CONNECTIVITY)

    all_2d = lc.all_cells_to_2d()

    assert all_2d.shape == (3, 3, 3)
    assert all_2d[1, 1, 1] == 1
    assert all_2d[0, 0, 0] == 1
    assert all_2d[2, 2, 2] == 1
    assert np.sum(all_2d) == 3


def test_labeled_cells_overlap_cells():
    """Test overlap_cells for simple 3D image.

    Asserts:
        The overlap image has correct shape and values.
    """
    img = simple_3d_img()
    lc = LabeledCells(img, DEFAULT_TEST_CONNECTIVITY)
    overlap = lc.overlap_cells()

    assert overlap.shape == (3, 3)
    assert overlap[0, 0] == 1
    assert overlap[2, 2] == 1
    assert overlap.sum() == 2
