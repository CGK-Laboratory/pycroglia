import numpy as np
import pytest

from pycroglia.core.connected_components import (
    ComponentConnectivity,
    get_connected_components,
    ConnectedComponents,
)
from pycroglia.core.errors.errors import PycrogliaException

DEFAULT_TEST_CONNECTIVITY = ComponentConnectivity.FACES


def simple_3d_img() -> np.ndarray:
    """Returns a simple 3D binary image with two separate voxels.

    Returns:
        np.ndarray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[0, 0, 0] = 1
    img[2, 2, 2] = 1
    return img


def touching_voxels_img() -> np.ndarray:
    """Returns a 3D image with two voxels touching by edge or corner only.

    Returns:
        np.ndarray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[1, 1, 1] = 1
    img[1, 2, 2] = 1
    return img


def diagonal_voxels_img() -> np.ndarray:
    """Returns a 3D image with two voxels touching only by corner.

    Returns:
        np.ndarray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[0, 0, 0] = 1
    img[1, 1, 1] = 1
    return img


def stacked_voxels_image() -> np.ndarray:
    """Returns a 3D image with three voxels stacked at the same (x, y) position.

    Returns:
        np.ndarray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[0, 1, 1] = 1
    img[1, 1, 1] = 1
    img[2, 1, 1] = 1
    return img


def separate_voxels_image() -> np.ndarray:
    """Returns a 3D image with three separate voxels at different (x, y) positions.

    Returns:
        np.ndarray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[0, 0, 0] = 1
    img[1, 1, 1] = 1
    img[2, 2, 2] = 1
    return img


@pytest.mark.parametrize(
    "img_fn, connectivity, expected_components",
    [
        (simple_3d_img, ComponentConnectivity.FACES, 2),
        (simple_3d_img, ComponentConnectivity.EDGES, 2),
        (simple_3d_img, ComponentConnectivity.CORNERS, 2),
        (touching_voxels_img, ComponentConnectivity.FACES, 2),
        (touching_voxels_img, ComponentConnectivity.EDGES, 1),
        (touching_voxels_img, ComponentConnectivity.CORNERS, 1),
        (diagonal_voxels_img, ComponentConnectivity.FACES, 2),
        (diagonal_voxels_img, ComponentConnectivity.EDGES, 2),
        (diagonal_voxels_img, ComponentConnectivity.CORNERS, 1),
    ],
)
def test_get_connected_components(img_fn, connectivity, expected_components):
    """Test get_connected_components for all connectivity and image cases.

    Args:
        img_fn (Callable): Function that returns a 3D image.
        connectivity (ComponentConnectivity): Connectivity type.
        expected_components (int): Expected number of components.

    Asserts:
        The number of unique labels matches the expected number of components.
    """
    img = img_fn()
    labels = get_connected_components(img, connectivity)
    assert labels.max() == expected_components
    assert set(np.unique(labels)) == set(range(expected_components + 1))


@pytest.mark.parametrize(
    "connectivity, expected",
    [
        (ComponentConnectivity.FACES, 2),
        (ComponentConnectivity.EDGES, 2),
        (ComponentConnectivity.CORNERS, 2),
    ],
)
def test_connected_components_len(connectivity, expected):
    """Test ConnectedComponents.len() functionality.

    Args:
        connectivity (ComponentConnectivity): Connectivity type.
        expected (int): Expected number of components.

    Asserts:
        The number of components matches the expected value.
    """
    img = simple_3d_img()
    cc = ConnectedComponents(img, connectivity)
    assert cc.len() == expected


def test_connected_components_component_to_2d():
    """Test component_to_2d for stacked voxels.

    Asserts:
        The 2D projection sums the stacked voxels correctly.
    """
    img = stacked_voxels_image()
    cc = ConnectedComponents(img, DEFAULT_TEST_CONNECTIVITY)

    assert cc.len() == 1
    got = cc.component_to_2d(1)

    assert got.shape == (3, 3)
    assert got[1, 1] == 3
    assert np.sum(got) == 3


@pytest.mark.parametrize(
    "img_fn, connectivity, index",
    [
        (stacked_voxels_image, ComponentConnectivity.FACES, -1),  # Invalid value
        (
            stacked_voxels_image,
            ComponentConnectivity.FACES,
            3,
        ),  # Bigger than number of components
    ],
)
def test_connected_components_component_to_2d_nok(img_fn, connectivity, index):
    """Test component_to_2d raises exception for invalid index.

    Args:
        img_fn (Callable): Function that returns a 3D image.
        connectivity (ComponentConnectivity): Connectivity type.
        index (int): Invalid component index.

    Asserts:
        PycrogliaException is raised for invalid index.
    """
    img = img_fn()
    cc = ConnectedComponents(img, connectivity)

    with pytest.raises(PycrogliaException) as err:
        cc.component_to_2d(index)

    assert err.value.error_code == 2000


def test_connected_components_all_components_to_2d():
    """Test all_components_to_2d for separate voxels.

    Asserts:
        The shape and values of the 3D array are as expected for separate voxels.
    """
    img = separate_voxels_image()
    cc = ConnectedComponents(img, DEFAULT_TEST_CONNECTIVITY)

    all_2d = cc.all_components_to_2d()

    assert all_2d.shape == (3, 3, 3)
    assert all_2d[1, 1, 1] == 1
    assert all_2d[0, 0, 0] == 1
    assert all_2d[2, 2, 2] == 1
    assert np.sum(all_2d) == 3


def test_overlap_components():
    """Test overlap_components for simple 3D image.

    Asserts:
        The overlap image has correct shape and values.
    """
    img = simple_3d_img()
    cc = ConnectedComponents(img, DEFAULT_TEST_CONNECTIVITY)
    overlap = cc.overlap_components()

    assert overlap.shape == (3, 3)
    assert overlap[0, 0] == 1
    assert overlap[2, 2] == 1
    assert overlap.sum() == 2
