import numpy as np
import pytest

from numpy.typing import NDArray
from pycroglia.core.clustering import get_number_of_nuclei, gaussian_mixture_predict
from pycroglia.core.labeled_cells import SkimageCellConnectivity
from pycroglia.core.errors.errors import PycrogliaException


def empty_img() -> NDArray:
    """Returns a 3D binary image with no nuclei.

    Returns:
        NDArray: 3D binary image.
    """
    return np.zeros((3, 3, 3), dtype=np.uint8)


def single_nucleus_img() -> NDArray:
    """Returns a 3D binary image with a single nucleus.

    Returns:
        NDArray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[1, 1, 1] = 1
    return img


def double_nucleus_img() -> NDArray:
    """Returns a 3D binary image with two separate nuclei.

    Returns:
        NDArray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[0, 0, 0] = 1
    img[2, 2, 2] = 1
    return img


@pytest.mark.parametrize(
    "img_fn, connectivity, expected",
    [
        (double_nucleus_img, SkimageCellConnectivity.FACES, 2),
        (double_nucleus_img, SkimageCellConnectivity.EDGES, 2),
        (
            single_nucleus_img,
            SkimageCellConnectivity.FACES,
            2,
        ),  # Should return 2 for single nucleus
    ],
)
def test_get_number_of_nuclei(img_fn, connectivity, expected):
    """Test get_number_of_nuclei returns correct number of nuclei.

    Args:
        img_fn (Callable): Function that returns a 3D image.
        connectivity (SkimageCellConnectivity): Connectivity type.
        expected (int): Expected number of nuclei.

    Asserts:
        The returned number matches the expected value.
    """
    img = img_fn()
    assert get_number_of_nuclei(img, connectivity) == expected


def test_get_number_of_nuclei_empty_image():
    """Test get_number_of_nuclei raises exception for empty image.

    Asserts:
        PycrogliaException is raised with error_code 2001.
    """
    img = empty_img()
    with pytest.raises(PycrogliaException) as err:
        get_number_of_nuclei(img, SkimageCellConnectivity.FACES)
    assert err.value.error_code == 2001


def test_gaussian_mixture_predict_clusters():
    """Test gaussian_mixture_predict returns correct number of clusters.

    Asserts:
        The number of returned masks matches n_clusters and each mask is binary.
    """
    img = double_nucleus_img()
    n_clusters = 2
    n_init = 1
    clusters = gaussian_mixture_predict(img, n_clusters, n_init)
    assert len(clusters) == n_clusters
    for mask in clusters:
        assert mask.shape == img.shape
        assert set(np.unique(mask)).issubset({0, 1})
        # Each mask should have at least one voxel
        assert np.sum(mask) >= 1
