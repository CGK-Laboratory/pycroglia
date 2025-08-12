import numpy as np

from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture

from pycroglia.core.errors.errors import PycrogliaException
from pycroglia.core.labeled_cells import LabeledCells, CellConnectivity


def get_number_of_nuclei(img: NDArray, connectivity: CellConnectivity) -> int:
    """Gets the number of nuclei in a 3D binary image.

    Args:
        img (NDArray): 3D binary image.
        connectivity (CellConnectivity): Connectivity type for labeling.

    Returns:
        int: Number of detected nuclei.

    Raises:
        PycrogliaException: If no nuclei are detected (error_code=2001).
    """
    number_of_nuclei = LabeledCells(img, connectivity).len()
    if number_of_nuclei == 0:
        raise PycrogliaException(error_code=2001)
    elif number_of_nuclei == 1:
        return 2

    return number_of_nuclei


def gaussian_mixture_predict(
    img: NDArray, n_clusters: int, n_init: int
) -> list[NDArray[np.uint8]]:
    """Applies a Gaussian Mixture Model (GMM) to segment a 3D binary image into clusters.

    Args:
        img (NDArray): 3D binary image.
        n_clusters (int): Number of clusters to segment.
        n_init (int): Number of initializations for the GMM.

    Returns:
        list[NDArray[np.uint8]]: List of 3D binary masks, one for each cluster.
    """
    points = np.column_stack(np.nonzero(img))

    gmm = GaussianMixture(n_components=n_clusters, n_init=n_init)
    gmm_index = gmm.fit_predict(points)

    clusters = []
    for i in range(n_clusters):
        cluster_points = points[gmm_index == i]

        flat_indexes = np.ravel_multi_index(
            (cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2]),
            img.shape,
        )
        cluster_map = np.zeros(img.shape, dtype=np.uint8)
        cluster_map.flat[flat_indexes] = 1

        clusters.append(cluster_map)

    return clusters
