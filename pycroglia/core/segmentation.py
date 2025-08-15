from dataclasses import dataclass
from typing import List, ClassVar
from numpy.typing import NDArray

from pycroglia.core.clustering import get_number_of_nuclei, gaussian_mixture_predict
from pycroglia.core.erosion import apply_binary_erosion, FootprintShape
from pycroglia.core.filters import remove_small_objects
from pycroglia.core.labeled_cells import (
    LabeledCells,
    CellConnectivity,
    SkimageImgLabeling,
)


@dataclass
class SegmentationConfig:
    """Configuration for cell segmentation.

    Attributes:
        DEFAULT_MIN_NUCLEUS_FRACTION (int): Default minimum nucleus fraction.
        DEFAULT_GMM_N_INIT (int): Default number of GMM initializations.
        cut_off_size (int): Minimum size for a cell to be segmented.
        noise (int): Minimum size for objects to keep after noise removal.
        connectivity (CellConnectivity): Connectivity rule for labeling.
        min_nucleus_fraction (int): Minimum nucleus fraction for erosion.
        gmm_n_init (int): Number of initializations for Gaussian Mixture Model.
    """

    DEFAULT_MIN_NUCLEUS_FRACTION: ClassVar[int] = 50
    DEFAULT_GMM_N_INIT: ClassVar[int] = 3

    cut_off_size: int
    noise: int
    connectivity: CellConnectivity
    min_nucleus_fraction: int = DEFAULT_MIN_NUCLEUS_FRACTION
    gmm_n_init: int = DEFAULT_GMM_N_INIT


def segment_cell(
    cells: LabeledCells, footprint: FootprintShape, config: SegmentationConfig
) -> List[NDArray]:
    """Segments cells using erosion, clustering, and noise removal.

    For each cell, if its size is above the cut_off_size, the function applies binary erosion,
    removes small objects, estimates the number of nuclei, and applies Gaussian Mixture Model clustering.
    Each resulting cluster is filtered for noise and relabeled. Small cells are returned as is.

    Args:
        cells (LabeledCells): LabeledCells object containing labeled cell regions.
        footprint (FootprintShape): Structuring element for erosion.
        config (SegmentationConfig): Configuration parameters for segmentation.

    Returns:
        List[NDArray]: List of segmented cell masks as 3D arrays.
    """
    cells_array = []

    for i in range(1, cells.len() + 1):
        if cells.get_cell_size(i) > config.cut_off_size:
            cell_matrix = cells.get_cell(i)

            eroded_matrix = apply_binary_erosion(cell_matrix, footprint)
            eroded_matrix = remove_small_objects(
                eroded_matrix, round(config.cut_off_size / config.min_nucleus_fraction)
            )

            number_of_nuclei = get_number_of_nuclei(eroded_matrix, config.connectivity)
            clusters = gaussian_mixture_predict(
                cell_matrix, number_of_nuclei, config.gmm_n_init
            )

            for cluster in clusters:
                cluster_filtered = remove_small_objects(cluster, config.noise)
                labeled_cluster = LabeledCells(
                    cluster_filtered, SkimageImgLabeling(config.connectivity)
                )

                for j in range(1, labeled_cluster.len() + 1):
                    cells_array.append(labeled_cluster.get_cell(j))
        else:
            cells_array.append(cells.get_cell(i))

    return cells_array
