import numpy as np

from enum import Enum
from numpy.typing import NDArray
from skimage import measure

from pycroglia.core.errors.errors import PycrogliaException


class CellConnectivity(Enum):
    """Defines connectivity options for labeling connected cell components in 3D images.

    Attributes:
        FACES (int): 6-connectivity (voxels connected by faces).
        EDGES (int): 18-connectivity (voxels connected by faces and edges).
        CORNERS (int): 26-connectivity (voxels connected by faces, edges, and corners).
    """

    FACES = 1
    EDGES = 2
    CORNERS = 3


def label_cells(img: NDArray, connectivity: CellConnectivity) -> NDArray:
    """Assigns unique integer labels to each connected cell in a 3D binary image.

    Args:
        img (NDArray): 3D binary array where nonzero values represent cell candidates.
        connectivity (CellConnectivity): Connectivity rule for defining cell neighborhoods.

    Returns:
        NDArray: 3D array with the same shape as `img`, where each connected cell has a unique integer label (0 is background).
    """
    labeled_cells = measure.label(img, connectivity=connectivity.value)
    return labeled_cells


class LabeledCells:
    """Represents labeled connected cell components in a 3D image.

    Provides methods to access individual cells, their sizes, and 2D projections.

    Attributes:
        ARRAY_ELEMENTS_TYPE (type): Data type for output arrays.
        z (int): Depth of the image.
        y (int): Height of the image.
        x (int): Width of the image.
        connectivity (CellConnectivity): Connectivity used for labeling.
        labels (NDArray): Labeled 3D array.
    """

    ARRAY_ELEMENTS_TYPE = np.uint8

    def __init__(self, img: NDArray, connectivity: CellConnectivity):
        """Initializes LabeledCells with a 3D image and connectivity.

        Args:
            img (NDArray): 3D binary image.
            connectivity (CellConnectivity): Connectivity rule for labeling.
        """
        self.z, self.y, self.x = img.shape
        self.connectivity = connectivity
        self.labels = label_cells(img, connectivity)

    def len(self) -> int:
        """Returns the number of labeled cells.

        Returns:
            int: Number of labeled cells (excluding background).
        """
        return self.labels.max()

    def _is_valid_index(self, index: int) -> bool:
        """Checks if the given index is a valid cell label.

        Args:
            index (int): Cell label index.

        Returns:
            bool: True if valid, False otherwise.
        """
        return 0 < index <= self.len()

    def get_cell(self, index: int) -> NDArray:
        """Returns a binary mask for the specified cell.

        Args:
            index (int): Cell label index.

        Returns:
            NDArray: 3D binary mask for the cell.

        Raises:
            PycrogliaException: If the index is invalid (error_code=2000).
        """
        if not self._is_valid_index(index):
            raise PycrogliaException(error_code=2000)

        return (self.labels == index).astype(self.ARRAY_ELEMENTS_TYPE)

    def get_cell_size(self, index: int) -> int:
        """Returns the size (number of voxels) of the specified cell.

        Args:
            index (int): Cell label index.

        Returns:
            int: Number of voxels in the cell.

        Raises:
            PycrogliaException: If the index is invalid (error_code=2000).
        """
        if not self._is_valid_index(index):
            raise PycrogliaException(error_code=2000)

        return np.sum(self.labels == index)

    def cell_to_2d(self, index: int) -> NDArray:
        """Projects a 3D cell to 2D by summing along the z-axis.

        Args:
            index (int): Cell label index.

        Returns:
            NDArray: 2D projection of the cell.

        Raises:
            PycrogliaException: If the index is invalid (error_code=2000).
        """
        if not self._is_valid_index(index):
            raise PycrogliaException(error_code=2000)

        cell_matrix = np.zeros((self.z, self.y, self.x), dtype=self.ARRAY_ELEMENTS_TYPE)
        cell_matrix[self.labels == index] = 1
        flatten = cell_matrix.sum(axis=0)

        return flatten

    def all_cells_to_2d(self) -> NDArray:
        """Projects all labeled cells to 2D and stacks them along a new axis.

        Returns:
            NDArray: 3D array where each slice is the 2D projection of a cell.
        """
        all_cells_matrix = np.zeros(
            (self.len(), self.y, self.x), dtype=self.ARRAY_ELEMENTS_TYPE
        )

        for i in range(1, self.len() + 1):
            cell_array = self.cell_to_2d(i)
            all_cells_matrix[i - 1, :, :] = cell_array

        return all_cells_matrix

    def overlap_cells(self) -> NDArray:
        """Computes the overlap image by summing all 2D projections of cells.

        Returns:
            NDArray: 2D array representing the overlap of all cells.
        """
        all_cells = self.all_cells_to_2d()
        overlap_img = all_cells.sum(axis=0)
        return overlap_img
