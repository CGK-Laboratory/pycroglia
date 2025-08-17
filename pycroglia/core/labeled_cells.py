import cv2
import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from numpy.typing import NDArray
from skimage import measure

from pycroglia.core.errors.errors import PycrogliaException


class LabelingStrategy(ABC):
    """Abstract base class for labeling strategies.

    Subclasses must implement the label method to generate labeled arrays from input images.

    Attributes:
        ARRAY_ELEMENTS_TYPE (type): Data type for output arrays.
    """

    ARRAY_ELEMENTS_TYPE = np.uint8

    @abstractmethod
    def label(self, img: NDArray) -> NDArray:
        """Labels the input image according to the strategy.

        Args:
            img (NDArray): Input image to label.

        Returns:
            NDArray: Labeled array.
        """
        pass


class SkimageCellConnectivity(Enum):
    """Defines connectivity options for labeling connected cell components in 3D images.

    Attributes:
        FACES (int): 6-connectivity (voxels connected by faces).
        EDGES (int): 18-connectivity (voxels connected by faces and edges).
        CORNERS (int): 26-connectivity (voxels connected by faces, edges, and corners).
    """

    FACES = 1
    EDGES = 2
    CORNERS = 3


class SkimageImgLabeling(LabelingStrategy):
    """Labeling strategy using skimage.measure.label.

    Attributes:
        connectivity (SkimageCellConnectivity): Connectivity rule for labeling.
    """

    def __init__(self, connectivity: SkimageCellConnectivity):
        """Initializes SkimageImgLabeling with the given connectivity.

        Args:
            connectivity (SkimageCellConnectivity): Connectivity rule for labeling.
        """
        self.connectivity = connectivity

    def label(self, img: NDArray) -> NDArray:
        """Labels the input image using skimage.measure.label.

        Args:
            img (NDArray): Input image to label.

        Returns:
            NDArray: Labeled array.
        """
        return measure.label(img, connectivity=self.connectivity.value)


class MaskListLabeling(LabelingStrategy):
    """Labeling strategy using a list of binary masks.

    Attributes:
        masks (list[NDArray]): List of binary masks.
        shape (tuple[int, int, int]): Shape of the output labeled array.
    """

    def __init__(self, masks: list[NDArray], shape: tuple[int, int, int]):
        """Initializes MaskListLabeling with masks and output shape.

        Args:
            masks (list[NDArray]): List of binary masks.
            shape (tuple[int, int, int]): Shape of the output labeled array.
        """
        self.masks = masks
        self.shape = shape

    def label(self, img: NDArray) -> NDArray:
        """Labels the input image using the provided masks.

        Args:
            img (NDArray): Input image to label (not used).

        Returns:
            NDArray: Labeled array.
        """
        labels = np.zeros(shape=self.shape, dtype=self.ARRAY_ELEMENTS_TYPE)
        for idx, mask in enumerate(self.masks, start=1):
            labels[mask > 0] = idx

        return labels


class LabeledCells:
    """Represents labeled connected cell components in a 3D image.

    Provides methods to access individual cells, their sizes, and 2D projections.

    Attributes:
        ARRAY_ELEMENTS_TYPE (type): Data type for output arrays.
        z (int): Depth of the image.
        y (int): Height of the image.
        x (int): Width of the image.
        labels (NDArray): Labeled 3D array.
    """

    ARRAY_ELEMENTS_TYPE = np.uint8

    def __init__(self, img: NDArray, labeling_strategy: LabelingStrategy):
        """Initializes LabeledCells with a 3D image and a labeling strategy.

        Args:
            img (NDArray): 3D binary image.
            labeling_strategy (LabelingStrategy): Strategy for labeling connected components.
        """
        self.x, self.y, self.z = img.shape
        self.labels = labeling_strategy.label(img)

        self._cell_sizes = np.bincount(self.labels.ravel())
        self._n_cells = len(self._cell_sizes) - 1

    def len(self) -> int:
        """Returns the number of labeled cells.

        Returns:
            int: Number of labeled cells (excluding background).
        """
        return self._n_cells

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

        return int(self._cell_sizes[index])

    def labels_to_2d(self) -> NDArray:
        return self.labels.max(axis=2)

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

        cell_matrix = np.zeros((self.x, self.y, self.z), dtype=self.ARRAY_ELEMENTS_TYPE)
        cell_matrix[self.labels == index] = 1
        flatten = cell_matrix.sum(axis=2)

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
