import numpy as np

from enum import Enum
from numpy.typing import NDArray
from skimage import measure

from pycroglia.core.errors.errors import PycrogliaException


class ComponentConnectivity(Enum):
    """Enumeration for 3D connected component connectivity.

    Attributes:
        FACES (int): 6-connectivity in 3D (faces).
        EDGES (int): 18-connectivity in 3D (edges).
        CORNERS (int): 26-connectivity in 3D (corners).
    """

    FACES = 1
    EDGES = 2
    CORNERS = 3


def get_connected_components(
    img: NDArray, connectivity: ComponentConnectivity
) -> NDArray:
    """Labels connected components in a 3D image using the specified connectivity.

    Args:
        img (NDArray): 3D binary image array.
        connectivity (ComponentConnectivity): Connectivity type for labeling.

    Returns:
        NDArray: Labeled image where each connected component has a unique integer label.
    """
    labeled_components = measure.label(img, connectivity=connectivity.value)
    return labeled_components


class ConnectedComponents:
    """Class for handling connected components in a 3D image.

    Attributes:
        ARRAY_ELEMENTS_TYPE (type): Data type for output arrays.
        z (int): Number of slices in the z dimension.
        y (int): Height of the image.
        x (int): Width of the image.
        labels (NDArray): Labeled image array.
    """

    ARRAY_ELEMENTS_TYPE = np.uint8

    def __init__(self, img: NDArray, connectivity: ComponentConnectivity):
        """Initializes the ConnectedComponents object.

        Args:
            img (NDArray): 3D binary image array.
            connectivity (ComponentConnectivity): Connectivity type for labeling.
        """
        self.z, self.y, self.x = img.shape
        self.labels = get_connected_components(img, connectivity)

    def len(self) -> int:
        """Returns the number of connected components.

        Returns:
            int: Number of connected components (maximum label value).
        """
        return self.labels.max()

    def component_to_2d(self, index: int) -> NDArray:
        """Projects a single connected component to a 2D array by summing along the z-axis.

        Args:
            index (int): Label index of the component (must be > 0 and <= number of components).

        Returns:
            NDArray: 2D array representing the projection of the component.

        Raises:
            PycrogliaException: If the index is out of valid range.
        """
        if index <= 0 or index > self.len():
            raise PycrogliaException(error_code=2000)

        component_matrix = np.zeros(
            (self.z, self.y, self.x), dtype=self.ARRAY_ELEMENTS_TYPE
        )
        component_matrix[self.labels == index] = 1
        flatten = component_matrix.sum(axis=0)

        return flatten

    def all_components_to_2d(self) -> NDArray:
        """Projects all connected components to 2D arrays.

        Returns:
            NDArray: 3D array of shape (num_components, y, x), where each slice is a 2D projection of a component.
        """
        all_components = np.zeros(
            (self.len(), self.y, self.x), dtype=self.ARRAY_ELEMENTS_TYPE
        )
        for i in range(1, self.len() + 1):
            component_array = self.component_to_2d(i)
            all_components[i - 1, :, :] = component_array

        return all_components

    def overlap_components(self) -> NDArray:
        """Computes the overlap image of all projected components.

        Returns:
            NDArray: 2D array where each pixel value is the sum of overlapping components at that location.
        """
        all_components = self.all_components_to_2d()
        overlap_img = all_components.sum(axis=0)
        return overlap_img
