from abc import ABC, abstractmethod
from numpy.typing import NDArray
from skimage import morphology


class FootprintShape(ABC):
    """Abstract base class for structuring element shapes."""

    @abstractmethod
    def get_shape(self) -> tuple:
        """Returns the structuring element shape.

        Returns:
            tuple: Structuring element for morphological operations.
        """
        pass


class DiamondFootprint(FootprintShape):
    """Diamond-shaped structuring element."""

    def __init__(self, r: int):
        """Initializes the diamond with a given radius.

        Args:
            r (int): Radius of the diamond.
        """
        self.r = r

    def get_shape(self) -> tuple:
        """Returns the diamond-shaped structuring element.

        Returns:
            tuple: Structuring element.
        """
        return morphology.diamond(radius=self.r)


class RectangleFootprint(FootprintShape):
    """Rectangle-shaped structuring element."""

    def __init__(self, x: int, y: int):
        """Initializes the rectangle with given dimensions.

        Args:
            x (int): Width of the rectangle.
            y (int): Height of the rectangle.
        """
        self.x = x
        self.y = y

    def get_shape(self) -> tuple:
        """Returns the rectangle-shaped structuring element.

        Returns:
            tuple: Structuring element.
        """
        return morphology.footprint_rectangle(shape=(self.y, self.x))


class DiskFootprint(FootprintShape):
    """Disk-shaped structuring element."""

    def __init__(self, r: int):
        """Initializes the disk with a given radius.

        Args:
            r (int): Radius of the disk.
        """
        self.r = r

    def get_shape(self) -> tuple:
        """Returns the disk-shaped structuring element.

        Returns:
            tuple: Structuring element.
        """
        return morphology.disk(radius=self.r)


def apply_binary_erosion(img: NDArray, footprint: FootprintShape) -> NDArray:
    """Applies binary erosion to an image using the given structuring element.

    Args:
        img (NDArray): Binary image.
        footprint (FootprintShape): Structuring element.

    Returns:
        NDArray: Eroded image.
    """
    return morphology.binary_erosion(img, footprint.get_shape())
