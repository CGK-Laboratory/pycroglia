from abc import ABC, abstractmethod
from numpy.typing import NDArray
from skimage import morphology
import numpy as np


class FootprintShape(ABC):
    """Abstract base class for structuring element shapes."""

    @abstractmethod
    def get_shape(self) -> tuple:
        """Returns the structuring element shape.

        Returns:
            tuple: Structuring element for morphological operations.
        """
        pass


class Diamond2DFootprint(FootprintShape):
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


class Rectangle2DFootprint(FootprintShape):
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


class Disk2DFootprint(FootprintShape):
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


class Octahedron3DFootprint(FootprintShape):
    """Octahedron-shaped structuring element in 3D."""

    def __init__(self, r: int):
        """Initializes the octahedron with a given radius.

        Args:
            r (int): Radius of the octahedron.
        """
        self.r = r

    def get_shape(self) -> tuple:
        """Returns the octahedron-shaped structuring element.

        Returns:
            tuple: Structuring element.
        """
        return morphology.octahedron(radius=self.r)


class Ball3DFootprint(FootprintShape):
    """Ball-shaped structuring element in 3D."""

    def __init__(self, r: int):
        """Initializes the ball with a given radius.

        Args:
            r (int): Radius of the ball.
        """
        self.r = r

    def get_shape(self) -> tuple:
        """Returns the ball-shaped structuring element.

        Returns:
            tuple: Structuring element.
        """
        return morphology.ball(radius=self.r)


class Rectangle3DFootprint(FootprintShape):
    """3D rectangular structuring element for binary erosion.

    Args:
        x (int): Size along the x-axis.
        y (int): Size along the y-axis.
        z (int): Size along the z-axis.

    Attributes:
        x (int): Size along the x-axis.
        y (int): Size along the y-axis.
        z (int): Size along the z-axis.
    """

    def __init__(self, x: int, y: int, z: int):
        """Initializes the rectangle with given dimensions.

        Args:
            x (int): Size along the x-axis.
            y (int): Size along the y-axis.
            z (int): Size along the z-axis.
        """
        self.x = x
        self.y = y
        self.z = z

    def get_shape(self) -> NDArray:
        """Returns a 3D rectangular footprint.

        Returns:
            NDArray: 3D array of ones with shape (z*2+1, y*2+1, x*2+1).
        """
        return np.ones((self.z * 2 + 1, self.y * 2 + 1, self.x * 2 + 1), dtype=bool)


def apply_binary_erosion(img: NDArray, footprint: FootprintShape) -> NDArray:
    """Applies binary erosion to an image using the given structuring element.

    Args:
        img (NDArray): Binary image.
        footprint (FootprintShape): Structuring element.

    Returns:
        NDArray: Eroded image.
    """
    return morphology.binary_erosion(img, footprint.get_shape())
