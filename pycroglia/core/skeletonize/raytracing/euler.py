import numpy as np
from scipy.ndimage import map_coordinates
from .stepper import Stepper


class Euler(Stepper):
    """Euler integration stepper for gradient-based tracing.

    This stepper implements the explicit Euler method to trace
    along the negative gradient of a distance map or potential field.
    It works in both 2D and 3D volumes.

    Attributes:
        step_size (float): The integration step size.
        gradient_volume (np.ndarray): The gradient field of the distance map.
            Shape is (..., dim), where dim is 2 or 3.
    """

    def __init__(self, step_size: float, gradient_volume: np.ndarray) -> None:
        self.step_size = step_size
        self.gradient_volume = gradient_volume

    def step(self, start_point: np.ndarray) -> np.ndarray | None:
        """
        Performs one Euler ray tracing step in a 2D or 3D gradient field.

        Args:
            start_point (np.ndarray): The starting position (2D or 3D float coordinates).

        Returns:
            np.ndarray: New point after the Euler step. Returns [0, 0] or [0, 0, 0] if out of bounds.

        """
        dim = start_point.size
        assert dim in (2, 3), "Start point must be 2D or 3D."
        assert dim == self.gradient_volume.shape[-1], "Coordinate dimension mismatch."

        shape = self.gradient_volume.shape[:-1]

        # Out of bounds check
        if np.any(start_point < 0) or np.any(start_point >= np.array(shape)):
            return None

        # Prepare coordinates for map_coordinates: shape (ndim, n_points)
        coords = np.array(start_point, dtype=np.float64).reshape(dim, 1)

        # Interpolate each gradient component at the point
        gradient = np.zeros(dim)
        for i in range(dim):
            gradient[i] = map_coordinates(
                self.gradient_volume[..., i], coords, order=1, mode="nearest"
            )[0]

        norm = np.linalg.norm(gradient) + np.finfo(np.float64).eps
        gradient /= norm

        # Euler step
        new_point = start_point + self.step_size * gradient

        # Check bounds
        if np.any(new_point < 0) or np.any(new_point >= np.array(shape)):
            return None

        return new_point
