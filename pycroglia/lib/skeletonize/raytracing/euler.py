import numpy as np
from scipy.ndimage import map_coordinates
from .stepper import Stepper


class Euler(Stepper):
    def __init__(self, step_size: float, gradient_volume: np.ndarray) -> None:
        self.step_size = step_size
        self.gradient_volume = gradient_volume

    def step(self, start_point: np.ndarray) -> np.ndarray:
        """
        Performs one Euler ray tracing step in a 2D or 3D gradient field.

        Args:
            start_point (np.ndarray): The starting position (2D or 3D float coordinates).

        Returns:
            np.ndarray: New point after the Euler step. Returns [0, 0] or [0, 0, 0] if out of bounds.

        """
        dim = start_point.size

        assert start_point.ndim == 1, "Start point must be a 1D coordinate array."
        assert dim in (2, 3), "Start point must be 2D or 3D."
        assert dim == len(self.gradient_volume.shape), (
            "The coordinates should be for the same dimension as the volume"
        )
        shape = self.gradient_volume.shape[:-1]

        if np.any(start_point < 0) or np.any(start_point >= np.array(shape)):
            return np.zeros(dim)  # Out of bounds

        # Prepare coordinates for interpolation: scipy uses (z, y, x) order
        coords = tuple(start_point[i] for i in range(dim))

        # Interpolate each gradient component at the start point
        gradient = np.array(
            [
                map_coordinates(
                    self.gradient_volume[..., i], [coords], order=1, mode="nearest"
                )[0]
                for i in range(dim)
            ]
        )
        norm = np.linalg.norm(gradient) + np.finfo(np.float64).eps
        gradient /= norm

        # Euler step in negative gradient direction
        new_point = start_point - self.step_size * gradient

        # Check bounds
        if np.any(new_point < 0) or np.any(new_point >= np.array(shape)):
            return np.zeros(dim)

        return new_point
