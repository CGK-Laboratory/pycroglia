import numpy as np
from scipy.ndimage import map_coordinates
from .stepper import Stepper


def _interpolate(field: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Interpolation of a 2D or 3D vector field at a given point.

    Args:
        field (np.ndarray): Vector field
        point (np.ndarray): Coordinate to interpolate at (x, y) or (x, y, z).

    Returns:
        np.ndarray: Interpolated vector at point.
    """
    point = np.asarray(point, dtype=np.float64)
    mode = "nearest"
    assert point.size in (2, 3), f"Point should be 2D or 3D, point={point}"
    assert field.ndim in (3, 4), "The field should be 2D or 3D"
    assert field.shape[-1] in (2, 3), "The field vectors be 2D or 3D"
    if point.size == 2 and field.ndim == 3 and field.shape[2] == 2:
        coords = [np.array([point[1]]), np.array([point[0]])]  # y, x
        return np.stack(
            [
                map_coordinates(
                    field[..., 0], coords, order=1, mode=mode, prefilter=False
                )[0],
                map_coordinates(
                    field[..., 1], coords, order=1, mode=mode, prefilter=False
                )[0],
            ]
        )
    elif point.size == 3 and field.ndim == 4 and field.shape[3] == 3:
        coords = [
            np.array([point[2]]),
            np.array([point[1]]),
            np.array([point[0]]),
        ]  # z, y, x
        return np.stack(
            [
                map_coordinates(
                    field[..., 0], coords, order=1, mode=mode, prefilter=False
                )[0],
                map_coordinates(
                    field[..., 1], coords, order=1, mode=mode, prefilter=False
                )[0],
                map_coordinates(
                    field[..., 2], coords, order=1, mode=mode, prefilter=False
                )[0],
            ]
        )
    else:
        raise ValueError("unexpected value")


class RK4(Stepper):
    """Runge-Kutta 4 (RK4) stepper for gradient-based tracing.

    This stepper integrates along the negative gradient of a vector field
    using the fourth-order Runge-Kutta method. It provides more accurate
    tracing than Euler integration at the cost of additional interpolation.

    Attributes:
        step_size (float): Integration step size.
        gradient_volume (np.ndarray): Vector field with shape (..., dim),
        where dim is 2 or 3.
    """

    def __init__(self, step_size: float, gradient_volume: np.ndarray) -> None:
        """Initialize the RK4 stepper.

        Args:
            step_size (float): Integration step size.
            gradient_volume (np.ndarray): Gradient field with shape (..., dim).
        """
        self.step_size = step_size
        self.gradient_volume = gradient_volume

    def step(self, start_point: np.ndarray) -> np.ndarray:
        """
        Efficient single RK4 step in a 2D or 3D vector field.

        Args:
            start_point (np.ndarray): Start coordinate (x, y) or (x, y, z).
            gradient_volume (np.ndarray): Vector field (H, W, 2) or (D, H, W, 3).
            step_size (float): Step size for integration.

        Returns:
            np.ndarray: New point after RK4 step, or zeros if outside domain.
        """
        shape = self.gradient_volume.shape[:-1]

        def inside(p):
            return np.all((p >= 0) & (p < np.array(shape)))

        k1 = _interpolate(self.gradient_volume, start_point)
        mid1 = start_point + 0.5 * self.step_size * k1
        if not inside(mid1):
            return np.zeros_like(start_point)

        k2 = _interpolate(self.gradient_volume, mid1)
        mid2 = start_point + 0.5 * self.step_size * k2
        if not inside(mid2):
            return np.zeros_like(start_point)

        k3 = _interpolate(self.gradient_volume, mid2)
        end = start_point + self.step_size * k3
        if not inside(end):
            return np.zeros_like(start_point)

        k4 = _interpolate(self.gradient_volume, end)

        result = start_point + (self.step_size / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return result if inside(result) else np.zeros_like(result)
