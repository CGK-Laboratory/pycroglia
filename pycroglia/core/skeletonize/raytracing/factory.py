from . import rk4, euler, simple, stepper
from enum import Enum
import numpy as np

class StepperType(Enum):
    RK4 = (0,)
    Euler = 1
    Simple = 2

def _point_min(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a normalized vector field pointing toward the local minimum direction
    in a 2D or 3D scalar field.

    Args:
        image (np.ndarray): Input 2D or 3D scalar field.

    Returns:
        tuple: A tuple of NumPy arrays representing the vector field components (fy, fx) for 2D,
        or (fy, fx, fz) for 3D.
    """
    assert image.ndim in (2, 3), "The image should be 2D or 3D"
    fx, fy, fz = (
        np.zeros_like(image, dtype=np.float64),
        np.zeros_like(image, dtype=np.float64),
        np.zeros_like(image, dtype=np.float64) if image.ndim == 3 else None,
    )
    padded_image = np.pad(image, 1, mode="constant", constant_values=image.max())

    # Define neighbor shifts
    if image.ndim == 2:
        neighbors = [d for d in product([-1, 0, 1], repeat=2) if d != (0, 0)]
    else:
        neighbors = [d for d in product([-1, 0, 1], repeat=3) if d != (0, 0, 0)]

    for dx, dy, *dz in neighbors:
        if image.ndim == 2:
            shifted = padded_image[
                1 + dx : 1 + dx + image.shape[0], 1 + dy : 1 + dy + image.shape[1]
            ]
            mask = shifted < image
            norm = np.sqrt(dx**2 + dy**2)
            fx[mask] = dx / norm
            fy[mask] = dy / norm
        else:
            dz = dz[0]
            shifted = padded_image[
                1 + dx : 1 + dx + image.shape[0],
                1 + dy : 1 + dy + image.shape[1],
                1 + dz : 1 + dz + image.shape[2],
            ]
            mask = shifted < image
            norm = np.sqrt(dx**2 + dy**2 + dz**2)
            fx[mask] = dx / norm
            fy[mask] = dy / norm
            if fz:
                fz[mask] = dz / norm

    return (fy, fx, fz) if image.ndim == 3 and fz else (fy, fx)
    
    
def make_stepper(
    distance_map: np.ndarray,
    type: StepperType = StepperType.RK4,
    step_size: float = 0.5,
) -> stepper.Stepper:
    """
        Create a stepper object for shortest-path tracing based on the specified method.

    Args:
        distance_map (np.ndarray): A 2D or 3D array representing the distance field.
        type (StepperType): The integration method to use. Options are:
            - StepperType.RK4: 4th-order Runge-Kutta integration.
            - StepperType.Euler: First-order Euler integration.
            - StepperType.Simple: Discrete descent using local minima.
        step_size (float): The integration step size for RK4 or Euler methods.

    Returns:
        Stepper: A stepper object (RK4, Euler, or Simple) with a `.step(point, field, step_size)` method.
    """
    assert distance_map.ndim in (2, 3), "Distance map should be 2D or 3D"
    f = _point_min(distance_map)
    if len(f) == 2:
        fy, fx = f
        gradient_volume = np.stack([-fx, -fy], axis=-1)
    elif len(f) == 3:
        fy, fx, fz = f
        gradient_volume = np.stack([-fx, -fy, fz], axis=-1)
    else:
        assert False, "Shouldn't happen"

    if type == StepperType.RK4:
        return rk4.RK4(step_size, gradient_volume)
    elif type == StepperType.Euler:
        return euler.Euler(step_size, gradient_volume)
    elif type == StepperType.Simple:
        return simple.Simple(distance_map)
