from abc import ABC, abstractmethod
import numpy as np


class Stepper(ABC):
    """Abstract base class for stepper strategies.

    A stepper defines how to move from a given starting point
    in a distance map (2D or 3D). Different subclasses
    implement different stepping strategies (e.g., Simple,
    Euler, Runge-Kutta).

    Subclasses must implement the :meth:`step` method.
    """

    @abstractmethod
    def step(self, start_point: np.ndarray) -> np.ndarray:
        """Compute the next point from a given starting location.

        Args:
            start_point (np.ndarray): A 1D coordinate array (2D or 3D)
                indicating the current location.

        Returns:
            np.ndarray: The next location as a coordinate array.
        """
        pass
