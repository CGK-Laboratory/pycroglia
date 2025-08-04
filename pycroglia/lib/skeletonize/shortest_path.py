from .raytracing import stepper
import numpy as np


class ShortestPath:
    """
    Traces the shortest path in a 2D or 3D distance map using numerical integration.

    This class supports path tracing via numerical integration methods such as:
    - Runge-Kutta 4th order (RK4)
    - Euler's method
    - Simple gradient descent

    The tracing starts from a given point and optionally stops early if a specified
    target (source) point is reached within a threshold distance.

    Attributes:
        type (StepperType): The integration method to use.
        step_size (float): The step size used in the integration process.
    """

    def __init__(
        self,
        stepper_type: stepper.StepperType = stepper.StepperType.RK4,
        step_size: float = 0.5,
    ):
        """
        Initializes the ShortestPath tracer.

        Args:
            stepper_type (StepperType, optional): The numerical integration method to use.
                Defaults to StepperType.RK4.
            step_size (float, optional): Step size used in each iteration. Defaults to 0.5.
        """

        self.type = stepper_type
        self.step_size = step_size

    def calculate(
        self,
        distance_map: np.ndarray,
        start_point: np.ndarray,
        source_point: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Traces the shortest path from a starting point in a 2D or 3D distance map.

        This function iteratively follows the gradient of the distance map, using
        the selected stepping method to trace a path. If a `source_point` is given,
        the path terminates early if it gets sufficiently close to any source point.

        Args:
            distance_map (np.ndarray): A 2D or 3D distance field array.
            start_point (np.ndarray): A point `[x, y]` or `[x, y, z]` indicating where to start the path.
            source_point (np.ndarray | None, optional): An array of shape `(N, D)` where
                each row is a source point in the same dimension as `start_point`. If not provided,
                the path will be traced until it reaches a boundary or stagnates.

        Returns:
            tuple[np.ndarray, bool]:
                - A NumPy array of shape `(M, D)` representing the traced shortest path.
                  Each row is a point along the path.
                - A boolean flag indicating whether the path **failed to reach** a source point.
                  If `True`, it did **not** reach any source point (e.g. due to early stop or divergence).

        """
        assert distance_map.ndim in (2, 3), "Distance map should be 2D or 3D"
        s = stepper.make_stepper(distance_map, self.type, self.step_size)
        max_length = 10000

        ndim = distance_map.ndim
        path = np.zeros((max_length, ndim), dtype=np.float64)

        current_point = start_point.astype(np.float64)
        i = 0

        while True:
            # Step forward using the given integration method
            current_point = s.step(current_point)
            # Distance to nearest source point
            if source_point:
                deltas = source_point - current_point  # shape (N, D)
                distances = np.linalg.norm(deltas, axis=1)
                distance_to_source = np.min(distances)
                nearest_source = source_point[np.argmin(distances)]
            else:
                distance_to_source = np.inf
                nearest_source = None

            # Movement compared to 10 steps ago
            if i >= 10:
                delta = current_point - path[i - 10]
                movement = np.linalg.norm(delta)
            else:
                movement = self.step_size + 1  # ensure not stopping too early

            # Termination conditions
            out_of_bounds = np.any(current_point < 0) or np.any(
                current_point >= distance_map.shape
            )
            if out_of_bounds or movement < self.step_size:
                break

            # Store current point
            if i >= max_length:
                max_length += 10000
                path = np.vstack([path, np.zeros((10000, ndim), dtype=np.float64)])

            path[i] = current_point
            i += 1

            # Stop if we're close enough to a source point
            if nearest_source and distance_to_source < self.step_size:
                if i >= max_length:
                    max_length += 10000
                    path = np.vstack([path, np.zeros((10000, ndim), dtype=np.float64)])
                path[i] = nearest_source
                i += 1
                break

        path_reached_source_point = not (distance_to_source > 1 and source_point)

        return (path[:i], not path_reached_source_point)
