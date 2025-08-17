import numpy as np
from .stepper import Stepper


class Simple(Stepper):
    """Stepper implementation that selects the lowest-valued neighbor.

    This class implements a simple stepping strategy for following
    the gradient in a distance map (2D or 3D). At each step, it
    searches in an expanding local neighborhood around a given
    start point and moves to the coordinate with the lowest
    distance-map value, if such a location exists.

    Attributes:
        distance_map (np.ndarray): The 2D or 3D distance map used
            to guide stepping.
    """
    def __init__(self, distance_map: np.ndarray) -> None:
        """Initializes the Simple stepper.

        Args:
            distance_map (np.ndarray): A 2D or 3D array representing the
                distance map. Each value corresponds to the cost or
                distance at that location.
        """       
        self.distance_map = distance_map

    def step(self, start_point: np.ndarray) -> np.ndarray:
        """
        Finds a lower pixel location in a local neighborhood.

        This function searches within an expanding local neighborhood around
        a given start point in a 2D or 3D volume and returns the coordinates
        of the location with the lowest value, if one exists.

        Args:
            start_point (np.ndarray): A 2D or 3D coordinate indicating the starting location in the volume.

        Returns:
            np.ndarray: The coordinates of the new location with lower value if found;
                        otherwise, returns the original start point.

        """
        assert start_point.ndim == 1, "Start point must be a 1D coordinate array."
        assert start_point.size in (2, 3), "Start point must be 2D or 3D."
        assert start_point.size == len(self.distance_map.shape), (
            "The coordinates should be for the same dimension as the volume"
        )

        start_point_rounded = np.round(start_point).astype(int)
        dims = self.distance_map.shape
        ndims = len(dims)

        for step_size in range(1, 4):
            lower_bounds = np.maximum(start_point_rounded - step_size, np.zeros(ndims))
            upper_bounds = np.minimum(start_point_rounded + step_size, dims)
            shape = tuple(ub - lb for lb, ub in zip(lower_bounds, upper_bounds))
            coordinates = np.indices(shape).reshape(len(shape), -1).T + np.array(
                lower_bounds
            )
            sub_volume = self.distance_map[tuple(coordinates)]
            center_value = self.distance_map[tuple(start_point_rounded)]
            mask = sub_volume < center_value
            if np.any(mask):
                idx = np.min(sub_volume)
                candidate_coordinates = coordinates[:, mask][:, idx]
                return candidate_coordinates.astype(int)
        return start_point_rounded
