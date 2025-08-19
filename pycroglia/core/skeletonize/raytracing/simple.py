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

    def step(self, start_point: np.ndarray) -> np.ndarray | None:
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
        assert start_point.size in (2, 3), "Start point must be 2D or 3D."
        assert start_point.size == len(self.distance_map.shape), (
            "The coordinates should be for the same dimension as the volume"
        )


        s = np.round(start_point).astype(int)
        end_point = s.copy()
        dims = s.size
        check = False

        if dims == 2:
            for step_size in range(1, 4):
                sxm = max(s[0] - step_size, 0)
                sym = max(s[1] - step_size, 0)
                sxp = min(s[0] + step_size, self.distance_map.shape[0] - 1)
                syp = min(s[1] + step_size, self.distance_map.shape[1] - 1)

                x = np.arange(sxm, sxp + 1)
                y = np.arange(sym, syp + 1)

                sub_volume = self.distance_map[sxm:sxp+1, sym:syp+1]
                c_volume = sub_volume < self.distance_map[s[0], s[1]]
                check = np.any(c_volume)
                if check:
                    ind = np.argmin(sub_volume)
                    i, j = np.unravel_index(ind, sub_volume.shape)
                    end_point = np.array([x[i], y[j]])
                    
                elif dims == 3:
                    for step_size in range(1, 4):
                        sxm = max(s[0] - step_size, 0)
                        sym = max(s[1] - step_size, 0)
                        szm = max(s[2] - step_size, 0)

                        sxp = min(s[0] + step_size, self.distance_map.shape[0] - 1)
                        syp = min(s[1] + step_size, self.distance_map.shape[1] - 1)
                        szp = min(s[2] + step_size, self.distance_map.shape[2] - 1)

                        x = np.arange(sxm, sxp + 1)
                        y = np.arange(sym, syp + 1)
                        z = np.arange(szm, szp + 1)

                        sub_volume = self.distance_map[sxm:sxp+1, sym:syp+1, szm:szp+1]
                        c_volume = sub_volume < self.distance_map[s[0], s[1], s[2]]
                        check = np.any(c_volume)
                        if check:
                            break

                        if check:
                            ind = np.argmin(sub_volume)
                            i, j, k = np.unravel_index(ind, sub_volume.shape)
                            end_point = np.array([x[i], y[j], z[k]])
                            
            return end_point
    
