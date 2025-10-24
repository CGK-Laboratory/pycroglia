from abc import ABC, abstractmethod
from typing import Any


class Computable(ABC):
    """Abstract base class for all computable objects.

    Classes inheriting from `Computable` must implement the `compute` method.
    The `compute` method executes a specific computation and returns its
    results as a dictionary, allowing flexible storage of heterogeneous values.
    """

    @abstractmethod
    def compute(self) -> dict[str, Any]:
        """Execute the computation and return its results.

        Returns:
            dict[str, Any]: A dictionary containing the results of the
            computation. Keys should be descriptive strings (e.g., "cell_id",
            "volume", "centroid_distance"), and values can be of any type
            depending on the computation.
        """
        pass
