from typing import Optional, List
from numpy.typing import NDArray

from PyQt6 import QtCore, QtWidgets


class DashboardGraphsGenerator(QtCore.QObject):
    """Utility to provide available graph names and generate requested graphs.

    This lightweight controller holds a reference to the image and cell masks
    and exposes a list of available graph types. The generate_graphs method is
    intended to be implemented to create or display the requested graphs.

    Attributes:
        _img (NDArray): Source image used for graph generation.
        _cells (List[NDArray]): Per-cell masks used by some graph algorithms.
        _graphs_list (List[str]): Names of graphs available for preview/generation.
    """

    @staticmethod
    def _make_default_graphs_list():
        """Return a default list of graph names.

        Returns:
            List[str]: Default graph names provided by the dashboard.
        """
        return [
            "Convex cells Images",
            "Skeleton Image",
            "Original Cell Image",
            "End Image",
            "Branches Image",
        ]

    def __init__(
        self,
        img: NDArray,
        cells: List[NDArray],
        graphs_list: Optional[List[str]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the graphs generator.

        Args:
            img (NDArray): Source image array.
            cells (List[NDArray]): Per-cell masks or arrays.
            graphs_list (Optional[List[str]]): Optional override list of graph names.
            parent (Optional[QtWidgets.QWidget]): Optional Qt parent.
        """
        super().__init__(parent=parent)

        # State
        self._img = img
        self._cells = cells

        # Configuration
        self._graphs_list = graphs_list or self._make_default_graphs_list()

    def get_graphs_list(self) -> List[str]:
        """Return the list of available graph names.

        Returns:
            List[str]: List of graph names suitable for the selector widget.
        """
        return self._graphs_list

    def generate_graphs(self, list_of_graphs: List[str]):
        """Generate or display the requested graphs.

        This method is a placeholder and should perform the actual graph
        generation or dispatch to the visualization layer.

        Args:
            list_of_graphs (List[str]): Names of graphs requested.

        Returns:
            None
        """
        pass
