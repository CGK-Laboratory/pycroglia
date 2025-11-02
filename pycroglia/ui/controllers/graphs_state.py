from typing import Optional, List
from numpy.typing import NDArray

from PyQt6 import QtCore, QtWidgets


class DashboardGraphsGenerator(QtCore.QObject):
    @staticmethod
    def _make_default_graphs_list():
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
        super().__init__(parent=parent)

        # State
        self._img = img
        self._cells = cells

        # Configuration
        self._graphs_list = graphs_list or self._make_default_graphs_list()

    def get_graphs_list(self) -> List[str]:
        return self._graphs_list

    def generate_graphs(self, list_of_graphs: List[str]):
        pass
