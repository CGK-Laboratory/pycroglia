from typing import Any, Optional, List
from numpy.typing import NDArray

from PyQt6 import QtCore, QtWidgets
from pycroglia.core.plot.branch import BranchpointsCellPlot
from pycroglia.core.plot.endpoints import EndpointsCellPlot

from pycroglia.core.plot.full_cell_analysis import FullCellAnalysisPlot
from pycroglia.core.plot.original_cell import OriginalCellPlot
from pycroglia.core.plot.skeleton import SkeletonCellPlot


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
        return self._graphs_list

    def generate_plots(
        self,
        selected_plots: list[str],
        masks: list[NDArray],
        branch_analysis: dict[int, Any],
        full_cell_analysis: dict[str, Any],
        scale: float,
        zscale: float,
    ):
        global_plot_map = {
            "Convex cells Images": lambda: FullCellAnalysisPlot(
                full_cell_analysis, masks
            ),
            "Original Cell Image": lambda: OriginalCellPlot(masks, scale, zscale),
        }

        per_cell_plot_map = {
            "Skeleton Image": lambda i: SkeletonCellPlot(
                i + 1, branch_analysis[i]["fullmasks"], scale, zscale
            ),
            "End Image": lambda i: EndpointsCellPlot(
                i + 1,
                branch_analysis[i]["fullmasks"],
                branch_analysis[i]["endpoints"],
                scale,
                zscale,
            ),
            "Branches Image": lambda i: BranchpointsCellPlot(
                i + 1,
                branch_analysis[i]["fullmasks"],
                branch_analysis[i]["allbranch"],
                scale,
                zscale,
            ),
        }

        for plot_name, plot_factory in global_plot_map.items():
            if plot_name in selected_plots:
                plotter = plot_factory()
                plotter.show_all()

        for i in range(len(masks)):
            for plot_name, plot_factory in per_cell_plot_map.items():
                if plot_name in selected_plots:
                    plotter = plot_factory(i)
                    plotter.show_all()
