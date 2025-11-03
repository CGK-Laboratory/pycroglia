from typing import List, Optional, Protocol
from numpy.typing import NDArray

from PyQt6 import QtCore, QtWidgets

from pycroglia.core.io.output import (
    AnalysisSummary,
    AnalysisSummaryConfig,
    CellAnalysis,
    CellAnalysisConfig,
)


class SummaryAnalysisCalculator(Protocol):
    """Protocol for objects that provide summary analysis data."""

    def get_analysis_data(self) -> AnalysisSummary:
        """Return analysis summary data.

        Returns:
            AnalysisSummary: Aggregated analysis metrics.
        """
        pass

    def get_analysis_data_config(self) -> AnalysisSummaryConfig:
        """Return configuration/labels for the analysis summary.

        Returns:
            AnalysisSummaryConfig: Labels and text used in the summary view.
        """
        pass


class CellAnalysisCalculator(Protocol):
    """Protocol for objects that provide per-cell analysis data."""

    def get_cells_data(self) -> List[CellAnalysis]:
        """Return a list of per-cell analysis results.

        Returns:
            List[CellAnalysis]: Per-cell analysis objects.
        """
        pass

    def get_cells_data_config(self) -> CellAnalysisConfig:
        """Return configuration/labels for per-cell data.

        Returns:
            CellAnalysisConfig: Labels and text used in the per-cell view.
        """
        pass


class CellGraphGenerator(Protocol):
    """Protocol for objects that can supply and generate graphs for cells."""

    def get_graphs_list(self) -> List[str]:
        """Return available graph names.

        Returns:
            List[str]: Names of available graphs that can be generated/previewed.
        """
        pass

    def generate_graphs(self, graphs_list: List[str]):
        """Generate or display the requested graphs.

        Args:
            graphs_list (List[str]): Names of graphs to generate or display.
        """
        pass


class ResultsProvider(
    SummaryAnalysisCalculator, CellAnalysisCalculator, CellGraphGenerator, Protocol
):
    """Aggregate protocol that provides all results-related data and graph functionality."""

    pass


class ResultsDashboardState(QtCore.QObject):
    """Lightweight state object providing placeholder results for demos and tests.

    This object implements the ResultsProvider protocol and returns simple,
    deterministic objects that satisfy the UI's expectations. It also acts as
    a Qt object so it can be parented to widgets if needed.
    """

    def __init__(self, img: NDArray, parent: Optional[QtWidgets.QWidget] = None):
        """Initialize the ResultsDashboardState.

        Args:
            img (NDArray): Input image or placeholder array for which results relate.
            parent (Optional[QtWidgets.QWidget]): Optional parent widget.
        """
        super().__init__(parent=parent)

        self._img = img

    def add_parent(self, parent: QtWidgets.QWidget):
        """Set a QWidget as the Qt parent of this state object.

        Args:
            parent (QtWidgets.QWidget): Widget to set as parent.
        """
        self.setParent(parent)

    def get_analysis_data(self) -> AnalysisSummary:
        """Return a demo AnalysisSummary with default values.

        Returns:
            AnalysisSummary: Demo analysis summary.
        """
        return AnalysisSummary(
            avg_centroid_distance=0,
            total_territorial_volume=0,
            total_unoccupied_volume=0,
            percent_occupied_volume=0,
            file="example.txt",
        )

    def get_analysis_data_config(self) -> AnalysisSummaryConfig:
        """Return a demo AnalysisSummaryConfig with label text.

        Returns:
            AnalysisSummaryConfig: Labels and text for the analysis summary.
        """
        return AnalysisSummaryConfig(
            file_txt="File",
            avg_centroid_distance_txt="Avg centroid distance",
            total_territorial_volume_txt="Total territorial volume",
            total_unoccupied_volume_txt="Total unoccupied volume",
            percent_occupied_volume_txt="Percent occupied volume",
        )

    def get_cells_data(self) -> List[CellAnalysis]:
        """Return a demo list with a single CellAnalysis.

        Returns:
            List[CellAnalysis]: Demo per-cell analysis data.
        """
        return [
            CellAnalysis(
                cell_territory_volume=0,
                cell_volume=0,
                ramification_index=0,
                number_of_endpoints=0,
                number_of_branches=0,
                avg_branch_length=0,
                max_branch_length=0,
                min_branch_length=0,
            )
        ]

    def get_cells_data_config(self) -> CellAnalysisConfig:
        """Return a demo CellAnalysisConfig with label text.

        Returns:
            CellAnalysisConfig: Labels and text for per-cell metrics.
        """
        return CellAnalysisConfig(
            cell_territory_volume_txt="Cell territory volume",
            cell_volume_txt="Cell volume",
            ramification_index_txt="Ramification index",
            number_of_endpoints_txt="Number of endpoints",
            number_of_branches_txt="Number of branches",
            avg_branch_length_txt="Avg branch length",
            max_branch_length_txt="Max branch length",
            min_branch_length_txt="Min branch length",
        )

    def get_graphs_list(self) -> List[str]:
        """Return available graph names.

        Returns:
            List[str]: Names of demo graphs.
        """
        return [
            "Convex cells Images",
            "Skeleton Image",
            "Original Cell Image",
            "End Image",
            "Branches Image",
        ]

    def generate_graphs(self, graphs_list: List[str]):
        """Placeholder for graph generation logic.

        Implementers should generate or display the requested graphs. This
        demo implementation is intentionally empty.

        Args:
            graphs_list (List[str]): Names of graphs to generate or display.
        """
        pass
