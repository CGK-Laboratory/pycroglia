from typing import Optional, List

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QVBoxLayout

from pycroglia.core.io.output import (
    AnalysisSummary,
    AnalysisSummaryConfig,
    CellAnalysis,
    CellAnalysisConfig,
)
from pycroglia.ui.widgets.common.two_column_list import TwoColumnList


class AnalysisSummaryViewer(QtWidgets.QWidget):
    """Widget to display an analysis summary in a two-column list.

    Attributes:
        headers (List[str]): Table headers.
        data (AnalysisSummary): Analysis summary data.
        data_config (AnalysisSummaryConfig): Configuration for labels.
        list (TwoColumnList): Widget displaying the summary.
    """

    def __init__(
        self,
        headers: List[str],
        data: AnalysisSummary,
        config: AnalysisSummaryConfig,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the analysis summary viewer.

        Args:
            headers (List[str]): Table headers.
            data (AnalysisSummary): Analysis summary data.
            config (AnalysisSummaryConfig): Configuration for labels.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent=parent)

        self.headers = headers
        self.data = data
        self.data_config = config

        # Widgets
        self.list = TwoColumnList(headers=headers, parent=self)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list)

        # Style
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setLayout(layout)

        # Setup
        self._setup_table(self.data, self.data_config)

    def _setup_table(self, data: AnalysisSummary, config: AnalysisSummaryConfig):
        """Populate the table with analysis summary data.

        Args:
            data (AnalysisSummary): Analysis summary data.
            config (AnalysisSummaryConfig): Configuration for labels.
        """
        self.list.clear()
        self.list.add_item(
            config.avg_centroid_distance_txt, str(data.avg_centroid_distance)
        )
        self.list.add_item(
            config.total_territorial_volume_txt, str(data.total_territorial_volume)
        )
        self.list.add_item(
            config.total_unoccupied_volume_txt, str(data.total_unoccupied_volume)
        )
        self.list.add_item(
            config.percent_occupied_volume_txt, str(data.percent_occupied_volume)
        )

    def update_data(
        self, data: AnalysisSummary, config: Optional[AnalysisSummaryConfig] = None
    ):
        """Update the analysis summary data displayed.

        If a new config is provided, the labels will be updated accordingly.

        Args:
            data (AnalysisSummary): New analysis summary data to display.
            config (Optional[AnalysisSummaryConfig]): Optional new configuration for labels.
        """
        if config:
            self.data_config = config

        self.data = data
        self._setup_table(self.data, self.data_config)


class CellAnalysisViewer(QtWidgets.QWidget):
    """Widget to display cell analysis data with a selector for multiple cells.

    Attributes:
        headers (List[str]): Table headers.
        cells (List[CellAnalysis]): List of cell analysis data.
        cells_config (CellAnalysisConfig): Configuration for labels.
        selector (QComboBox): Dropdown to select a cell.
        table (TwoColumnList): Widget displaying the cell data.
    """

    def __init__(
        self,
        headers: List[str],
        cells: List[CellAnalysis],
        config: CellAnalysisConfig,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the cell analysis viewer.

        Args:
            headers (List[str]): Table headers.
            cells (List[CellAnalysis]): List of cell analysis data.
            config (CellAnalysisConfig): Configuration for labels.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent=parent)

        self.headers = headers
        self.cells = cells
        self.cells_config = config

        # Widgets
        self.selector = QtWidgets.QComboBox(parent=self)
        for idx, _ in enumerate(cells):
            self.selector.addItem(f"Cell {idx + 1}")

        self.table = TwoColumnList(headers=headers, parent=self)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.selector)
        layout.addWidget(self.table)
        self.setLayout(layout)

        # Style
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Connections
        self.selector.currentIndexChanged.connect(self.update_table)

        # Setup
        self.update_table(0)

    def update_table(self, idx: int):
        """Update the table to show data for the selected cell.

        Args:
            idx (int): Index of the selected cell.
        """
        self.table.clear()

        cell = self.cells[idx]

        self.table.add_item(
            self.cells_config.cell_territory_volume_txt, str(cell.cell_territory_volume)
        )
        self.table.add_item(self.cells_config.cell_volume_txt, str(cell.cell_volume))
        self.table.add_item(
            self.cells_config.ramification_index_txt, str(cell.ramification_index)
        )
        self.table.add_item(
            self.cells_config.number_of_endpoints_txt, str(cell.number_of_endpoints)
        )
        self.table.add_item(
            self.cells_config.number_of_branches_txt, str(cell.number_of_branches)
        )
        self.table.add_item(
            self.cells_config.avg_branch_length_txt, str(cell.avg_branch_length)
        )
        self.table.add_item(
            self.cells_config.max_branch_length_txt, str(cell.max_branch_length)
        )
        self.table.add_item(
            self.cells_config.min_branch_length_txt, str(cell.min_branch_length)
        )

    def update_data(
        self, cells: List[CellAnalysis], config: Optional[CellAnalysisConfig] = None
    ):
        """Update the cell analysis data displayed.

        Args:
            cells (List[CellAnalysis]): New list of cell analysis data to display.
            config (Optional[CellAnalysisConfig]): Optional new configuration for labels.
        """
        if config:
            self.cells_config = config

        self.cells = cells

        # Update selector items without emitting index change signals
        self.selector.blockSignals(True)
        self.selector.clear()
        for idx, _ in enumerate(cells):
            self.selector.addItem(f"Cell {idx + 1}")

        if not cells:
            self.selector.blockSignals(False)
            self.table.model.clear()
            self.table.model.setHorizontalHeaderLabels(self.table.headers)
            return

        # Select first cell and refresh table
        self.selector.setCurrentIndex(0)
        self.selector.blockSignals(False)
        self.update_table(0)


class FullAnalysisViewer(QtWidgets.QWidget):
    """Widget to display both analysis summary and cell analysis viewers.

    Attributes:
        analysis (AnalysisSummaryViewer): Widget for analysis summary.
        cells (CellAnalysisViewer): Widget for cell analysis.
    """

    def __init__(
        self,
        summary_headers: List[str],
        cell_headers: List[str],
        analysis_data: AnalysisSummary,
        cells_data: List[CellAnalysis],
        analysis_config: AnalysisSummaryConfig,
        cells_config: CellAnalysisConfig,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the full analysis viewer.

        Args:
            summary_headers (List[str]): Headers for summary table.
            cell_headers (List[str]): Headers for cell table.
            analysis_data (AnalysisSummary): Analysis summary data.
            cells_data (List[CellAnalysis]): List of cell analysis data.
            analysis_config (AnalysisSummaryConfig): Config for summary labels.
            cells_config (CellAnalysisConfig): Config for cell labels.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent=parent)

        # Widgets
        self.analysis = AnalysisSummaryViewer(
            summary_headers, analysis_data, analysis_config, parent=self
        )
        self.cells = CellAnalysisViewer(
            cell_headers, cells_data, cells_config, parent=self
        )

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.analysis)
        layout.addWidget(self.cells)
        self.setLayout(layout)

    def update_data(
        self,
        summary: AnalysisSummary,
        cells: List[CellAnalysis],
        summary_config: Optional[AnalysisSummaryConfig] = None,
        cells_config: Optional[CellAnalysisConfig] = None,
    ):
        """Update both the summary and per-cell viewers with new data.

        Args:
            summary (AnalysisSummary): Updated analysis summary data.
            cells (List[CellAnalysis]): Updated list of per-cell analysis data.
            summary_config (Optional[AnalysisSummaryConfig]): Optional new summary display config.
            cells_config (Optional[CellAnalysisConfig]): Optional new cell display config.
        """
        self.analysis.update_data(summary, summary_config)
        self.cells.update_data(cells, cells_config)
