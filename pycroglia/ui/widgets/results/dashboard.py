from typing import List, Optional, Type
from PyQt6 import QtWidgets, QtCore

from pycroglia.core.io.output import AnalysisSummary, CellAnalysis, AnalysisSummaryConfig, CellAnalysisConfig, \
    OutputWriter
from pycroglia.ui.widgets.results.graphs import GraphSelectionWidget
from pycroglia.ui.widgets.results.output import OutputConfigurator
from pycroglia.ui.widgets.results.viewers import FullAnalysisViewer


class ResultsDashboard(QtWidgets.QWidget):

    def __init__(self,
                 table_summary_headers: List[str],
                 table_cell_headers: List[str],
                 table_analysis_data: AnalysisSummary,
                 table_cells_data: List[CellAnalysis],
                 table_analysis_config: AnalysisSummaryConfig,
                 table_cells_config: CellAnalysisConfig,
                 graphs_list: List[str],
                 graphs_label_txt: str,
                 graphs_button_txt: str,
                 configurator_title: str,
                 configurator_selection_label: str,
                 configurator_button_txt: str,
                 configurator_folder_path_text: str,
                 configurator_dialog_title: str,
                 folder_dialog_path: str = QtCore.QDir.homePath(),
                 writers: Optional[Type[OutputWriter]] = None,
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        # Widgets
        self.table =  FullAnalysisViewer(
            summary_headers=table_summary_headers,
            cell_headers=table_cell_headers,
            analysis_data=table_analysis_data,
            cells_data=table_cells_data,
            analysis_config=table_analysis_config,
            cells_config=table_cells_config,
            parent=self
        )
        self.graphs = GraphSelectionWidget(
            graphs_list=graphs_list,
            label_txt=graphs_label_txt,
            button_txt=graphs_button_txt,
            parent=self
        )
        self.configurator = OutputConfigurator(
            writer_widget_title=configurator_title,
            folder_selection_label=configurator_selection_label,
            folder_button_txt=configurator_button_txt,
            folder_path_display_text=configurator_folder_path_text,
            folder_dialog_title=configurator_dialog_title,
            folder_dialog_path=folder_dialog_path,
            writers=writers
        )

        # Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.graphs)
        layout.addWidget(self.configurator)
        self.setLayout(layout)