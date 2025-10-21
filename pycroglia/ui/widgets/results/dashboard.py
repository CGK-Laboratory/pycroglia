from __future__ import annotations
from typing import List, Optional, Type
from types import SimpleNamespace
import sys
from PyQt6 import QtWidgets, QtCore

from pycroglia.core.io.output import (
    AnalysisSummary,
    CellAnalysis,
    AnalysisSummaryConfig,
    CellAnalysisConfig,
    OutputWriter,
)
from pycroglia.ui.widgets.results.graphs import GraphSelectionWidget
from pycroglia.ui.widgets.results.output import OutputConfigurator
from pycroglia.ui.widgets.results.viewers import FullAnalysisViewer


class ResultsDashboard(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        # Widgets
        self.table: Optional[FullAnalysisViewer] = None
        self.graphs: Optional[GraphSelectionWidget] = None
        self.configurator: Optional[OutputConfigurator] = None

    def add_results_table(
        self,
        summary_headers: List[str],
        cell_headers: List[str],
        analysis_data: AnalysisSummary,
        cells_data: List[CellAnalysis],
        analysis_config: AnalysisSummaryConfig,
        cells_config: CellAnalysisConfig,
    ) -> ResultsDashboard:
        self.table = FullAnalysisViewer(
            summary_headers=summary_headers,
            cell_headers=cell_headers,
            analysis_data=analysis_data,
            cells_data=cells_data,
            analysis_config=analysis_config,
            cells_config=cells_config,
            parent=self,
        )
        return self

    def add_graphs_list(
        self,
        graphs_list: List[str],
        label_text: Optional[str] = None,
        button_txt: Optional[str] = None,
    ) -> ResultsDashboard:
        self.graphs = GraphSelectionWidget(
            graphs_list=graphs_list,
            label_txt=label_text,
            button_txt=button_txt,
            parent=self,
        )
        return self

    def add_build_configurator(
        self,
        title: str,
        selection_label: str,
        button_txt: str,
        display_txt: str,
        dialog_title: str,
        dialog_path: str = QtCore.QDir.homePath(),
        writers: Optional[Type[OutputWriter]] = None,
    ) -> ResultsDashboard:
        self.configurator = OutputConfigurator(
            writer_widget_title=title,
            folder_selection_label=selection_label,
            folder_button_txt=button_txt,
            folder_path_display_text=display_txt,
            folder_dialog_title=dialog_title,
            folder_dialog_path=dialog_path,
            writers=writers,
            parent=self,
        )
        return self

    def _build_layout(self):
        layout = QtWidgets.QHBoxLayout()

        lvertical_layout = QtWidgets.QVBoxLayout()
        lvertical_layout.addWidget(self.graphs)
        lvertical_layout.addWidget(self.configurator)

        layout.addWidget(self.table)
        layout.addLayout(lvertical_layout)
        self.setLayout(layout)

    def _validate_components(self) -> None:
        missing = [
            name
            for name, widget in (
                ("table", self.table),
                ("graphs", self.graphs),
                ("configurator", self.configurator),
            )
            if widget is None
        ]
        if missing:
            raise RuntimeError(
                f"Cannot build ResultsDashboard, missing widgets: {', '.join(missing)}"
            )

    def build(self) -> ResultsDashboard:
        self._validate_components()
        self._build_layout()
        return self


# Minimal runnable demo for quick manual testing
def _make_dummy_objects():
    analysis_data = SimpleNamespace(
        avg_centroid_distance=0,
        total_territorial_volume=0,
        total_unoccupied_volume=0,
        percent_occupied_volume=0,
    )
    analysis_config = SimpleNamespace(
        avg_centroid_distance_txt="Avg centroid distance",
        total_territorial_volume_txt="Total territorial volume",
        total_unoccupied_volume_txt="Total unoccupied volume",
        percent_occupied_volume_txt="Percent occupied volume",
    )

    cell = SimpleNamespace(
        cell_territory_volume=0,
        cell_volume=0,
        ramification_index=0,
        number_of_endpoints=0,
        number_of_branches=0,
        avg_branch_length=0,
        max_branch_length=0,
        min_branch_length=0,
    )
    cells_config = SimpleNamespace(
        cell_territory_volume_txt="Cell territory volume",
        cell_volume_txt="Cell volume",
        ramification_index_txt="Ramification index",
        number_of_endpoints_txt="Number of endpoints",
        number_of_branches_txt="Number of branches",
        avg_branch_length_txt="Avg branch length",
        max_branch_length_txt="Max branch length",
        min_branch_length_txt="Min branch length",
    )

    return (
        ["Metric", "Value"],
        ["Property", "Value"],
        analysis_data,
        [cell],
        analysis_config,
        cells_config,
    )


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    (
        summary_headers,
        cell_headers,
        analysis_data,
        cells_data,
        analysis_config,
        cells_config,
    ) = _make_dummy_objects()
    graphs_list = ["Overview", "Volume distribution", "Branch lengths", "Ramification"]

    dashboard = ResultsDashboard()
    dashboard.add_results_table(
        summary_headers=summary_headers,
        cell_headers=cell_headers,
        analysis_data=analysis_data,
        cells_data=cells_data,
        analysis_config=analysis_config,
        cells_config=cells_config,
    ).add_graphs_list(
        graphs_list=graphs_list,
        label_text="Select graphs:",
        button_txt="Show",
    ).add_build_configurator(
        title="Output Writer",
        selection_label="Destination folder:",
        button_txt="Browse...",
        display_txt="No folder selected",
        dialog_title="Select output folder",
    ).build()

    dashboard.show()
    raise SystemExit(app.exec())
