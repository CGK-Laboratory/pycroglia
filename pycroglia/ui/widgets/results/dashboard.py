from __future__ import annotations
from typing import List, Optional, Type
from numpy.typing import NDArray
from PyQt6 import QtWidgets, QtCore

from pycroglia.ui.controllers.dashboard_state import (
    ResultsDashboardState,
    ResultsDashboardTextConfig,
)
from pycroglia.core.io.output import OutputWriter
from pycroglia.ui.controllers.graphs_state import DashboardGraphsGenerator
from pycroglia.ui.widgets.results.graphs import GraphSelectionWidget
from pycroglia.ui.widgets.results.output import OutputConfigurator
from pycroglia.ui.widgets.results.scale import ScaleConfigWidget
from pycroglia.ui.widgets.results.viewers import FullAnalysisViewer
import copy


class ResultsDashboard(QtWidgets.QWidget):
    """Dashboard widget that aggregates results viewers, graph selectors and output configurator.

    The dashboard is constructed in three steps via builder-style methods:
    add_results_table, add_graphs_list and add_build_configurator. After
    configuring the three components call build() to validate and assemble
    the layout.

    Attributes:
        state (ResultsProvider): Source of analysis, cells and graph data.
        table (Optional[FullAnalysisViewer]): Results table widget (set by add_results_table).
        graphs (Optional[GraphSelectionWidget]): Graph selection widget (set by add_graphs_list).
        configurator (Optional[OutputConfigurator]): Output configuration widget (set by add_build_configurator).
    """

    def __init__(
        self,
        file: str,
        img: NDArray,
        cells_masks: List[NDArray],
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the ResultsDashboard.

        The dashboard creates an internal ResultsDashboardState from the provided
        image/array and prepares placeholders for child widgets; actual child
        widgets are constructed via the builder methods.

        Args:
            img (NDArray): Image or labeled array used to construct the internal state.
            parent (Optional[QtWidgets.QWidget]): Optional parent widget.
        """
        super().__init__(parent=parent)

        # Config
        self._text_config = ResultsDashboardTextConfig(file=file, parent=self)

        # State
        self._state = ResultsDashboardState(
            file=file, img=img, cells_masks=cells_masks, parent=self
        )
        self._graphs_generator = DashboardGraphsGenerator(
            img=img, cells=cells_masks, parent=self
        )

        # Widgets
        self.table: Optional[FullAnalysisViewer] = None
        self.graphs: Optional[GraphSelectionWidget] = None
        self.scales: Optional[ScaleConfigWidget] = None
        self.configurator: Optional[OutputConfigurator] = None
        self.cell_masks = copy.copy(cells_masks)
        # Connections
        self._state.resultsChanged.connect(self._update_results_view)

    def add_results_table(
        self,
        summary_headers: List[str],
        cell_headers: List[str],
    ) -> ResultsDashboard:
        """Create and attach a FullAnalysisViewer using state-provided data.

        Args:
            summary_headers (List[str]): Column headers for the summary table.
            cell_headers (List[str]): Column headers for the per-cell table.

        Returns:
            ResultsDashboard: self, to allow chaining.
        """
        self.table = FullAnalysisViewer(
            summary_headers=summary_headers,
            cell_headers=cell_headers,
            analysis_data=self._state.get_summary(),
            cells_data=self._state.get_per_cell(),
            analysis_config=self._text_config.get_summary_text_config(),
            cells_config=self._text_config.get_per_cell_text_config(),
            parent=self,
        )

        return self

    def add_scale_config(
        self,
        scale_txt: Optional[str] = None,
        z_scale_txt: Optional[str] = None,
        button_txt: Optional[str] = None,
    ):
        self.scales = ScaleConfigWidget(
            scale_txt=scale_txt,
            z_scale_txt=z_scale_txt,
            button_txt=button_txt,
        )

        # Connections
        self.scales.clicked.connect(self._compute_on_demand)

        return self

    def add_graphs_list(
        self,
        label_text: Optional[str] = None,
        button_txt: Optional[str] = None,
    ) -> ResultsDashboard:
        """Create and attach a GraphSelectionWidget using state-provided graphs list.

        Args:
            label_text (Optional[str]): Optional label text for the graphs selector.
            button_txt (Optional[str]): Optional button text.

        Returns:
            ResultsDashboard: self, to allow chaining.
        """
        self.graphs = GraphSelectionWidget(
            graphs_list=self._graphs_generator.get_graphs_list(),
            label_txt=label_text,
            button_txt=button_txt,
            parent=self,
        )

        # Connections
        self.graphs.buttonClicked.connect(self._preview_clicked)

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
        """Create and attach an OutputConfigurator.

        Args:
            title (str): Title for the writer widget.
            selection_label (str): Label describing the folder selector.
            button_txt (str): Text for the folder browse button.
            display_txt (str): Initial display text for the folder path.
            dialog_title (str): Title for the folder selection dialog.
            dialog_path (str): Initial path for the dialog.
            writers (Optional[Type[OutputWriter]]): Optional writer class or registry.

        Returns:
            ResultsDashboard: self, to allow chaining.
        """
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
        """Assemble the dashboard layout.

        Places the results table on the left and stacks the graphs selector
        and configurator vertically on the right.
        """
        layout = QtWidgets.QHBoxLayout()

        first_row = QtWidgets.QVBoxLayout()
        first_row.addWidget(self.table)
        first_row.addWidget(self.scales)

        second_row = QtWidgets.QVBoxLayout()
        second_row.addWidget(self.graphs)
        second_row.addWidget(self.configurator)

        layout.addLayout(first_row)
        layout.addLayout(second_row)
        self.setLayout(layout)

    def _validate_components(self) -> None:
        """Validate that required child widgets have been constructed.

        Raises:
            RuntimeError: If one or more required widgets are missing.
        """
        missing = [
            name
            for name, widget in (
                ("table", self.table),
                ("graphs", self.graphs),
                ("configurator", self.configurator),
                ("scales", self.scales),
            )
            if widget is None
        ]
        if missing:
            raise RuntimeError(
                f"Cannot build ResultsDashboard, missing widgets: {', '.join(missing)}"
            )

    def build(self) -> ResultsDashboard:
        """Finalize the dashboard: validate components and build layout.

        Returns:
            ResultsDashboard: self, to allow chaining.

        Raises:
            RuntimeError: If validation fails because some components are missing.
        """
        self._validate_components()
        self._build_layout()

        return self

    def _compute_on_demand(self):
        self._state.calculate_results(
            scale=self.scales.get_scale(),
            z_scale=self.scales.get_z_scale(),
        )

    def _preview_clicked(self, selected_plots: List[str]):
        """Handle preview requests coming from the GraphSelectionWidget.

        Delegates the list of selected graph names to the state's generate_graphs
        implementation so the graphs are generated or displayed.

        Args:
            graphs_list (List[str]): Names of graphs requested for preview.
        """

        cells = self._state.get_per_cell()
        self._graphs_generator.generate_plots(
            selected_plots,
            self.cell_masks,
            cells[0].branch_analysis,
            cells[0].full_cell_analysis,
            self.scales.get_scale(),
            self.scales.get_z_scale(),
        )

    def _update_results_view(self):
        self.table.update_data(
            summary=self._state.get_summary(), cells=self._state.get_per_cell()
        )
