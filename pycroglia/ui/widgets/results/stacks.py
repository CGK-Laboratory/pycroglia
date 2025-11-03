from pathlib import Path
from typing import Optional, List, Type

from PyQt6 import QtWidgets

from pycroglia.core.io.output import OutputWriter
from pycroglia.ui.widgets.common.results import ImgWithPathResults
from pycroglia.ui.widgets.results.dashboard import ResultsDashboard


class ResultsDashboardStack(QtWidgets.QWidget):
    """Widget that manages a tabbed collection of ResultsDashboard instances.

    Each tab contains a ResultsDashboard configured for a single image (provided
    as ImgWithPathResults). The stack exposes defaults for header texts, graph
    labels and output configurator texts which are applied when creating each
    dashboard.

    Attributes:
        summary_headers (List[str]): Column headers for summary table.
        cell_headers (List[str]): Column headers for per-cell table.
        graph_text (str): Label text for the graphs selector.
        graph_button_txt (str): Button text for the graphs preview button.
        output_title_txt (str): Title for the output configurator widget.
        output_select_txt (str): Label for folder selection.
        output_button_txt (str): Text for the browse button.
        output_display_txt (str): Initial display text for the folder path.
        output_dialog_title_txt (str): Title for the folder selection dialog.
        writers (Optional[Type[OutputWriter]]): Available output writers.
    """

    # UI Text Constants
    DEFAULT_RESULTS_HEADERS = ["Metric", "Value"]
    DEFAULT_GRAPHS_TEXT = "Select graphs:"
    DEFAULT_GRAPHS_BUTTON_TXT = "Preview"
    DEFAULT_OUTPUT_TITLE_TEXT = "Output formats"
    DEFAULT_OUTPUT_SELECT_TEXT = "Destination folder"
    DEFAULT_OUTPUT_BUTTON_TEXT = "Browse"
    DEFAULT_OUTPUT_DISPLAY_TXT = "No folder selected"
    DEFAULT_OUTPUT_DIALOG_TITLE_TXT = "Select a folder"
    DEFAULT_WRITERS = OutputWriter.get_writers()

    def __init__(
        self,
        summary_headers: Optional[List[str]] = None,
        cell_headers: Optional[List[str]] = None,
        graphs_text: Optional[str] = None,
        graph_button_txt: Optional[str] = None,
        output_title_txt: Optional[str] = None,
        output_select_txt: Optional[str] = None,
        output_button_txt: Optional[str] = None,
        output_display_txt: Optional[str] = None,
        output_dialog_title_txt: Optional[str] = None,
        writers: Optional[Type[OutputWriter]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the ResultsDashboardStack.

        Args:
            summary_headers (Optional[List[str]]): Custom summary headers, falls back to defaults if None.
            cell_headers (Optional[List[str]]): Custom cell headers, falls back to defaults if None.
            graphs_text (Optional[str]): Label text for the graphs selector.
            graph_button_txt (Optional[str]): Button text for the graphs preview button.
            output_title_txt (Optional[str]): Title for the output configurator widget.
            output_select_txt (Optional[str]): Label for folder selection.
            output_button_txt (Optional[str]): Text for the browse button.
            output_display_txt (Optional[str]): Initial display text for the folder path.
            output_dialog_title_txt (Optional[str]): Title for the folder selection dialog.
            writers (Optional[Type[OutputWriter]]): Optional writer registry or list of writers.
            parent (Optional[QtWidgets.QWidget]): Optional parent widget.
        """
        super().__init__(parent=parent)

        # Text properties
        self.summary_headers = summary_headers or self.DEFAULT_RESULTS_HEADERS
        self.cell_headers = cell_headers or self.DEFAULT_RESULTS_HEADERS
        self.graph_text = graphs_text or self.DEFAULT_GRAPHS_TEXT
        self.graph_button_txt = graph_button_txt or self.DEFAULT_GRAPHS_BUTTON_TXT
        self.output_title_txt = output_title_txt or self.DEFAULT_OUTPUT_TITLE_TEXT
        self.output_select_txt = output_select_txt or self.DEFAULT_OUTPUT_SELECT_TEXT
        self.output_button_txt = output_button_txt or self.DEFAULT_OUTPUT_BUTTON_TEXT
        self.output_display_txt = output_display_txt or self.DEFAULT_OUTPUT_DISPLAY_TXT
        self.output_dialog_title_txt = (
            output_dialog_title_txt or self.DEFAULT_OUTPUT_DIALOG_TITLE_TXT
        )
        self.writers = writers or self.DEFAULT_WRITERS

        # Widgets
        self.tabs = QtWidgets.QTabWidget(self)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def add_tabs(self, results: List[ImgWithPathResults]):
        """Clear existing tabs and add a dashboard tab for each provided image.

        For each ImgWithPathResults element creates a ResultsDashboard, configures
        it with the stack's header/button texts and writers, builds it and adds
        it to the tab widget.

        Args:
            results (List[ImgWithPathResults]): List of images with paths to create tabs for.
        """
        self.tabs.clear()

        # TODO - Add scale text
        for elem in results:
            dashboard = ResultsDashboard(
                file=elem.file_path, img=elem.img, cells_masks=elem.cells_masks
            )
            dashboard.add_results_table(
                summary_headers=self.summary_headers, cell_headers=self.cell_headers
            ).add_graphs_list(
                label_text=self.graph_text, button_txt=self.graph_button_txt
            ).add_build_configurator(
                title=self.output_title_txt,
                selection_label=self.output_select_txt,
                button_txt=self.output_button_txt,
                display_txt=self.output_display_txt,
                dialog_title=self.output_dialog_title_txt,
                writers=self.writers,
            ).add_scale_config().build()

            self.tabs.addTab(dashboard, f"{Path(elem.file_path).name}")
