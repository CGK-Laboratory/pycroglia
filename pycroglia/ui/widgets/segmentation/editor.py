from typing import Optional

from PyQt6 import QtWidgets
from numpy.typing import NDArray

from pycroglia.core.labeled_cells import LabelingStrategy, LabeledCells
from pycroglia.ui.controllers.segmentation_state import SegmentationEditorState
from pycroglia.ui.widgets.cells.cells_panel import CellsPanel


class SegmentationControlPanel(QtWidgets.QWidget):
    """Control panel widget for segmentation operations.

    Provides buttons for cell segmentation and rollback operations.

    Attributes:
        DEFAULT_ROLLBACK_BUTTON_TEXT (str): Default text for the rollback button.
        DEFAULT_SEGMENTATION_BUTTON_TEXT (str): Default text for the segmentation button.
    """

    DEFAULT_ROLLBACK_BUTTON_TEXT = "Roll back segmentation"
    DEFAULT_SEGMENTATION_BUTTON_TEXT = "Segment Cell"

    def __init__(
        self,
        rollback_button_text: Optional[str] = None,
        segmentation_button_text: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initializes the SegmentationControlPanel widget.

        Args:
            rollback_button_text (Optional[str], optional): Custom text for rollback button.
            segmentation_button_text (Optional[str], optional): Custom text for segmentation button.
            parent (Optional[QtWidgets.QWidget], optional): Parent widget.
        """
        super().__init__(parent=parent)

        # Store text parameters
        self.rollback_button_text = (
            rollback_button_text or self.DEFAULT_ROLLBACK_BUTTON_TEXT
        )
        self.segmentation_button_text = (
            segmentation_button_text or self.DEFAULT_SEGMENTATION_BUTTON_TEXT
        )

        # Widgets
        self.segment_button = QtWidgets.QPushButton(self.segmentation_button_text)
        self.segment_button.setEnabled(False)

        self.rollback_button = QtWidgets.QPushButton(self.rollback_button_text)
        self.rollback_button.setEnabled(False)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.segment_button)
        layout.addWidget(self.rollback_button)
        self.setLayout(layout)


class SegmentationEditor(QtWidgets.QWidget):
    """Widget for interactive cell segmentation editing.

    Provides a comprehensive UI for visualizing, segmenting, and rolling back cell
    segmentations. Displays a list of cells, a multi-cell viewer, and a single cell viewer
    with integrated segmentation controls.

    Attributes:
        DEFAULT_HEADERS_TEXT (list[str]): Default column headers for the cell list.
        DEFAULT_ROLLBACK_BUTTON_TEXT (str): Default text for the rollback button.
        DEFAULT_SEGMENTATION_BUTTON_TEXT (str): Default text for the segmentation button.
        DEFAULT_PROGRESS_TITLE (str): Default title for the progress dialog.
        DEFAULT_PROGRESS_CANCEL_TEXT (str): Default cancel button text for progress dialog.
        DEFAULT_PROGRESS_MAX (int): Maximum value for the progress bar.
        DEFAULT_PROGRESS_MIN (int): Minimum value for the progress bar.
        LIST_STRETCH_FACTOR (int): Stretch factor for list widget in layout.
        VIEWER_STRETCH_FACTOR (int): Stretch factor for viewer widgets in layout.
    """

    # UI Text Constants
    DEFAULT_HEADERS_TEXT = ["Cell number", "Cell size"]
    DEFAULT_ROLLBACK_BUTTON_TEXT = "Roll back segmentation"
    DEFAULT_SEGMENTATION_BUTTON_TEXT = "Segment Cell"
    DEFAULT_PROGRESS_TITLE = "Segmenting cell..."
    DEFAULT_PROGRESS_CANCEL_TEXT = "Cancel"

    # Progress Dialog Constants
    DEFAULT_PROGRESS_MAX = 100
    DEFAULT_PROGRESS_MIN = 0

    # Layout Constants
    LIST_STRETCH_FACTOR = 1
    VIEWER_STRETCH_FACTOR = 2

    def __init__(
        self,
        img: NDArray,
        labeling_strategy: LabelingStrategy,
        min_size: int,
        with_progress_bar: bool = False,
        headers: Optional[list[str]] = None,
        rollback_button_text: Optional[str] = None,
        segmentation_button_text: Optional[str] = None,
        progress_title: Optional[str] = None,
        progress_cancel_text: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initializes the SegmentationEditor widget.

        Args:
            img (NDArray): 3D binary image to segment.
            labeling_strategy (LabelingStrategy): Strategy for labeling connected components.
            min_size (int): Minimum size for objects to keep after noise removal.
            with_progress_bar (bool, optional): Whether to show a progress bar during segmentation.
            headers (Optional[list[str]], optional): Column headers for the cell list.
            rollback_button_text (Optional[str], optional): Text for the rollback button.
            segmentation_button_text (Optional[str], optional): Text for the segmentation button.
            progress_title (Optional[str], optional): Title for the progress dialog.
            progress_cancel_text (Optional[str], optional): Cancel button text for progress dialog.
            parent (Optional[QtWidgets.QWidget], optional): Parent widget.
        """
        super().__init__(parent=parent)

        # Configurable text properties
        self.headers_text = headers or self.DEFAULT_HEADERS_TEXT
        self.rollback_button_text = (
            rollback_button_text or self.DEFAULT_ROLLBACK_BUTTON_TEXT
        )
        self.segmentation_button_text = (
            segmentation_button_text or self.DEFAULT_SEGMENTATION_BUTTON_TEXT
        )
        self.progress_title = progress_title or self.DEFAULT_PROGRESS_TITLE
        self.progress_cancel_text = (
            progress_cancel_text or self.DEFAULT_PROGRESS_CANCEL_TEXT
        )

        # Properties
        self.state = SegmentationEditorState(img, labeling_strategy, min_size)
        self.with_progress_bar = with_progress_bar

        # Widgets
        self.control_panel = SegmentationControlPanel(
            rollback_button_text=self.rollback_button_text,
            segmentation_button_text=self.segmentation_button_text,
            parent=self,
        )
        self.viewer = CellsPanel(
            self.state.get_state(),
            headers=self.headers_text,
            control_panel=self.control_panel,
            parent=self,
        )

        # Connections
        self.viewer.cell_list.selectionChanged.connect(self._on_cell_selection_changed)
        self.control_panel.segment_button.clicked.connect(self._on_cell_segmentation)
        self.control_panel.rollback_button.clicked.connect(self._on_rollback_request)
        self.state.stateChanged.connect(self._load_data)

        # Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.viewer)
        self.setLayout(layout)

        # Load data
        self._load_data()

    def _load_data(self):
        """Loads and displays the current segmentation state in the UI.

        Updates the viewer with the current state and enables/disables the rollback
        button based on state history availability.
        """
        actual_state = self.state.get_state()

        self.viewer.load_data(actual_state)
        self.control_panel.rollback_button.setEnabled(self.state.has_prev_state())

    def _on_cell_selection_changed(self):
        """Handles cell selection changes from the cell list.

        Enables or disables the segment button based on whether a cell is selected.
        """
        self.control_panel.segment_button.setEnabled(
            self.viewer.cell_list.get_selected_cell_id() is not None
        )

    def _on_cell_segmentation(self):
        """Handles the segmentation request for the selected cell.

        Shows a progress bar if enabled and performs segmentation of the currently
        selected cell. Updates the UI after completion.
        """
        selected_cell_info = self.viewer.cell_list.get_selected_cell_info()
        if selected_cell_info is None:
            return

        progress_bar = None
        if self.with_progress_bar:
            progress_bar = QtWidgets.QProgressDialog(
                self.progress_title,
                self.progress_cancel_text,
                self.DEFAULT_PROGRESS_MIN,
                self.DEFAULT_PROGRESS_MAX,
            )
            progress_bar.setModal(True)
            progress_bar.show()

        try:
            self.state.segment_cell(
                selected_cell_info[0], selected_cell_info[1], progress_bar
            )
        finally:
            if progress_bar:
                progress_bar.close()
            self.viewer.load_data(self.state.get_state())

    def _on_rollback_request(self):
        """Handles the rollback request to restore the previous segmentation state.

        Reverts to the previous segmentation state and updates the UI display.
        """
        self.state.rollback()
        self.viewer.load_data(self.state.get_state())

    def get_results(self) -> LabeledCells:
        """Get the current segmentation results.

        Returns the current state of the labeled cells after all segmentation
        operations have been performed.

        Returns:
            LabeledCells: The current labeled cells object containing all
                segmentation results.
        """
        return self.state.get_state()
