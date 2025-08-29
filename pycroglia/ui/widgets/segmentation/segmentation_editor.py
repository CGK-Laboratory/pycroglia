from typing import Optional

from PyQt6 import QtWidgets
from numpy.typing import NDArray

from pycroglia.core.labeled_cells import LabelingStrategy
from pycroglia.ui.widgets.common.img_viewer import CustomImageViewer
from pycroglia.ui.widgets.segmentation.cell_list import CellList
from pycroglia.ui.widgets.segmentation.multi_cell_img_viewer import MultiCellImageViewer
from pycroglia.ui.controllers.segmentation_state import SegmentationEditorState


class SegmentationEditor(QtWidgets.QWidget):
    """Widget for interactive cell segmentation editing.

    Provides a UI for visualizing, segmenting, and rolling back cell segmentations.
    Displays a list of cells, a multi-cell viewer, and a single cell viewer.
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

        self.state = SegmentationEditorState(img, labeling_strategy, min_size)
        self.with_progress_bar = with_progress_bar

        # Widgets
        self.list = CellList(headers=self.headers_text, parent=self)

        self.segment_button = QtWidgets.QPushButton(self.segmentation_button_text)
        self.segment_button.setEnabled(False)

        self.rollback_button = QtWidgets.QPushButton(self.rollback_button_text)
        self.rollback_button.setEnabled(False)

        self.multi_cell_viewer = MultiCellImageViewer(parent=self)
        self.cell_viewer = CustomImageViewer(parent=self)

        # Connections
        self.list.selectionChanged.connect(self._on_cell_selection_changed)
        self.segment_button.clicked.connect(self._on_cell_segmentation_request)
        self.rollback_button.clicked.connect(self._on_rollback_request)
        self.state.stateChanged.connect(self._load_data)

        # Layout
        list_layout = QtWidgets.QVBoxLayout()
        list_layout.addWidget(self.list, stretch=self.LIST_STRETCH_FACTOR)
        list_layout.addWidget(self.segment_button)
        list_layout.addWidget(self.rollback_button)
        list_container = QtWidgets.QWidget()
        list_container.setLayout(list_layout)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(list_container, stretch=self.LIST_STRETCH_FACTOR)
        layout.addWidget(self.multi_cell_viewer, stretch=self.VIEWER_STRETCH_FACTOR)

        # For consistency
        # TODO - Improve style
        cell_viewer_layout = QtWidgets.QVBoxLayout()
        cell_viewer_layout.addWidget(self.cell_viewer)
        cell_viewer_container = QtWidgets.QWidget()
        cell_viewer_container.setLayout(cell_viewer_layout)
        layout.addWidget(cell_viewer_container, stretch=self.VIEWER_STRETCH_FACTOR)

        self.setLayout(layout)

        # Loads data
        self._load_data()

    def _load_data(self):
        """Loads and displays the current segmentation state in the UI."""
        actual_state = self.state.get_state()

        self.list.clear_cells()
        self.list.add_cells(actual_state)
        self.multi_cell_viewer.set_cells_img(actual_state)

        self.rollback_button.setEnabled(self.state.has_prev_state())

    def _on_cell_selection_changed(self):
        """Handles cell selection changes in the list and updates the cell viewer."""
        selected_cell = self.list.get_selected_cell_id()
        self.segment_button.setEnabled(selected_cell is not None)
        if selected_cell is None:
            return

        cell_2d = self.state.get_state().cell_to_2d(selected_cell)
        self.cell_viewer.set_image(cell_2d)

    def _on_cell_segmentation_request(self):
        """Handles the segmentation request for the selected cell, showing a progress bar if enabled."""
        selected_cell_info = self.list.get_selected_cell_info()
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

    def _on_rollback_request(self):
        """Handles the rollback request to restore the previous segmentation state."""
        self.state.rollback()
