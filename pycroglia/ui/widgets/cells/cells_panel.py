from typing import Optional

from PyQt6 import QtWidgets

from pycroglia.core.labeled_cells import LabeledCells
from pycroglia.ui.widgets.cells.cell_list import CellList
from pycroglia.ui.widgets.cells.multi_cell_img_viewer import MultiCellImageViewer
from pycroglia.ui.widgets.common.img_viewer import CustomImageViewer


class CellsPanel(QtWidgets.QWidget):
    """Widget panel for displaying and interacting with labeled cells.

    Provides a comprehensive view of labeled cells including a list view,
    multi-cell viewer, and individual cell viewer. Includes optional control panel.

    Attributes:
        LIST_STRETCH_FACTOR (int): Stretch factor for the list widget in the layout.
        VIEWER_STRETCH_FACTOR (int): Stretch factor for viewer widgets in the layout.
        DEFAULT_HEADERS_TEXT (list[str]): Default column headers for the cell list.
    """

    LIST_STRETCH_FACTOR = 1
    VIEWER_STRETCH_FACTOR = 2
    DEFAULT_HEADERS_TEXT = ["Cell number", "Cell Size"]

    def __init__(
        self,
        img: LabeledCells,
        headers: Optional[list[str]] = None,
        control_panel: Optional[QtWidgets.QWidget] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initializes the CellsPanel widget.

        Args:
            img (LabeledCells): Labeled cells object containing the cell data.
            headers (Optional[list[str]], optional): Custom column headers for cell list.
            control_panel (Optional[QtWidgets.QWidget], optional): Additional control widget to display.
            parent (Optional[QtWidgets.QWidget], optional): Parent widget.
        """
        super().__init__(parent=parent)

        # Store text parameters
        self.headers = headers or self.DEFAULT_HEADERS_TEXT

        # State properties
        self.img = img

        # Widgets
        self.cell_list = CellList(headers=self.headers, parent=self)
        self.multi_viewer = MultiCellImageViewer(parent=self)
        self.cell_viewer = CustomImageViewer(parent=self)
        self.control_panel = control_panel

        # Connections
        self.cell_list.selectionChanged.connect(self._handle_cell_selection)

        # Layout
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.cell_list)
        if self.control_panel:
            left_layout.addWidget(self.control_panel)

        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_layout)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(left_widget, stretch=self.LIST_STRETCH_FACTOR)
        layout.addWidget(self.multi_viewer, stretch=self.VIEWER_STRETCH_FACTOR)
        layout.addWidget(self.cell_viewer, stretch=self.VIEWER_STRETCH_FACTOR)
        self.setLayout(layout)

        # Load data
        self.cell_list.clear_cells()
        self.cell_list.add_cells(self.img)
        self.multi_viewer.set_cells_img(self.img)

    def load_data(self, img: LabeledCells):
        """Loads new labeled cell data into the panel.

        Updates all child widgets with the new cell data and refreshes displays.

        Args:
            img (LabeledCells): New labeled cells object to display.
        """
        self.img = img

        self.cell_list.clear_cells()
        self.cell_list.add_cells(img)
        self.multi_viewer.set_cells_img(img)

    def _handle_cell_selection(self):
        """Handles cell selection changes from the cell list.

        Updates the individual cell viewer to display the 2D projection
        of the currently selected cell.
        """
        selected_cell = self.cell_list.get_selected_cell_id()
        if selected_cell:
            cell_2d = self.img.cell_to_2d(selected_cell)
            self.cell_viewer.set_image(cell_2d)
