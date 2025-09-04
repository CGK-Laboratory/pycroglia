from typing import Optional, Set, Dict

from PyQt6 import QtWidgets, QtGui

from pycroglia.core.labeled_cells import LabeledCells
from pycroglia.ui.widgets.common.img_viewer import CustomImageViewer
from pycroglia.ui.widgets.cells.cell_list import CellList
from pycroglia.ui.widgets.cells.multi_cell_img_viewer import MultiCellImageViewer
from pycroglia.ui.widgets.common.labeled_widgets import LabeledSpinBox

class CellSelector(QtWidgets.QWidget):
    # UI Text Constants
    DEFAULT_HEADERS_TEXT = ["Cell number", "Cell size"]
    DEFAULT_REMOVE_BUTTON_TEXT = "Remove Cell"
    DEFAULT_SIZE_LABEL_TEXT = "Cell Size"
    DEFAULT_SIZE_BUTTON_TEXT = "Remove smaller than"

    # Layout Constants
    LIST_STRETCH_FACTOR = 1
    VIEWER_STRETCH_FACTOR = 2

    # Color Constants
    UNSELECTED_COLOR = QtGui.QColor(255, 200, 200)
    SELECTED_COLOR = QtGui.QBrush()

    def __init__(
        self,
        img: LabeledCells,
        headers: Optional[list[str]] = None,
        remove_button_text: Optional[str] = None,
        size_label_text: Optional[str] = None,
        size_button_text: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(parent=parent)

        # State
        self.img = img
        self.unselected_cells = set()
        self._cell_to_row_cache: Dict[int, int] = {}

        self.headers_text = headers or self.DEFAULT_HEADERS_TEXT
        self.remove_button_text = remove_button_text or self.DEFAULT_REMOVE_BUTTON_TEXT
        self.size_label_text = size_label_text or self.DEFAULT_SIZE_LABEL_TEXT
        self.size_button_text = size_button_text or self.DEFAULT_SIZE_BUTTON_TEXT

        # Widgets
        self.cell_list = CellList(headers=self.headers_text, parent=self)

        self.remove_btn = QtWidgets.QPushButton(parent=self)
        self.remove_btn.setText(self.remove_button_text)
        self.remove_btn.setEnabled(False)

        max_cell_size = max(img.get_cell_size(i) for i in range(1, img.len() + 1))
        self.size_input = LabeledSpinBox(label_text=self.size_label_text, min_value=0, max_value=max_cell_size, parent=self)
        self.size_btn = QtWidgets.QPushButton(parent=self)
        self.size_btn.setText(self.size_button_text)

        self.multi_viewer = MultiCellImageViewer(parent=self)
        self.cell_viewer = CustomImageViewer(parent=self)

        # Connections
        self.cell_list.selectionChanged.connect(self._on_cell_selection_changed)
        self.remove_btn.clicked.connect(self._on_remove_button_clicked)
        self.size_btn.clicked.connect(self._on_size_button_clicked)

        # Layout
        list_layout = QtWidgets.QVBoxLayout()
        list_layout.addWidget(self.cell_list)
        list_layout.addWidget(self.remove_btn)
        list_layout.addWidget(self.size_input)
        list_layout.addWidget(self.size_btn)
        list_layout_widget = QtWidgets.QWidget()
        list_layout_widget.setLayout(list_layout)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(list_layout_widget, stretch=self.LIST_STRETCH_FACTOR)
        layout.addWidget(self.multi_viewer, stretch=self.VIEWER_STRETCH_FACTOR)
        layout.addWidget(self.cell_viewer, stretch=self.VIEWER_STRETCH_FACTOR)
        self.setLayout(layout)

        # Load data
        self._load_data(self.img)

    def _load_data(self, img: LabeledCells):
        self.cell_list.clear_cells()
        self.cell_list.add_cells(img)
        self.multi_viewer.set_cells_img(img)
        self._build_cell_to_row_cache()

    def _build_cell_to_row_cache(self):
        """Build a cache mapping cell_id to row index for efficient lookups."""
        self._cell_to_row_cache.clear()
        model = self.cell_list.list.model

        for row in range(model.rowCount()):
            cell_id = int(model.item(row, 0).text())
            self._cell_to_row_cache[cell_id] = row

    def _set_row_color(self, cell_id: int, is_unselected: bool):
        """Set the color of a specific row based on selection state."""
        if cell_id not in self._cell_to_row_cache:
            return

        row = self._cell_to_row_cache[cell_id]
        model = self.cell_list.list.model
        color = self.UNSELECTED_COLOR if is_unselected else self.SELECTED_COLOR

        for col in range(model.columnCount()):
            item = model.item(row, col)
            item.setBackground(color)

    def _update_colors_batch(self, cell_ids: Set[int], is_unselected: bool):
        """Update colors for multiple cells efficiently."""
        for cell_id in cell_ids:
            self._set_row_color(cell_id, is_unselected)

    def _on_cell_selection_changed(self):
        selected_cell = self.cell_list.get_selected_cell_id()
        self.remove_btn.setEnabled(selected_cell is not None)

        if selected_cell is None:
            return

        cell_2d = self.img.cell_to_2d(selected_cell)
        self.cell_viewer.set_image(cell_2d)

    def _on_remove_button_clicked(self):
        selected_cell = self.cell_list.get_selected_cell_id()
        if selected_cell is None:
            return

        if selected_cell in self.unselected_cells:
            self.unselected_cells.remove(selected_cell)
            self._set_row_color(selected_cell, False)
        else:
            self.unselected_cells.add(selected_cell)
            self._set_row_color(selected_cell, True)

    def _on_size_button_clicked(self):
        threshold = self.size_input.get_value()

        cells_to_unselect = set()
        for cell_id in range(1, self.img.len() + 1):
            cell_size = self.img.get_cell_size(cell_id)
            if cell_size < threshold:
                cells_to_unselect.add(cell_id)

        self.unselected_cells.update(cells_to_unselect)
        self._update_colors_batch(cells_to_unselect, True)

    def get_selected_cells(self) -> Set[int]:
        all_cells = set(range(1, self.img.len() + 1))
        return all_cells - self.unselected_cells

    def get_unselected_cells(self) -> Set[int]:
        return self.unselected_cells.copy()


# --- Replace with your TIFF file path ---
import sys
from pycroglia.core.labeled_cells import SkimageImgLabeling
from pycroglia.core.enums import SkimageCellConnectivity
from pycroglia.core.files import TiffReader
from pycroglia.core.filters import remove_small_objects, calculate_otsu_threshold
TIFF_PATH = "/Users/framos/Desktop/TrialControlZip.tif"
CHANNELS = 5  # Set according to your file
CHANNEL_OF_INTEREST = 2  # 1-based index


def main():
    # Read TIFF file
    reader = TiffReader(TIFF_PATH)
    img = reader.read(CHANNELS, CHANNEL_OF_INTEREST)

    # Filters
    noise = 100
    img = calculate_otsu_threshold(img, 1.0)
    img = remove_small_objects(img, noise, connectivity=SkimageCellConnectivity.CORNERS)

    # Create a dummy labeling strategy (replace with your actual strategy)
    labeling_strategy = SkimageImgLabeling(SkimageCellConnectivity.CORNERS)
    labeled_cell = LabeledCells(img, labeling_strategy)

    # Create the application and widget
    app = QtWidgets.QApplication(sys.argv)
    editor = CellSelector(labeled_cell)
    editor.show()
    sys.exit(app.exec())


main()