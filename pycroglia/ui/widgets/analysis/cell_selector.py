from typing import Optional, Set

from PyQt6 import QtWidgets, QtGui

from pycroglia.core.labeled_cells import LabeledCells
from pycroglia.ui.widgets.common.img_viewer import CustomImageViewer
from pycroglia.ui.widgets.cells.cell_list import CellList
from pycroglia.ui.widgets.cells.multi_cell_img_viewer import  MultiCellImageViewer

class CellSelector(QtWidgets.QWidget):
    # UI Text Constants
    DEFAULT_HEADERS_TEXT = ["Cell number", "Cell size"]
    DEFAULT_UNSELECT_CELL_BUTTON_TEXT = "Remove Cell"

    # Layout Constants
    LIST_STRETCH_FACTOR = 1
    VIEWER_STRETCH_FACTOR = 2

    def __init__(
        self,
        img: LabeledCells,
        headers: Optional[list[str]] = None,
        unselect_button_text: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(parent=parent)

        # State
        self.img = img
        self.unselected_cells = set()

        self.headers_text = headers or self.DEFAULT_HEADERS_TEXT
        self.unselect_button_text = unselect_button_text or self.DEFAULT_UNSELECT_CELL_BUTTON_TEXT

        # Widgets
        self.list = CellList(headers=self.headers_text, parent=self)

        self.unselect_button = QtWidgets.QPushButton(parent=self)
        self.unselect_button.setText(self.unselect_button_text)
        self.unselect_button.setEnabled(False)

        self.multi_cell_viewer = MultiCellImageViewer(parent=self)
        self.cell_viewer = CustomImageViewer(parent=self)

        # Connections
        self.list.selectionChanged.connect(self._on_cell_selection_changed)
        self.unselect_button.clicked.connect(self._on_unselect_button_clicked)

        # Layout
        list_layout = QtWidgets.QVBoxLayout()
        list_layout.addWidget(self.list)
        list_layout.addWidget(self.unselect_button)
        list_layout_widget = QtWidgets.QWidget()
        list_layout_widget.setLayout(list_layout)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(list_layout_widget, stretch=self.LIST_STRETCH_FACTOR)
        layout.addWidget(self.multi_cell_viewer, stretch=self.VIEWER_STRETCH_FACTOR)
        layout.addWidget(self.cell_viewer, stretch=self.VIEWER_STRETCH_FACTOR)
        self.setLayout(layout)

        # Load data
        self._load_data(self.img)

    def _load_data(self, img: LabeledCells):
        self.list.clear_cells()
        self.list.add_cells(img)
        self.multi_cell_viewer.set_cells_img(img)

    def _on_cell_selection_changed(self):
        selected_cell = self.list.get_selected_cell_id()
        self.unselect_button.setEnabled(selected_cell is not None)

        if selected_cell is None:
            return

        cell_2d = self.img.cell_to_2d(selected_cell)
        self.cell_viewer.set_image(cell_2d)

    def _on_unselect_button_clicked(self):
        selected_cell = self.list.get_selected_cell_id()
        if selected_cell is None:
            return

        if selected_cell in self.unselected_cells:
            self.unselected_cells.remove(selected_cell)
        else:
            self.unselected_cells.add(selected_cell)

        self._update_row_color(selected_cell)

    def _update_row_color(self, cell_id: int):
        model = self.list.list.model

        for row in range(model.rowCount()):
            if int(model.item(row, 0).text()) == cell_id:
                color = QtGui.QColor(255, 200, 200) if cell_id in self.unselected_cells else None

                for col in range(model.columnCount()):
                    item = model.item(row, col)
                    if color:
                        item.setBackground(color)
                    else:
                        item.setBackground(QtGui.QBrush())
                break

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