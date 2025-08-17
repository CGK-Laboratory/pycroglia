import sys
import numpy as np

from typing import Optional
from numpy.typing import NDArray
from PyQt6 import QtWidgets, QtCore

from pycroglia.ui.widgets.two_column_list import TwoColumnList
from pycroglia.ui.widgets.img_viewer import ImageViewer
from pycroglia.core.labeled_cells import (
    LabeledCells,
    LabelingStrategy,
    SkimageImgLabeling,
)
from pycroglia.core.enums import SkimageCellConnectivity
from pycroglia.core.files import TiffReader
from pycroglia.core.filters import remove_small_objects, calculate_otsu_threshold


class CellListWidget(QtWidgets.QWidget):
    def __init__(self, headers: list[str], parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent)

        # Widgets
        self.list = TwoColumnList(headers=headers, parent=self)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list)
        self.setLayout(layout)

    @property
    def selectionChanged(self) -> QtCore.pyqtSignal:
        return self.list.selectionChanged

    def get_selected_cell(self) -> Optional[int]:
        selected = self.list.get_selected_item()
        if selected:
            return int(selected[0])
        return None

    def add_cells(self, cells: LabeledCells):
        list_of_cells = sorted(
            [(i, cells.get_cell_size(i)) for i in range(1, cells.len() + 1)],
            key=lambda x: x[1],
            reverse=True,
        )

        for cell in list_of_cells:
            self.list.add_item(str(cell[0]), str(cell[1]))


class MultiCellImageViewer(QtWidgets.QWidget):
    DEFAULT_RGB_SEED: int = 42

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent)

        # Widgets
        self.img_viewer = ImageViewer(parent=self)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.img_viewer)
        self.setLayout(layout)

    def set_cells_img(self, cells: LabeledCells, rgb_seed: int = DEFAULT_RGB_SEED):
        label_2d = cells.labels_to_2d()
        n_cells = cells.len()

        rng = np.random.default_rng(rgb_seed)
        lut = np.zeros((n_cells + 1, 4), dtype=np.uint8)
        lut[0] = (0, 0, 0, 255)
        lut[1:] = np.concatenate(
            [
                rng.integers(0, 256, size=(n_cells, 3), dtype=np.uint8),
                np.full((n_cells, 1), 255, dtype=np.uint8),
            ],
            axis=1,
        )

        self.img_viewer.set_image(label_2d)
        self.img_viewer.set_lookup_table(lut)


class SegmentationEditor(QtWidgets.QWidget):
    HEADERS_TEXT = ["Cell number", "Cell size"]

    def __init__(
        self,
        img: NDArray,
        labeling_strategy: LabelingStrategy,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent=parent)

        self.labeled_cells = LabeledCells(img=img, labeling_strategy=labeling_strategy)

        # Widgets
        self.list = CellListWidget(headers=self.HEADERS_TEXT, parent=self)
        self.multi_cell_viewer = MultiCellImageViewer(parent=self)
        self.cell_viewer = ImageViewer(parent=self)

        # Load data
        self.list.add_cells(self.labeled_cells)
        self.multi_cell_viewer.set_cells_img(self.labeled_cells)

        self.list.selectionChanged.connect(self._on_cell_selected)

        # Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.list, stretch=1)
        layout.addWidget(self.multi_cell_viewer, stretch=2)

        # For consistency
        # TODO - Improve style
        cell_viewer_layout = QtWidgets.QVBoxLayout()
        cell_viewer_layout.addWidget(self.cell_viewer)
        cell_viewer_container = QtWidgets.QWidget()
        cell_viewer_container.setLayout(cell_viewer_layout)
        layout.addWidget(cell_viewer_container, stretch=2)

        self.setLayout(layout)

    def _on_cell_selected(self):
        selected_cell = self.list.get_selected_cell()
        if not selected_cell:
            return

        cell_2d = self.labeled_cells.cell_to_2d(selected_cell)
        self.cell_viewer.set_image(cell_2d)


# --- Replace with your TIFF file path ---
TIFF_PATH = "/Users/framos/Desktop/TrialControlZip.tif"
CHANNELS = 5  # Set according to your file
CHANNEL_OF_INTEREST = 2  # 1-based index


def main():
    # Read TIFF file
    reader = TiffReader(TIFF_PATH)
    img = reader.read(CHANNELS, CHANNEL_OF_INTEREST)
    img = calculate_otsu_threshold(img, 1.0)
    img = remove_small_objects(img, 10)

    # Create a dummy labeling strategy (replace with your actual strategy)
    labeling_strategy = SkimageImgLabeling(SkimageCellConnectivity.FACES)

    # Create the application and widget
    app = QtWidgets.QApplication(sys.argv)
    editor = SegmentationEditor(img, labeling_strategy)
    editor.show()
    sys.exit(app.exec())


main()
