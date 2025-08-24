import sys
import numpy as np

from typing import Optional, Tuple
from numpy.typing import NDArray
from PyQt6 import QtWidgets, QtCore

from pycroglia.ui.widgets.two_column_list import TwoColumnList
from pycroglia.ui.widgets.img_viewer import ImageViewer
from pycroglia.core.labeled_cells import (
    LabeledCells,
    LabelingStrategy,
    SkimageImgLabeling,
    MaskListLabeling
)
from pycroglia.core.enums import SkimageCellConnectivity
from pycroglia.core.erosion import Octahedron3DFootprint
from pycroglia.core.files import TiffReader
from pycroglia.core.filters import remove_small_objects, calculate_otsu_threshold
from pycroglia.core.segmentation import segment_single_cell, SegmentationConfig


class CellSegmentationList(QtWidgets.QWidget):
    cellSelectionChanged = QtCore.pyqtSignal()

    def __init__(self,
                 headers: list[str],
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent)

        # Widgets
        self.list = TwoColumnList(headers=headers, parent=self)

        # Connections
        self.list.selectionChanged.connect(self._on_selection_changed)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list)

        # Style
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        self.setLayout(layout)

    def clear_cells(self):
        self.list.model.clear()
        self.list.model.setHorizontalHeaderLabels(self.list.headers)
        self.list.dataChanged.emit()

    def _on_selection_changed(self):
        self.cellSelectionChanged.emit()

    @property
    def selectionChanged(self) -> QtCore.pyqtSignal:
        return self.list.selectionChanged

    def get_selected_cell_id(self) -> Optional[int]:
        selected = self.list.get_selected_item()
        if selected:
            return int(selected[0])
        return None

    def get_selected_cell_info(self) -> Optional[Tuple[int, int]]:
        selected = self.list.get_selected_item()
        if selected:
            return int(selected[0]), int(selected[1])
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
    ROLLBACK_BUTTON_TEXT = "Roll back segmentation"
    SEGMENTATION_BUTTON_TEXT = "Segment Cell"

    SEGMENTATION_DEFAULT_FOOTPRINT = Octahedron3DFootprint(r=1)
    SEGMENTATION_DEFAULT_CONNECTIVITY = SkimageCellConnectivity.CORNERS

    def __init__(
        self,
        img: NDArray,
        labeling_strategy: LabelingStrategy,
        noise: int,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent=parent)

        self.labeled_cells = LabeledCells(img=img, labeling_strategy=labeling_strategy)
        self.shape = (self.labeled_cells.z, self.labeled_cells.y, self.labeled_cells.x)
        self.prev_state: Optional[LabeledCells] = None
        self.noise = noise

        # Widgets
        self.list = CellSegmentationList(headers=self.HEADERS_TEXT, parent=self)
        self.segment_button = QtWidgets.QPushButton(self.SEGMENTATION_BUTTON_TEXT, parent=self)
        self.segment_button.setEnabled(False)
        self.rollback_button = QtWidgets.QPushButton(self.ROLLBACK_BUTTON_TEXT, parent=self)
        self.rollback_button.setEnabled(False)

        self.multi_cell_viewer = MultiCellImageViewer(parent=self)
        self.cell_viewer = ImageViewer(parent=self)

        # Load data
        self.list.add_cells(self.labeled_cells)
        self.multi_cell_viewer.set_cells_img(self.labeled_cells)

        # Connections
        self.list.selectionChanged.connect(self._on_cell_selected)
        self.list.cellSelectionChanged.connect(self._on_selection_changed)
        self.segment_button.clicked.connect(self._on_cell_segmentation_request)
        self.rollback_button.clicked.connect(self._on_rollback_request)

        # Layout
        list_layout = QtWidgets.QVBoxLayout()
        list_layout.addWidget(self.list, stretch=1)
        list_layout.addWidget(self.segment_button)
        list_layout.addWidget(self.rollback_button)
        list_container = QtWidgets.QWidget()
        list_container.setLayout(list_layout)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(list_container, stretch=1)
        layout.addWidget(self.multi_cell_viewer, stretch=2)

        # For consistency
        # TODO - Improve style
        cell_viewer_layout = QtWidgets.QVBoxLayout()
        cell_viewer_layout.addWidget(self.cell_viewer)
        cell_viewer_container = QtWidgets.QWidget()
        cell_viewer_container.setLayout(cell_viewer_layout)
        layout.addWidget(cell_viewer_container, stretch=2)

        self.setLayout(layout)

    def _on_selection_changed(self):
        has_selection = self.list.get_selected_cell_id() is not None
        self.segment_button.setEnabled(has_selection)

    def _on_cell_selected(self):
        selected_cell = self.list.get_selected_cell_id()
        if not selected_cell:
            return

        cell_2d = self.labeled_cells.cell_to_2d(selected_cell)
        self.cell_viewer.set_image(cell_2d)

    def _on_cell_segmentation_request(self):
        selected_cell_info = self.list.get_selected_cell_info()
        if not selected_cell_info:
            return
        cell_to_segment, cell_size = selected_cell_info

        list_of_cells: list[NDArray] = []
        number_of_cells = self.labeled_cells.len()

        # Progress dialog popup
        progress_dialog = QtWidgets.QProgressDialog(
            "Segmenting cells...", "Cancel", 0, number_of_cells, self
        )
        progress_dialog.setWindowTitle("Segmentation Progress")
        progress_dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)

        cancelled = False

        for i in range(1, number_of_cells + 1):
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                cancelled = True
                break

            if cell_to_segment == i:
                segmented_cell = segment_single_cell(
                    cell_matrix=self.labeled_cells.get_cell(i),
                    footprint=self.SEGMENTATION_DEFAULT_FOOTPRINT,
                    config=SegmentationConfig(
                        cut_off_size=cell_size,
                        noise=self.noise,
                        connectivity=self.SEGMENTATION_DEFAULT_CONNECTIVITY
                    )
                )
                list_of_cells.extend(segmented_cell)
            else:
                list_of_cells.append(self.labeled_cells.get_cell(i))

            progress_dialog.setValue(i)

        progress_dialog.close()

        if cancelled:
            return

        # Save prev state
        # TODO - Could be abstracted
        self.prev_state = self.labeled_cells

        # Creates new state
        label_strategy = MaskListLabeling(list_of_cells)
        # TODO - Hardcoded dtype
        self.labeled_cells = LabeledCells(np.zeros(self.shape, dtype=np.uint8), label_strategy)
        self.list.clear_cells()
        self.list.add_cells(self.labeled_cells)
        self.multi_cell_viewer.set_cells_img(self.labeled_cells)
        self.rollback_button.setEnabled(True)
        self.segment_button.setEnabled(False)

    def _on_rollback_request(self):
        if self.prev_state is None:
            return

        self.labeled_cells = self.prev_state
        self.prev_state = None

        self.list.clear_cells()
        self.list.add_cells(self.labeled_cells)
        self.multi_cell_viewer.set_cells_img(self.labeled_cells)



# --- Replace with your TIFF file path ---
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

    # Create the application and widget
    app = QtWidgets.QApplication(sys.argv)
    editor = SegmentationEditor(img, labeling_strategy, noise)
    editor.show()
    sys.exit(app.exec())


main()
