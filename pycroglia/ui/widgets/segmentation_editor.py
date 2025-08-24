import sys
import numpy as np

from typing import Optional, Tuple
from numpy.typing import NDArray
from PyQt6 import QtCore, QtWidgets

from pycroglia.core.enums import SkimageCellConnectivity
from pycroglia.core.erosion import Octahedron3DFootprint
from pycroglia.core.segmentation import segment_single_cell, SegmentationConfig
from pycroglia.core.labeled_cells import (
    LabeledCells,
    LabelingStrategy,
    SkimageImgLabeling,
    MaskListLabeling,
)

from pycroglia.ui.widgets.two_column_list import TwoColumnList
from pycroglia.ui.widgets.img_viewer import ImageViewer

# TODO - To delete, these are only for testing
from pycroglia.core.files import TiffReader
from pycroglia.core.filters import remove_small_objects, calculate_otsu_threshold


class SegmentationEditorState(QtCore.QObject):
    ARRAY_ELEMENTS_TYPE = np.uint8

    DEFAULT_EROSION_FOOTPRINT = Octahedron3DFootprint(r=1)
    DEFAULT_SKIMAGE_CONNECTIVITY = SkimageCellConnectivity.CORNERS

    DEFAULT_PROGRESS_BAR_TEXT = "Processing cells..."

    @staticmethod
    def DEFAULT_PROGRESS_BAR_TEXT_GENERATOR(cell, total):
        return f"Processing cell {cell} of {total}"

    stateChanged = QtCore.pyqtSignal()

    def __init__(
        self,
        img: NDArray,
        labeling_strategy: LabelingStrategy,
        min_size: int,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent=parent)

        self._shape = img.shape

        self._actual_state = LabeledCells(img, labeling_strategy)
        self._prev_state: Optional[LabeledCells] = None

        self._min_size = min_size

    def get_state(self) -> LabeledCells:
        return self._actual_state

    def has_prev_state(self) -> bool:
        return self._prev_state is not None

    def _update_state(self, new_state: LabeledCells):
        self._prev_state = self._actual_state
        self._actual_state = new_state

    def segment_cell(
        self,
        cell_index: int,
        cell_size: int,
        progress_bar: Optional[QtWidgets.QProgressDialog] = None,
    ):
        list_of_cells: list[NDArray] = []
        number_of_cells = self._actual_state.len()

        # If progress bar was passed
        if progress_bar:
            progress_bar.setMaximum(number_of_cells)
            progress_bar.setValue(0)
            progress_bar.setLabelText(self.DEFAULT_PROGRESS_BAR_TEXT)
            QtCore.QCoreApplication.processEvents()

        for i in range(1, number_of_cells + 1):
            if progress_bar:
                progress_bar.setValue(i)
                progress_bar.setLabelText(
                    self.DEFAULT_PROGRESS_BAR_TEXT_GENERATOR(i, number_of_cells)
                )
                QtCore.QCoreApplication.processEvents()

                if progress_bar.wasCanceled():
                    return

            if cell_index == i:
                segmented_cell = segment_single_cell(
                    cell_matrix=self._actual_state.get_cell(i),
                    footprint=self.DEFAULT_EROSION_FOOTPRINT,
                    config=SegmentationConfig(
                        cut_off_size=cell_size,
                        noise=self._min_size,
                        connectivity=self.DEFAULT_SKIMAGE_CONNECTIVITY,
                    ),
                )
                list_of_cells.extend(segmented_cell)
            else:
                list_of_cells.append(self._actual_state.get_cell(i))

        if progress_bar:
            progress_bar.setValue(number_of_cells)

        new_state = LabeledCells(
            np.zeros(self._shape, dtype=self.ARRAY_ELEMENTS_TYPE),
            MaskListLabeling(list_of_cells),
        )
        self._update_state(new_state)
        self.stateChanged.emit()

    def rollback(self):
        if self._prev_state is None:
            return

        self._actual_state = self._prev_state
        self._prev_state = None
        self.stateChanged.emit()


class CellList(QtWidgets.QWidget):
    def __init__(self, headers: list[str], parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent)

        # Widgets
        self.list = TwoColumnList(headers=headers, parent=self)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list)

        # Style
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setLayout(layout)

    @property
    def selectionChanged(self) -> QtCore.pyqtSignal:
        return self.list.selectionChanged

    def add_cells(self, cells: LabeledCells):
        list_of_cells = sorted(
            [(i, cells.get_cell_size(i)) for i in range(1, cells.len() + 1)],
            key=lambda x: x[1],
            reverse=True,
        )

        for cell in list_of_cells:
            self.list.add_item(str(cell[0]), str(cell[1]))

    def clear_cells(self):
        self.list.model.clear()
        self.list.model.setHorizontalHeaderLabels(self.list.headers)
        self.list.dataChanged.emit()

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

    def __init__(
        self,
        img: NDArray,
        labeling_strategy: LabelingStrategy,
        min_size: int,
        with_progress_bar: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent=parent)

        self.state = SegmentationEditorState(img, labeling_strategy, min_size)
        self.with_progress_bar = with_progress_bar

        # Widgets
        self.list = CellList(headers=self.HEADERS_TEXT, parent=self)

        self.segment_button = QtWidgets.QPushButton(self.SEGMENTATION_BUTTON_TEXT)
        self.segment_button.setEnabled(False)

        self.rollback_button = QtWidgets.QPushButton(self.ROLLBACK_BUTTON_TEXT)
        self.rollback_button.setEnabled(False)

        self.multi_cell_viewer = MultiCellImageViewer(parent=self)
        self.cell_viewer = ImageViewer(parent=self)

        # Connections
        self.list.selectionChanged.connect(self._on_cell_selection_changed)
        self.segment_button.clicked.connect(self._on_cell_segmentation_request)
        self.rollback_button.clicked.connect(self._on_rollback_request)
        self.state.stateChanged.connect(self._load_data)

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

        # Loads data
        self._load_data()

    def _load_data(self):
        actual_state = self.state.get_state()

        self.list.clear_cells()
        self.list.add_cells(actual_state)
        self.multi_cell_viewer.set_cells_img(actual_state)

        self.rollback_button.setEnabled(self.state.has_prev_state())

    def _on_cell_selection_changed(self):
        selected_cell = self.list.get_selected_cell_id()
        self.segment_button.setEnabled(selected_cell is not None)
        if selected_cell is None:
            return

        cell_2d = self.state.get_state().cell_to_2d(selected_cell)
        self.cell_viewer.set_image(cell_2d)

    def _on_cell_segmentation_request(self):
        selected_cell_info = self.list.get_selected_cell_info()
        if selected_cell_info is None:
            return

        progress_bar = None
        if self.with_progress_bar:
            # TODO - Hardcoded values
            progress_bar = QtWidgets.QProgressDialog(
                "Segmenting cell...", "Cancel", 0, 100, self
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
        self.state.rollback()


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
    editor = SegmentationEditor(img, labeling_strategy, noise, True)
    editor.show()
    sys.exit(app.exec())


main()
