from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtWidgets
from numpy.typing import NDArray

from pycroglia.core.enums import SkimageCellConnectivity
from pycroglia.core.erosion import Octahedron3DFootprint
from pycroglia.core.labeled_cells import LabelingStrategy, LabeledCells, MaskListLabeling
from pycroglia.core.segmentation import segment_single_cell, SegmentationConfig


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
                        min_size=self._min_size,
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
