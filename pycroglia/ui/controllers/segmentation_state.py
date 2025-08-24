from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtWidgets
from numpy.typing import NDArray

from pycroglia.core.enums import SkimageCellConnectivity
from pycroglia.core.erosion import Octahedron3DFootprint
from pycroglia.core.labeled_cells import (
    LabelingStrategy,
    LabeledCells,
    MaskListLabeling,
)
from pycroglia.core.segmentation import segment_single_cell, SegmentationConfig


class SegmentationEditorState(QtCore.QObject):
    """Manages the state of cell segmentation in the editor.

    Handles the current and previous segmentation states, segmentation operations,
    and progress bar updates.

    Attributes:
        ARRAY_ELEMENTS_TYPE (type): Data type for output arrays.
        DEFAULT_EROSION_FOOTPRINT (FootprintShape): Default structuring element for erosion.
        DEFAULT_SKIMAGE_CONNECTIVITY (SkimageCellConnectivity): Default connectivity for labeling.
        DEFAULT_PROGRESS_BAR_TEXT (str): Default text for the progress bar.
    """

    ARRAY_ELEMENTS_TYPE = np.uint8

    DEFAULT_EROSION_FOOTPRINT = Octahedron3DFootprint(r=1)
    DEFAULT_SKIMAGE_CONNECTIVITY = SkimageCellConnectivity.CORNERS

    DEFAULT_PROGRESS_BAR_TEXT = "Processing cells..."

    @staticmethod
    def DEFAULT_PROGRESS_BAR_TEXT_GENERATOR(cell, total):
        """Generates progress bar text for the current cell being processed.

        Args:
            cell (int): Current cell index.
            total (int): Total number of cells.

        Returns:
            str: Progress bar label text.
        """
        return f"Processing cell {cell} of {total}"

    stateChanged = QtCore.pyqtSignal()

    def __init__(
        self,
        img: NDArray,
        labeling_strategy: LabelingStrategy,
        min_size: int,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initializes the segmentation editor state.

        Args:
            img (NDArray): 3D binary image.
            labeling_strategy (LabelingStrategy): Strategy for labeling connected components.
            min_size (int): Minimum size for objects to keep after noise removal.
            parent (Optional[QtWidgets.QWidget], optional): Parent widget. Defaults to None.
        """
        super().__init__(parent=parent)

        self._shape = img.shape

        self._actual_state = LabeledCells(img, labeling_strategy)
        self._prev_state: Optional[LabeledCells] = None

        self._min_size = min_size

    def get_state(self) -> LabeledCells:
        """Returns the current segmentation state.

        Returns:
            LabeledCells: Current labeled cells state.
        """
        return self._actual_state

    def has_prev_state(self) -> bool:
        """Checks if there is a previous segmentation state.

        Returns:
            bool: True if a previous state exists, False otherwise.
        """
        return self._prev_state is not None

    def _update_state(self, new_state: LabeledCells):
        """Updates the current state and stores the previous state.

        Args:
            new_state (LabeledCells): New labeled cells state.
        """
        self._prev_state = self._actual_state
        self._actual_state = new_state

    def segment_cell(
        self,
        cell_index: int,
        cell_size: int,
        progress_bar: Optional[QtWidgets.QProgressDialog] = None,
    ):
        """Segments a specific cell and updates the segmentation state.

        Args:
            cell_index (int): Index of the cell to segment.
            cell_size (int): Minimum size for segmentation.
            progress_bar (Optional[QtWidgets.QProgressDialog], optional): Progress dialog to update. Defaults to None.
        """
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
        """Restores the previous segmentation state, if available."""
        if self._prev_state is None:
            return

        self._actual_state = self._prev_state
        self._prev_state = None
        self.stateChanged.emit()
