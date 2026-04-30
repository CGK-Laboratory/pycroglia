from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtWidgets
from numpy.typing import NDArray

from pycroglia.core.enums import SkimageCellConnectivity
from pycroglia.core.erosion import Diamond2DFootprint
from pycroglia.core.labeled_cells import LabelingStrategy, LabeledCells
from pycroglia.core.segmentation import segment_single_cell, SegmentationConfig


class SegmentationEditorState(QtCore.QObject):
    """Manages the state of cell segmentation in the editor.

    Handles the current and previous segmentation states, segmentation operations,
    and progress bar updates.

    Attributes:
        ARRAY_ELEMENTS_TYPE (type): Data type for output arrays.
        DEFAULT_SKIMAGE_CONNECTIVITY (SkimageCellConnectivity): Default connectivity for labeling.
        DEFAULT_PROGRESS_BAR_TEXT (str): Default text for the progress bar.
    """

    ARRAY_ELEMENTS_TYPE = np.uint8

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
        erosion_radius: int = 3,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initializes the segmentation editor state.

        Args:
            img (NDArray): 3D binary image.
            labeling_strategy (LabelingStrategy): Strategy for labeling connected components.
            min_size (int): Minimum size for objects to keep after noise removal.
            erosion_radius (int): Radius for the diamond erosion footprint. Default is 3.
            parent (Optional[QtWidgets.QWidget], optional): Parent widget. Defaults to None.
        """
        super().__init__(parent=parent)

        self._shape = img.shape

        self._actual_state = LabeledCells(img, labeling_strategy)
        self._prev_state: Optional[LabeledCells] = None
        self._min_size = min_size
        self._erosion_footprint = Diamond2DFootprint(r=erosion_radius)

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
        from pycroglia.core.labeled_cells import PrecomputedLabeling

        # If progress bar was passed
        if progress_bar:
            progress_bar.setMaximum(100)
            progress_bar.setValue(0)
            progress_bar.setLabelText(f"Segmenting cell {cell_index}...")
            QtCore.QCoreApplication.processEvents()

        # Step 1: Segment the requested cell into multiple new sub-cells
        segmented_cells = segment_single_cell(
            cell_matrix=self._actual_state.get_cell(cell_index),
            footprint=self._erosion_footprint,
            config=SegmentationConfig(
                cut_off_size=cell_size,
                min_size=self._min_size,
                connectivity=self.DEFAULT_SKIMAGE_CONNECTIVITY,
            ),
        )
        # Step 2: Copy existing labels and remove the cell we are segmenting
        new_labels = self._actual_state.labels.copy()
        new_labels[new_labels == cell_index] = 0

        if progress_bar and progress_bar.wasCanceled():
            return

        # Step 3: Insert the new sub-cells into the label array
        # Reuse the original cell_index for the first sub-cell to avoid gaps.
        # Assign new sequential labels for the rest.
        max_label = new_labels.max()
        for idx, mask in enumerate(segmented_cells):
            if idx == 0:
                new_labels[mask > 0] = cell_index
            else:
                max_label += 1
                new_labels[mask > 0] = max_label

        if progress_bar:
            progress_bar.setValue(100)

        new_state = LabeledCells(
            np.zeros(self._shape, dtype=self.ARRAY_ELEMENTS_TYPE),
            PrecomputedLabeling(new_labels),
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
