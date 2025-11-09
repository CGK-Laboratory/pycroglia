import numpy as np
from numpy.typing import NDArray
from typing import Optional, Set, Dict, List
from enum import Enum

from PyQt6 import QtWidgets, QtGui

from pycroglia.core.labeled_cells import LabeledCells
from pycroglia.ui.widgets.cells.cells_panel import CellsPanel
from pycroglia.ui.widgets.common.labeled_widgets import LabeledIntSpinBox
from pycroglia.ui.widgets.analysis.dialog import PreviewDialog


class ColorType(Enum):
    """Enumeration for cell row color types in the cell selector.

    Attributes:
        SELECTED: Color type for selected (normal) cells.
        UNSELECTED: Color type for unselected cells that will be filtered out.
    """

    SELECTED = "selected"
    UNSELECTED = "unselected"


class CellSelectorControlPanel(QtWidgets.QWidget):
    """Control panel widget for cell selection operations.

    Provides buttons and controls for removing cells, filtering by size,
    handling border cells, and previewing results.

    Attributes:
        DEFAULT_REMOVE_BUTTON_TEXT (str): Default text for the remove button.
        DEFAULT_SIZE_LABEL_TEXT (str): Default text for the size input label.
        DEFAULT_SIZE_BUTTON_TEXT (str): Default text for the size filter button.
        DEFAULT_PREVIEW_BUTTON_TEXT (str): Default text for the preview button.
        DEFAULT_BORDER_CHECKBOX_TEXT (str): Default text for the border checkbox.
    """

    DEFAULT_REMOVE_BUTTON_TEXT = "Remove Cell"
    DEFAULT_SIZE_LABEL_TEXT = "Cell Size (pixels)"
    DEFAULT_SIZE_BUTTON_TEXT = "Remove smaller than"
    DEFAULT_PREVIEW_BUTTON_TEXT = "Preview"
    DEFAULT_BORDER_CHECKBOX_TEXT = "Remove border cells"

    def __init__(
        self,
        max_cell_size: int,
        remove_button_text: Optional[str] = None,
        size_label_text: Optional[str] = None,
        size_button_text: Optional[str] = None,
        preview_button_text: Optional[str] = None,
        border_checkbox_text: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initializes the CellSelectorControlPanel widget.

        Args:
            max_cell_size (int): Maximum cell size for the size input spinbox.
            remove_button_text (Optional[str], optional): Custom text for remove button.
            size_label_text (Optional[str], optional): Custom text for size label.
            size_button_text (Optional[str], optional): Custom text for size filter button.
            preview_button_text (Optional[str], optional): Custom text for preview button.
            border_checkbox_text (Optional[str], optional): Custom text for border checkbox.
            parent (Optional[QtWidgets.QWidget], optional): Parent widget.
        """
        super().__init__(parent=parent)

        # Text properties
        self.remove_button_text = remove_button_text or self.DEFAULT_REMOVE_BUTTON_TEXT
        self.size_label_text = size_label_text or self.DEFAULT_SIZE_LABEL_TEXT
        self.size_button_text = size_button_text or self.DEFAULT_SIZE_BUTTON_TEXT
        self.preview_button_text = (
            preview_button_text or self.DEFAULT_PREVIEW_BUTTON_TEXT
        )
        self.border_checkbox_text = (
            border_checkbox_text or self.DEFAULT_BORDER_CHECKBOX_TEXT
        )

        # Widgets
        self.remove_btn = QtWidgets.QPushButton(parent=self)
        self.remove_btn.setText(self.remove_button_text)
        self.remove_btn.setEnabled(False)

        self.size_input = LabeledIntSpinBox(
            label_text=self.size_label_text, min_value=0, max_value=max_cell_size
        )
        self.size_btn = QtWidgets.QPushButton(parent=self)
        self.size_btn.setText(self.size_button_text)

        self.border_checkbox = QtWidgets.QCheckBox(parent=self)
        self.border_checkbox.setText(self.border_checkbox_text)

        self.preview_btn = QtWidgets.QPushButton(parent=self)
        self.preview_btn.setText(self.preview_button_text)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.remove_btn)
        layout.addWidget(self.size_input)
        layout.addWidget(self.size_btn)
        layout.addWidget(self.border_checkbox)
        layout.addWidget(self.preview_btn)
        self.setLayout(layout)


class CellSelector(QtWidgets.QWidget):
    """Widget for interactive cell selection and filtering.

    Provides a comprehensive interface for selecting, filtering, and previewing cells
    from a labeled cell image. Users can remove individual cells, filter by size,
    exclude border cells, and preview the results.

    Attributes:
        DEFAULT_HEADERS_TEXT (list[str]): Default column headers for the cell list.
        DEFAULT_REMOVE_BUTTON_TEXT (str): Default text for the remove button.
        DEFAULT_SIZE_LABEL_TEXT (str): Default text for the size input label.
        DEFAULT_SIZE_BUTTON_TEXT (str): Default text for the size filter button.
        DEFAULT_PREVIEW_BUTTON_TEXT (str): Default text for the preview button.
        DEFAULT_BORDER_CHECKBOX_TEXT (str): Default text for the border checkbox.
        UNSELECTED_COLOR (QtGui.QColor): Color for unselected cell rows.
        SELECTED_COLOR (QtGui.QBrush): Color for selected cell rows.
    """

    # UI Text Constants
    DEFAULT_HEADERS_TEXT = ["Cell number", "Cell size"]
    DEFAULT_REMOVE_BUTTON_TEXT = "Remove Cell"
    DEFAULT_SIZE_LABEL_TEXT = "Cell Size"
    DEFAULT_SIZE_BUTTON_TEXT = "Remove smaller than"
    DEFAULT_PREVIEW_BUTTON_TEXT = "Preview"
    DEFAULT_BORDER_CHECKBOX_TEXT = "Remove border cells"

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
        preview_button_text: Optional[str] = None,
        border_checkbox_text: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initializes the CellSelector widget.

        Args:
            img (LabeledCells): Labeled cells object containing the cell data.
            headers (Optional[list[str]], optional): Custom column headers for cell list.
            remove_button_text (Optional[str], optional): Custom text for remove button.
            size_label_text (Optional[str], optional): Custom text for size label.
            size_button_text (Optional[str], optional): Custom text for size filter button.
            preview_button_text (Optional[str], optional): Custom text for preview button.
            border_checkbox_text (Optional[str], optional): Custom text for border checkbox.
            parent (Optional[QtWidgets.QWidget], optional): Parent widget.
        """
        super().__init__(parent=parent)

        # Text properties
        self.headers_text = headers or self.DEFAULT_HEADERS_TEXT
        self.remove_button_text = remove_button_text or self.DEFAULT_REMOVE_BUTTON_TEXT
        self.size_label_text = size_label_text or self.DEFAULT_SIZE_LABEL_TEXT
        self.size_button_text = size_button_text or self.DEFAULT_SIZE_BUTTON_TEXT
        self.preview_button_text = (
            preview_button_text or self.DEFAULT_PREVIEW_BUTTON_TEXT
        )
        self.border_checkbox_text = (
            border_checkbox_text or self.DEFAULT_BORDER_CHECKBOX_TEXT
        )

        # State
        self.img = img
        self.unselected_cells = set()
        self.border_cells = self.img.get_border_cells()
        self._cell_to_row_cache: Dict[int, int] = {}

        # Widgets
        self.control_panel = CellSelectorControlPanel(
            max_cell_size=max(img.get_cell_size(i) for i in range(1, img.len() + 1)),
            remove_button_text=self.remove_button_text,
            size_label_text=self.size_label_text,
            size_button_text=self.size_button_text,
            preview_button_text=self.preview_button_text,
            border_checkbox_text=self.border_checkbox_text,
            parent=self,
        )
        self.viewer = CellsPanel(
            img=self.img,
            headers=self.headers_text,
            control_panel=self.control_panel,
            parent=self,
        )

        # Connections
        self.viewer.cell_list.selectionChanged.connect(self._on_cell_selection_changed)
        self.control_panel.remove_btn.clicked.connect(self._on_remove_button_clicked)
        self.control_panel.size_btn.clicked.connect(self._on_size_button_clicked)
        self.control_panel.border_checkbox.toggled.connect(
            self._on_border_checkbox_toggled
        )
        self.control_panel.preview_btn.clicked.connect(self._on_preview_button_clicked)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.viewer)
        self.setLayout(layout)

        self._load_data(self.img)

    def _load_data(self, img: LabeledCells):
        """Loads cell data into the viewer and builds the cell-to-row cache.

        Args:
            img (LabeledCells): Labeled cells object to load.
        """
        self.viewer.load_data(img)
        self._build_cell_to_row_cache()

    def _on_cell_selection_changed(self):
        """Handles cell selection changes in the cell list.

        Enables or disables the remove button based on whether a cell is selected.
        """
        self.control_panel.remove_btn.setEnabled(
            self.viewer.cell_list.get_selected_cell_id() is not None
        )

    def _build_cell_to_row_cache(self):
        """Builds a cache mapping cell_id to row index for efficient color updates.

        This cache improves performance when updating row colors by avoiding
        the need to search through all rows for each cell.
        """
        self._cell_to_row_cache.clear()
        model = self.viewer.cell_list.list.model

        for row in range(model.rowCount()):
            cell_id = int(model.item(row, 0).text())
            self._cell_to_row_cache[cell_id] = row

    def _set_row_color(self, cell_id: int, color_type: ColorType):
        """Sets the background color of a specific cell row.

        Args:
            cell_id (int): ID of the cell whose row color to change.
            color_type (ColorType): Type of color to apply (selected or unselected).
        """
        if cell_id not in self._cell_to_row_cache:
            return

        row = self._cell_to_row_cache[cell_id]
        model = self.viewer.cell_list.list.model

        if color_type == ColorType.UNSELECTED:
            color = self.UNSELECTED_COLOR
        else:  # ColorType.SELECTED
            color = self.SELECTED_COLOR

        for col in range(model.columnCount()):
            item = model.item(row, col)
            item.setBackground(color)

    def _update_colors_batch(self, cell_ids: Set[int], color_type: ColorType):
        """Updates colors for multiple cells efficiently in batch.

        Args:
            cell_ids (Set[int]): Set of cell IDs to update.
            color_type (ColorType): Color type to apply to all specified cells.
        """
        for cell_id in cell_ids:
            self._set_row_color(cell_id, color_type)

    def _on_remove_button_clicked(self):
        """Handles the remove button click event.

        Toggles the selection state of the currently selected cell and updates
        its visual appearance accordingly.
        """
        selected_cell = self.viewer.cell_list.get_selected_cell_id()
        if selected_cell is None:
            return

        if selected_cell in self.unselected_cells:
            self.unselected_cells.remove(selected_cell)
            self._set_row_color(selected_cell, ColorType.SELECTED)
        else:
            self.unselected_cells.add(selected_cell)
            self._set_row_color(selected_cell, ColorType.UNSELECTED)

    def _on_size_button_clicked(self):
        """Handles the size filter button click event.

        Marks all cells smaller than the specified threshold as unselected
        and updates their visual appearance.
        """
        threshold = self.control_panel.size_input.get_value()

        cells_to_unselect = set()
        for cell_id in range(1, self.img.len() + 1):
            cell_size = self.img.get_cell_size(cell_id)
            if cell_size < threshold:
                cells_to_unselect.add(cell_id)

        self.unselected_cells.update(cells_to_unselect)
        self._update_colors_batch(cells_to_unselect, ColorType.UNSELECTED)

    def _on_border_checkbox_toggled(self, checked: bool):
        """Handles the border cells checkbox toggle event.

        When checked, marks all border cells as unselected. When unchecked,
        removes border cells from the unselected set and restores their normal color.

        Args:
            checked (bool): Whether the checkbox is checked.
        """
        if checked:
            # Add border cells to unselected and mark as unselected color
            self.unselected_cells.update(self.border_cells)
            self._update_colors_batch(self.border_cells, ColorType.UNSELECTED)
        else:
            # Remove border cells from unselected and restore normal color
            self.unselected_cells.difference_update(self.border_cells)
            self._update_colors_batch(self.border_cells, ColorType.SELECTED)

    def _on_preview_button_clicked(self):
        """Handles the preview button click event.

        Creates a 2D preview of the selected cells by combining their projections
        and displays it in a preview dialog.
        """
        selected_cells = self.get_selected_cells()

        if not selected_cells:
            preview_2d = np.zeros(
                (self.img.y, self.img.x), dtype=self.img.ARRAY_ELEMENTS_TYPE
            )
        else:
            preview_2d = np.zeros(
                (self.img.y, self.img.x), dtype=self.img.ARRAY_ELEMENTS_TYPE
            )
            # Iterate through selected cells and combine their 2D projections
            for cell_id in selected_cells:
                cell_2d = self.img.cell_to_2d(cell_id)
                preview_2d += cell_2d

        dialog = PreviewDialog(preview_2d, parent=self)
        dialog.exec()

    def get_selected_cells(self) -> Set[int]:
        """Returns the set of currently selected (not unselected) cells.

        Returns:
            Set[int]: Set of cell IDs that are currently selected.
        """
        all_cells = set(range(1, self.img.len() + 1))
        return all_cells - self.unselected_cells

    def get_unselected_cells(self) -> Set[int]:
        """Returns a copy of the set of unselected cells.

        Returns:
            Set[int]: Copy of the set of cell IDs that are unselected.
        """
        return self.unselected_cells.copy()

    def get_border_cells(self) -> Set[int]:
        """Returns a copy of the set of border cells.

        Returns:
            Set[int]: Copy of the set of cell IDs that touch the image borders.
        """
        return self.border_cells.copy()

    def get_selected_cells_3d(self) -> NDArray:
        """Return a 3D array containing only the currently selected cells.

        Delegates to LabeledCells.selected_cells_mask to perform a vectorized,
        single-pass generation of the combined mask.

        Returns:
            NDArray: 3D array (z, y, x) with selected cells combined.
        """
        selected = self.get_selected_cells()
        return self.img.selected_cells_mask(selected)

    def get_cells_masks(self) -> List[NDArray]:
        all_masks = self.img.get_cells_list()
        selected_ids = sorted(self.get_selected_cells())
        # map 1-based cell IDs to 0-based list indices and guard bounds
        return [all_masks[i - 1] for i in selected_ids if 1 <= i <= len(all_masks)]
