from typing import Optional, Tuple

from PyQt6 import QtWidgets, QtCore

from pycroglia.core.labeled_cells import LabeledCells
from pycroglia.ui.widgets.two_column_list import TwoColumnList


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
