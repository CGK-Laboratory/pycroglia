from PyQt6 import QtWidgets, QtCore, QtGui
from typing import Optional, List


class TwoColumnList(QtWidgets.QWidget):
    """Widget for displaying a two-column list with optional selection and data change signals.

    Attributes:
        headers (List[str]): Column headers.
        table_view (QtWidgets.QTableView): Table view widget.
        model (QtGui.QStandardItemModel): Model for the table view.
        dataChanged (QtCore.pyqtSignal): Signal emitted when the data changes.
        selectionChanged (QtCore.pyqtSignal): Signal emitted when the selection changes.
    """

    DEFAULT_HEADER_RESIZE_MODES = [
        QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
        QtWidgets.QHeaderView.ResizeMode.Stretch,
    ]

    dataChanged = QtCore.pyqtSignal()
    selectionChanged = QtCore.pyqtSignal()

    def __init__(self, headers: List[str], parent: Optional[QtWidgets.QWidget] = None):
        """Initialize the two-column list widget.

        Args:
            headers (List[str]): Column headers.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        # Configuration
        self.headers: List[str] = headers

        # Table view - Behavior
        self.table_view = QtWidgets.QTableView()
        self.table_view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table_view.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )

        # Table view -Header
        header = self.table_view.horizontalHeader()
        for i, mode in enumerate(self.DEFAULT_HEADER_RESIZE_MODES):
            header.setSectionResizeMode(i, mode)
        header.setStretchLastSection(True)
        self.table_view.verticalHeader().hide()

        # Table model
        self.model = QtGui.QStandardItemModel(0, 2)
        self.model.setHorizontalHeaderLabels(self.headers)
        self.table_view.setModel(self.model)

        # Connections
        self.table_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table_view)

        self.setLayout(layout)

    def add_item(self, first_row: str, second_row: str):
        """Add an item to the list.

        Args:
            first_row (str): Value for the first column.
            second_row (str): Value for the second column.
        """
        first_row = QtGui.QStandardItem(first_row)
        second_row = QtGui.QStandardItem(second_row)

        first_row.setEditable(False)
        second_row.setEditable(False)

        self.model.appendRow([first_row, second_row])
        self.dataChanged.emit()

    def get_column(self, column_index: int) -> List[str]:
        """Return all items from the specified column (excluding header).

        Args:
            column_index (int): Index of the column (0 or 1).

        Returns:
            List[str]: List of items in the specified column.

        Raises:
            ValueError: If column_index is not 0 or 1.
        """
        if column_index >= 2:
            raise ValueError("Column index must be 0 or 1.")

        return [
            self.model.item(row, column_index).text()
            for row in range(self.model.rowCount())
        ]

    def get_selected_item(self) -> Optional[tuple[str, str]]:
        """Return the currently selected item as a tuple.

        Returns:
            Optional[tuple[str, str]]: Tuple of (first_column, second_column) if selected, else None.
        """
        indexes = self.table_view.selectionModel().selectedRows()
        if not indexes:
            return None

        row = indexes[0].row()
        return (self.model.item(row, 0).text(), self.model.item(row, 1).text())

    def _on_selection_changed(self):
        """Emit the selectionChanged signal when the selection changes."""
        self.selectionChanged.emit()


class TwoColumnListWithDelete(TwoColumnList):
    """Widget for displaying a two-column list with a delete button.

    Attributes:
        delete_button (QtWidgets.QPushButton): Button to delete selected items.
    """

    def __init__(
        self,
        headers: List[str],
        delete_button_text: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the two-column list widget with a delete button.

        Args:
            headers (List[str]): Column headers.
            delete_button_text (str): Text for the delete button.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(headers=headers, parent=parent)

        # Delete button
        self.delete_button = QtWidgets.QPushButton(delete_button_text)
        self.delete_button.setEnabled(False)

        # Connections
        self.delete_button.clicked.connect(self._remove_selected_item)

        # Layout
        self.layout().addWidget(self.delete_button)

    def _on_selection_changed(self):
        """Enable or disable the delete button based on selection and emit signal."""
        super()._on_selection_changed()
        has_selection = self.table_view.selectionModel().hasSelection()
        self.delete_button.setEnabled(has_selection)

    def _remove_selected_item(self):
        """Remove the currently selected item(s) from the list."""
        selection_model = self.table_view.selectionModel()
        selected_rows = selection_model.selectedRows()

        for model_index in reversed(selected_rows):
            self.model.removeRow(model_index.row())

        self.dataChanged.emit()
