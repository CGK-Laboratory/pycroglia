from PyQt6 import QtWidgets, QtCore, QtGui
from typing import Optional, List


class TwoColumnList(QtWidgets.QWidget):
    """Widget for displaying a two-column file list with delete option.

    Attributes:
        headers (List[str]): Column headers.
        delete_button_text (str): Text for the delete button.
        table_view (QtWidgets.QTableView): Table view widget.
        model (QtGui.QStandardItemModel): Model for the table view.
        delete_button (QtWidgets.QPushButton): Button to delete selected items.
        dataChanged (QtCore.pyqtSignal): Signal emitted when the data changes.
    """

    dataChanged = QtCore.pyqtSignal()

    def __init__(
        self,
        headers: List[str],
        delete_button_text: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the two-column list widget.

        Args:
            headers (List[str]): Column headers.
            delete_button_text (str): Delete button text.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        # Configuration
        self.headers: List[str] = headers
        self.delete_button_text: str = delete_button_text

        # Table view - Behavior
        self.table_view = QtWidgets.QTableView()
        self.table_view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table_view.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )

        # Table view - Header
        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setStretchLastSection(True)
        self.table_view.verticalHeader().hide()

        # Table model
        self.model = QtGui.QStandardItemModel(0, 2)
        self.model.setHorizontalHeaderLabels(self.headers)
        self.table_view.setModel(self.model)

        # Delete button
        self.delete_button = QtWidgets.QPushButton(delete_button_text)
        self.delete_button.setEnabled(False)

        # Connections
        self.table_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self.delete_button.clicked.connect(self._remove_selected_item)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table_view)
        layout.addWidget(self.delete_button)

        self.setLayout(layout)

    def add_item(self, file_type: str, file_path: str):
        """Add an item to the list.

        Args:
            file_type (str): File type.
            file_path (str): File path.
        """
        type_item = QtGui.QStandardItem(file_type)
        path_item = QtGui.QStandardItem(file_path)

        type_item.setEditable(False)
        path_item.setEditable(False)

        self.model.appendRow([type_item, path_item])
        self.dataChanged.emit()

    def get_column(self, column_index: int) -> List[str]:
        """Return all items from the specified column (excluding header).

        Args:
            column_index (int): Index of the column (0 or 1).

        Returns:
            List[str]: List of items in the specified column.
        """
        if column_index >= 2:
            raise ValueError("Column index must be 0 or 1.")

        return [
            self.model.item(row, column_index).text()
            for row in range(self.model.rowCount())
        ]

    def _on_selection_changed(self):
        """Enable or disable the delete button based on selection."""
        has_selection = self.table_view.selectionModel().hasSelection()
        self.delete_button.setEnabled(has_selection)

    def _remove_selected_item(self):
        """Remove the selected items from the list."""
        selection_model = self.table_view.selectionModel()
        selected_rows = selection_model.selectedRows()

        for model_index in reversed(selected_rows):
            self.model.removeRow(model_index.row())

        self.dataChanged.emit()
