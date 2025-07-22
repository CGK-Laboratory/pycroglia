from PyQt6 import QtCore, QtWidgets
from typing import Optional, List
from pathlib import Path

from pycroglia.ui.widgets.file_selector import FileSelector
from pycroglia.ui.widgets.two_column_list import TwoColumnList


class FileSelectionEditor(QtWidgets.QWidget):
    """Widget for selecting and listing files with delete options.

    Attributes:
        file_list (TwoColumnList): Widget displaying the list of files.
        file_selector (FileSelector): Widget for selecting files.
        dataChanged (QtCore.pyqtSignal): Signal emitted when the file list changes.
    """

    dataChanged = QtCore.pyqtSignal()

    def __init__(
        self,
        headers: List[str],
        delete_button_text: str,
        open_file_text: str,
        open_button_text: str,
        open_dialog_title: str,
        open_dialog_default_path: str,
        file_filters: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the file selection editor.

        Args:
            headers (List[str]): Table headers.
            delete_button_text (str): Delete button text.
            open_file_text (str): File selection label text.
            open_button_text (str): Open button text.
            open_dialog_title (str): Open dialog title.
            open_dialog_default_path (str): Default dialog path.
            file_filters (str): File filters.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        # Widgets
        self.file_list = TwoColumnList(
            headers=headers, delete_button_text=delete_button_text, parent=self
        )
        self.file_selector = FileSelector(
            label_text=open_file_text,
            button_text=open_button_text,
            dialog_title=open_dialog_title,
            file_filters=file_filters,
            dialog_path=open_dialog_default_path,
        )

        # Connections
        self.file_selector.fileSelected.connect(self._on_file_added)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.file_list)
        layout.addWidget(self.file_selector)
        self.setLayout(layout)

    def get_files(self):
        """Get the list of file paths.

        Returns:
            List[str]: List of file paths from the file list.
        """
        return [item for item in self.file_list.get_column(1)]

    def _on_file_added(self, path: str):
        """Add a file to the list when selected.

        Args:
            path (str): Path of the selected file.
        """
        if path:
            self.file_list.add_item(Path(path).suffix, path)
            self.dataChanged.emit()
