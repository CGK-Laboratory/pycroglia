from PyQt6 import QtWidgets, QtCore
from typing import Optional


class FileSelector(QtWidgets.QWidget):
    """Widget for selecting a file from the filesystem.

    Attributes:
        label_text (str): Text for the label.
        button_text (str): Text for the button.
        dialog_title (str): Title for the file dialog.
        file_filters (str): File filters for the dialog.
        dialog_path (str): Initial path for the dialog.
        label (QtWidgets.QLabel): Label widget.
        button (QtWidgets.QPushButton): Button widget.
        fileSelected (QtCore.pyqtSignal): Signal emitted when a file is selected.
    """

    # Signals
    fileSelected = QtCore.pyqtSignal(str)

    def __init__(
        self,
        label_text: str,
        button_text: str,
        dialog_title: str,
        file_filters: str,
        dialog_path: str = QtCore.QDir.homePath(),
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the file selector.

        Args:
            label_text (str): Label text.
            button_text (str): Button text.
            dialog_title (str): Dialog title.
            file_filters (str): File filters.
            dialog_path (str): Initial dialog path.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        # Configuration
        self.label_text = label_text
        self.button_text = button_text
        self.dialog_title = dialog_title
        self.file_filters = file_filters
        self.dialog_path = dialog_path

        # Widgets
        self.label = QtWidgets.QLabel()
        self.label.setText(self.label_text)

        self.button = QtWidgets.QPushButton(self.button_text)
        self.button.clicked.connect(self._on_button_click)

        # Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def _on_button_click(self):
        """Handle the button click event to open the file dialog."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption=self.dialog_title,
            filter=self.file_filters,
            directory=self.dialog_path,
        )
        if file_path:
            self.fileSelected.emit(file_path)
