from PyQt6 import QtWidgets, QtCore
from typing import Optional


class FolderSelector(QtWidgets.QWidget):
    """Widget for selecting a folder from the filesystem.

    Attributes:
        label_text (str): Text displayed in the label.
        button_text (str): Text displayed on the selection button.
        dialog_title (str): Title of the folder selection dialog.
        dialog_path (str): Initial path shown when the dialog opens.
        label (QtWidgets.QLabel): Label widget instance.
        button (QtWidgets.QPushButton): Button widget instance.
        folderSelected (QtCore.pyqtSignal): Signal emitted with the selected folder path.
    """

    folderSelected = QtCore.pyqtSignal(str)

    def __init__(
        self,
        label_text: str,
        button_text: str,
        dialog_title: str,
        dialog_path: str = QtCore.QDir.homePath(),
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the folder selector widget.

        Args:
            label_text (str): Text for the label describing the selection.
            button_text (str): Text for the button that opens the dialog.
            dialog_title (str): Title displayed on the folder selection dialog.
            dialog_path (str): Starting directory for the dialog.
            parent (Optional[QtWidgets.QWidget]): Optional parent widget.
        """
        super().__init__(parent=parent)

        # Configuration
        self.label_text = label_text
        self.button_text = button_text
        self.dialog_title = dialog_title
        self.dialog_path = dialog_path

        # Widgets
        self.label = QtWidgets.QLabel(parent=self)
        self.label.setText(self.label_text)

        self.button = QtWidgets.QPushButton(self.button_text, parent=parent)
        self.button.clicked.connect(self._on_button_click)

        # Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def _on_button_click(self):
        """Open a directory selection dialog and emit the chosen path if valid."""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            parent=self, caption=self.dialog_title, directory=self.dialog_path
        )
        if folder_path:
            self.folderSelected.emit(folder_path)
