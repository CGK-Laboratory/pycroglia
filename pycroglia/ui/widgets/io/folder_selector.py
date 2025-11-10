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
        path_display_text: str,
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
        self.path_display_text = path_display_text
        self.dialog_title = dialog_title
        self.dialog_path = dialog_path

        # State
        self._folder_selected = None

        # Widgets
        self.label = QtWidgets.QLabel(parent=self)
        self.label.setText(self.label_text)

        self.button = QtWidgets.QPushButton(self.button_text, parent=self)
        self.button.clicked.connect(self._on_button_click)

        self.path_display = QtWidgets.QLineEdit(parent=self)
        self.path_display.setReadOnly(True)
        self.path_display.setPlaceholderText(self.path_display_text)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout_h = QtWidgets.QHBoxLayout()
        layout_h.addWidget(self.label)
        layout_h.addWidget(self.button)
        layout.addLayout(layout_h)
        layout.addWidget(self.path_display)
        self.setLayout(layout)

    def _on_button_click(self):
        """Open a directory selection dialog and emit the chosen path if valid."""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            parent=self, caption=self.dialog_title, directory=self.dialog_path
        )
        if folder_path:
            self._folder_selected = folder_path
            self.path_display.setText(folder_path)
            self.folderSelected.emit(folder_path)

    def has_folder_selected(self) -> bool:
        return self._folder_selected is not None

    def get_selected_folder(self) -> Optional[str]:
        return self._folder_selected
