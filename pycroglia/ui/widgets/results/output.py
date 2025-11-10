from PyQt6 import QtWidgets, QtCore
from typing import Optional, Type

from pycroglia.core.io.output import OutputWriter
from pycroglia.ui.widgets.common.labeled_widgets import LabeledLineEdit
from pycroglia.ui.widgets.io.folder_selector import FolderSelector
from pycroglia.ui.widgets.results.writer import OutputWriterSelector


class OutputConfigurator(QtWidgets.QWidget):
    """Widget that groups selection of an output writer and a destination folder.

    Provides two sub-widgets: one to choose the writer implementation and another
    to select the target folder where output will be written.

    Attributes:
        writer_selector (OutputWriterSelector): Widget for selecting an output writer.
        folder_selector (FolderSelector): Widget for selecting the destination folder.
    """

    DEFAULT_SAVE_BUTTON_TXT = "Save"
    DEFAULT_FILENAME_PLACEHOLDER_TXT = "Filename"

    # Folder path, filename, list of writers
    buttonCliched = QtCore.pyqtSignal(str, str, object)

    def __init__(
        self,
        writer_widget_title: str,
        folder_selection_label: str,
        folder_button_txt: str,
        folder_path_display_text: str,
        folder_dialog_title: str,
        folder_dialog_path: str = QtCore.QDir.homePath(),
        save_button_txt: Optional[str] = None,
        filename_placeholder: Optional[str] = None,
        writers: Optional[Type[OutputWriter]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the OutputConfigurator widget.

        Args:
            writer_widget_title (str): Title text shown above the writer selection widget.
            folder_selection_label (str): Label text describing the folder selection section.
            folder_button_txt (str): Text displayed on the button that opens the folder dialog.
            folder_path_display_text (str): Placeholder text for the folder path display field.
            folder_dialog_title (str): Title used for the folder selection dialog window.
            folder_dialog_path (str): Initial directory path when the folder dialog opens.
            writers (Optional[Type[OutputWriter]]): Collection or type providing available writer classes.
                If None, uses OutputWriter.get_writers().
            parent (Optional[QtWidgets.QWidget]): Optional parent widget.
        """
        super().__init__(parent=parent)

        # State
        self.results_ready = False

        # Widgets
        self.writer_selector = OutputWriterSelector(
            writers=writers if writers else OutputWriter.get_writers(),
            title_text=writer_widget_title,
            parent=self,
        )
        self.folder_selector = FolderSelector(
            label_text=folder_selection_label,
            button_text=folder_button_txt,
            path_display_text=folder_path_display_text,
            dialog_title=folder_dialog_title,
            dialog_path=folder_dialog_path,
            parent=self,
        )

        self.filename_input = LabeledLineEdit(
            label_text=filename_placeholder or self.DEFAULT_FILENAME_PLACEHOLDER_TXT,
            parent=self,
        )
        self.button = QtWidgets.QPushButton(
            save_button_txt or self.DEFAULT_SAVE_BUTTON_TXT, parent=self
        )
        self.button.setEnabled(False)

        self.button.clicked.connect(self._on_button_clicked)
        self.folder_selector.folderSelected.connect(self._on_status_changed)
        self.writer_selector.itemChanged.connect(self._on_status_changed)
        self.filename_input.valueChanged.connect(self._on_status_changed)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.writer_selector)
        layout.addWidget(self.folder_selector)
        layout.addWidget(self.filename_input)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def set_results_ready(self, ready: bool):
        self.results_ready = ready
        self._on_status_changed()

    def _on_status_changed(self):
        if (
            self.writer_selector.has_selected_writers()
            and self.folder_selector.has_folder_selected()
            and self.filename_input.has_text()
            and self.results_ready
        ):
            self.button.setEnabled(True)
        else:
            self.button.setEnabled(False)

    def _on_button_clicked(self):
        folder = self.folder_selector.get_selected_folder()
        file_name = self.filename_input.get_text()
        writers = self.writer_selector.get_selected_writers()

        self.buttonCliched.emit(folder, file_name, writers)
