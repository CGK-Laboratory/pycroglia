from PyQt6 import QtWidgets, QtCore
from typing import Optional, Type

from pycroglia.core.io.output import OutputWriter
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

    def __init__(
        self,
        writer_widget_title: str,
        folder_selection_label: str,
        folder_button_txt: str,
        folder_path_display_text: str,
        folder_dialog_title: str,
        folder_dialog_path: str = QtCore.QDir.homePath(),
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

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.writer_selector)
        layout.addWidget(self.folder_selector)
        self.setLayout(layout)
