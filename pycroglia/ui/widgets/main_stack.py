from typing import Optional, List
from pathlib import Path
from PyQt6 import QtWidgets, QtCore

from pycroglia.ui.widgets.file_selection_editor import FileSelectionEditor
from pycroglia.ui.widgets.ch_img import MultiChannelFilterEditor


class FilterEditorStack(QtWidgets.QWidget):
    """Widget that manages a tabbed interface for multi-channel filter editors.

    Attributes:
        tabs (QtWidgets.QTabWidget): Tab widget containing filter editors.
        _previous_tab_index (int): Index of the previously selected tab.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """Initialize the FilterEditorStack widget.

        Args:
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        self._previous_tab_index = -1

        # Widgets
        self.tabs = QtWidgets.QTabWidget(self)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def add_tabs(self, files: List[str]):
        """Clear and add a tab for each file, each with a MultiChannelFilterEditor.

        Args:
            files (List[str]): List of file paths to add as tabs.
        """
        self.tabs.clear()

        for file in files:
            editor = MultiChannelFilterEditor(file, parent=self)
            self.tabs.addTab(editor, f"{Path(file).name}")


class MainStack(QtWidgets.QWidget):
    """Main widget that manages the application workflow using a stacked layout.

    Attributes:
        stacked (QtWidgets.QStackedWidget): Stacked widget for navigation.
        page_1_file_editor (FileSelectionEditor): File selection editor widget.
        page_1_next_btn (QtWidgets.QPushButton): Button to proceed to the next page.
        page_2_filter_tabs (FilterEditorStack): Tabbed filter editor stack.
        page_2_back_btn (QtWidgets.QPushButton): Button to go back to the previous page.
        page_2_next_btn (QtWidgets.QPushButton): Button to proceed to the next step.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """Initialize the MainStack widget.

        Args:
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        # Widgets
        self.stacked = QtWidgets.QStackedWidget(self)

        page_1 = QtWidgets.QWidget(self)
        self.page_1_file_editor = FileSelectionEditor(
            headers=["File Type", "File Path"],
            delete_button_text="Delete",
            open_file_text="Select Files:",
            open_button_text="Open",
            open_dialog_title="Open File",
            open_dialog_default_path=QtCore.QDir.homePath(),
            file_filters="All Files (*);;Image Files (*.lsm *.tiff *.tif)",
            parent=self,
        )
        self.page_1_next_btn = QtWidgets.QPushButton("Next")

        page_2 = QtWidgets.QWidget(self)
        self.page_2_filter_tabs = FilterEditorStack(self)
        self.page_2_back_btn = QtWidgets.QPushButton("Back")
        self.page_2_next_btn = QtWidgets.QPushButton("Next")

        # Connections
        self.page_1_next_btn.clicked.connect(self._on_page_1_next_press)
        self.page_2_back_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(0))

        # Layout
        page_1_layout = QtWidgets.QVBoxLayout()
        page_1_layout.addWidget(self.page_1_file_editor)
        page_1_layout.addWidget(self.page_1_next_btn)
        page_1.setLayout(page_1_layout)

        page_2_layout = QtWidgets.QVBoxLayout()
        page_2_layout.addWidget(self.page_2_filter_tabs)
        page_2_btn_layout = QtWidgets.QHBoxLayout()
        page_2_btn_layout.addWidget(self.page_2_back_btn)
        page_2_btn_layout.addWidget(self.page_2_next_btn)
        page_2_layout.addLayout(page_2_btn_layout)
        page_2.setLayout(page_2_layout)

        self.stacked.addWidget(page_1)
        self.stacked.addWidget(page_2)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.stacked)
        self.setLayout(main_layout)

    def _on_page_1_next_press(self):
        """Handle the event when the 'Next' button is pressed on the first page.

        Retrieves selected file paths and updates the filter editor tabs.
        """
        file_paths = self.page_1_file_editor.get_files()
        self.page_2_filter_tabs.add_tabs(file_paths)
        self.stacked.setCurrentIndex(1)
