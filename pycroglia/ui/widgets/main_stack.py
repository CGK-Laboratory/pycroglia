from typing import Optional, List
from pathlib import Path
from PyQt6 import QtWidgets, QtCore

from pycroglia.core.enums import SkimageCellConnectivity
from pycroglia.core.labeled_cells import SkimageImgLabeling
from pycroglia.ui.widgets.io.file_selection_editor import FileSelectionEditor
from pycroglia.ui.widgets.imagefilters.editors import MultiChannelFilterEditor
from pycroglia.ui.widgets.imagefilters.results import FilterResults
from pycroglia.ui.widgets.segmentation.segmentation_editor import SegmentationEditor


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

    def get_results(self) -> List[FilterResults]:
        list_of_results = []

        for i in range(self.tabs.count()):
            editor = self.tabs.widget(i)
            if isinstance(editor, MultiChannelFilterEditor):
                list_of_results.append(editor.get_filter_results())

        return list_of_results


class SegmentationEditorStack(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget]):
        super().__init__(parent=parent)

        self._previous_tab_index = -1

        # Widgets
        self.tabs = QtWidgets.QTabWidget(self)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def add_tabs(self, results: list[FilterResults]):
        self.tabs.clear()

        for elem in results:
            editor = SegmentationEditor(
                img=elem.small_object_filtered_img,
                labeling_strategy=SkimageImgLabeling(SkimageCellConnectivity.CORNERS),
                min_size=elem.min_size,
                with_progress_bar=True,
                parent=self,
            )
            self.tabs.addTab(editor, f"{Path(elem.file_path).name}")


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

        # Page 1 Widgets
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

        # Page 2 Widgets
        page_2 = QtWidgets.QWidget(self)
        self.page_2_filter_tabs = FilterEditorStack(self)
        self.page_2_back_btn = QtWidgets.QPushButton("Back")
        self.page_2_next_btn = QtWidgets.QPushButton("Next")

        # Page 3 Widgets
        page_3 = QtWidgets.QWidget(self)
        self.page_3_segmentation_tabs = SegmentationEditorStack(self)
        self.page_3_back_btn = QtWidgets.QPushButton("Back")
        self.page_3_next_btn = QtWidgets.QPushButton("Next")

        # Connections
        self.page_1_next_btn.clicked.connect(self._on_page_1_next_press)
        self.page_2_back_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.page_2_next_btn.clicked.connect(self._on_page_2_next_prest)
        self.page_3_back_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(1))

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

        page_3_layout = QtWidgets.QVBoxLayout()
        page_3_layout.addWidget(self.page_3_segmentation_tabs)
        page_3_btn_layout = QtWidgets.QHBoxLayout()
        page_3_btn_layout.addWidget(self.page_3_back_btn)
        page_3_btn_layout.addWidget(self.page_3_next_btn)
        page_3_layout.addLayout(page_3_btn_layout)
        page_3.setLayout(page_3_layout)

        self.stacked.addWidget(page_1)
        self.stacked.addWidget(page_2)
        self.stacked.addWidget(page_3)

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

    def _on_page_2_next_prest(self):
        results = self.page_2_filter_tabs.get_results()
        self.page_3_segmentation_tabs.add_tabs(results)
        self.stacked.setCurrentIndex(2)
