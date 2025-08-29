from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, List, Any
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


class BasePage(QtCore.QObject, ABC):
    def __init__(
        self, main_widget: QtWidgets.QWidget, parent: Optional[QtWidgets.QWidget]
    ):
        super().__init__(parent=parent)

        self.main_widget = main_widget
        self.page_widget = QtWidgets.QWidget()

    @abstractmethod
    def get_state(self) -> Optional[dict[str, Any]]:
        pass

    @abstractmethod
    def set_data(self, data: Optional[dict[str, Any]]):
        pass


class FileSelectionPage(BasePage):
    def __init__(
        self,
        main_widget: FileSelectionEditor,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(main_widget, parent)
        self.main_widget = main_widget

    def get_state(self) -> Optional[dict[str, Any]]:
        return {"files": self.main_widget.get_files()}

    def set_data(self, data: Optional[dict[str, Any]]):
        pass


class FilterEditorPage(BasePage):
    def __init__(
        self, main_widget: FilterEditorStack, parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(main_widget, parent)
        self.main_widget = main_widget

    def get_state(self) -> Optional[dict[str, Any]]:
        list_of_results = self.main_widget.get_results()
        list_of_dicts = [result.as_dict() for result in list_of_results]
        return {"results": list_of_dicts}

    def set_data(self, data: Optional[dict[str, list[str]]]):
        self.main_widget.add_tabs(data["files"])


class SegmentationEditorPage(BasePage):
    def __init__(
        self,
        main_widget: SegmentationEditorStack,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(main_widget, parent)
        self.main_widget = main_widget

    def get_state(self) -> Optional[dict[str, Any]]:
        pass

    def set_data(self, data: Optional[dict[str, list[FilterResults]]]):
        list_of_results = [FilterResults(**elem) for elem in data]
        self.main_widget.add_tabs(list_of_results)


class PageManager(QtCore.QObject):
    DEFAULT_BACK_BTN_TXT = "Back"
    DEFAULT_NEXT_BTN_TXT = "Next"

    def __init__(
        self,
        stacked_widget: QtWidgets.QStackedWidget,
        parent: Optional[QtWidgets.QWidget],
    ):
        super().__init__(parent=parent)

        self.stacked = stacked_widget
        self.pages: List[BasePage] = []
        self.current_index = 0

    def add_page(
        self,
        page: BasePage,
        show_back_btn: bool = True,
        show_next_btn: bool = True,
        back_btn_txt: Optional[str] = None,
        next_btn_txt: Optional[str] = None,
    ):
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(page.main_widget)

        if show_back_btn or show_next_btn:
            btn_layout = QtWidgets.QHBoxLayout()

            if show_back_btn:
                back_btn = QtWidgets.QPushButton(
                    back_btn_txt if back_btn_txt else self.DEFAULT_BACK_BTN_TXT
                )
                back_btn.clicked.connect(lambda: self._handle_back(len(self.pages)))
                btn_layout.addWidget(back_btn)

            if show_next_btn:
                next_btn = QtWidgets.QPushButton(
                    next_btn_txt if next_btn_txt else self.DEFAULT_NEXT_BTN_TXT
                )
                next_btn.clicked.connect(lambda: self._handle_next(len(self.pages)))
                btn_layout.addWidget(next_btn)

            layout.addLayout(btn_layout)

        page.page_widget.setLayout(layout)
        self.pages.append(page)
        self.stacked.addWidget(page.page_widget)

    def _handle_back(self, page_index: int):
        if page_index > 0:
            self.current_index = page_index - 1
            self.stacked.setCurrentIndex(self.current_index)

    def _handle_next(self, page_index: int):
        actual_page = self.pages[page_index]
        state_of_page = actual_page.get_state()

        if page_index + 1 < len(self.pages):
            next_page = self.pages[page_index + 1]
            next_page.set_data(state_of_page)
            self.current_index = page_index + 1
            self.stacked.setCurrentIndex(self.current_index)




