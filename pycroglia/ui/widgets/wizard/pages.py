from abc import ABC, abstractmethod
from typing import Optional, Any, List

from PyQt6 import QtCore, QtWidgets

from pycroglia.ui.widgets.imagefilters.results import FilterResults
from pycroglia.ui.widgets.imagefilters.stacks import FilterEditorStack
from pycroglia.ui.widgets.io.file_selection_editor import FileSelectionEditor
from pycroglia.ui.widgets.segmentation.stacks import SegmentationEditorStack


class BasePage(ABC):
    """Abstract base class for wizard pages.

    Provides a common interface for wizard pages that can maintain state
    and exchange data between workflow steps.

    Attributes:
        main_widget (QtWidgets.QWidget): The main widget contained in this page.
        page_widget (QtWidgets.QWidget): The wrapper widget for the page.
    """

    def __init__(self, main_widget: QtWidgets.QWidget):
        """Initialize the base page.

        Args:
            main_widget (QtWidgets.QWidget): The main widget to be contained in this page.
        """
        super().__init__()

        self.main_widget = main_widget
        self.page_widget = QtWidgets.QWidget()

    @abstractmethod
    def get_state(self) -> Optional[dict[str, Any]]:
        """Get the current state of the page.

        Returns:
            Optional[dict[str, Any]]: Dictionary containing the page's current state,
                or None if no state is available.
        """
        pass

    @abstractmethod
    def set_data(self, data: Optional[dict[str, Any]]):
        """Set data to initialize or update the page.

        Args:
            data (Optional[dict[str, Any]]): Data dictionary to set in the page,
                or None if no data is provided.
        """
        pass


class FileSelectionPage(BasePage):
    """Page for file selection functionality.

    Handles file selection and provides the selected files to subsequent pages.

    Attributes:
        main_widget (FileSelectionEditor): The file selection editor widget.
    """

    def __init__(self, main_widget: FileSelectionEditor):
        """Initialize the file selection page.

        Args:
            main_widget (FileSelectionEditor): The file selection editor widget.
        """
        super().__init__(main_widget)
        self.main_widget = main_widget

    def get_state(self) -> Optional[dict[str, Any]]:
        """Get the current state containing selected files.

        Returns:
            Optional[dict[str, Any]]: Dictionary with 'files' key containing
                list of selected file paths.
        """
        return {"files": self.main_widget.get_files()}

    def set_data(self, data: Optional[dict[str, Any]]):
        """Set data for file selection page.

        Note:
            This page doesn't accept external data as it's the entry point.

        Args:
            data (Optional[dict[str, Any]]): Ignored for this page.
        """
        pass


class FilterEditorPage(BasePage):
    """Page for image filtering functionality.

    Handles multi-channel image filtering and provides filter results to subsequent pages.

    Attributes:
        main_widget (FilterEditorStack): The filter editor stack widget.
    """

    def __init__(
        self,
        main_widget: FilterEditorStack,
    ):
        """Initialize the FilterEditorPage.

        Args:
            main_widget (FilterEditorStack): The main filter editor stack widget.
        """
        super().__init__(main_widget)
        self.main_widget = main_widget

    def get_state(self) -> Optional[dict[str, Any]]:
        """Get the current state containing filter results.

        Returns:
            Optional[dict[str, Any]]: Dictionary with 'results' key containing
                list of filter result dictionaries.
        """
        list_of_results = self.main_widget.get_results()
        list_of_dicts = [result.as_dict() for result in list_of_results]
        return {"results": list_of_dicts}

    def set_data(self, data: Optional[dict[str, list[str]]]):
        """Set file data to initialize filter editors.

        Args:
            data (Optional[dict[str, list[str]]]): Dictionary containing 'files' key
                with list of file paths to be processed.
        """
        self.main_widget.add_tabs(data["files"])


class SegmentationEditorPage(BasePage):
    """Page for cell segmentation functionality.

    Handles cell segmentation using filter results from previous page.

    Attributes:
        main_widget (SegmentationEditorStack): The segmentation editor stack widget.
    """

    def __init__(
        self,
        main_widget: SegmentationEditorStack,
    ):
        """Initialize the SegmentationEditorPage.

        Args:
            main_widget (SegmentationEditorStack): The main segmentation editor stack widget.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(main_widget)
        self.main_widget = main_widget

    def get_state(self) -> Optional[dict[str, Any]]:
        """Get the current state of segmentation results.

        Note:
            Currently returns None as this is typically the final page.

        Returns:
            Optional[dict[str, Any]]: None, as no further processing is expected.
        """
        pass

    def set_data(self, data: Optional[dict[str, list[FilterResults]]]):
        """Set filter results data to initialize segmentation editors.

        Args:
            data (Optional[dict[str, list[FilterResults]]]): Dictionary containing
                'results' key with list of FilterResults dictionaries.
        """
        list_of_results = [FilterResults(**elem) for elem in data["results"]]
        self.main_widget.add_tabs(list_of_results)


class PageManager(QtCore.QObject):
    """Manages wizard page navigation and data flow.

    Handles the creation of pages with navigation buttons and manages
    the flow of data between pages in the wizard workflow.

    Attributes:
        DEFAULT_BACK_BTN_TXT (str): Default text for back buttons.
        DEFAULT_NEXT_BTN_TXT (str): Default text for next buttons.
        stacked (QtWidgets.QStackedWidget): The stacked widget containing pages.
        pages (List[BasePage]): List of managed pages.
        current_index (int): Index of the currently displayed page.
    """

    DEFAULT_BACK_BTN_TXT = "Back"
    DEFAULT_NEXT_BTN_TXT = "Next"

    def __init__(
        self,
        stacked_widget: QtWidgets.QStackedWidget,
        parent: Optional[QtWidgets.QWidget],
    ):
        """Initialize the page manager.

        Args:
            stacked_widget (QtWidgets.QStackedWidget): The stacked widget to manage.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
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
        """Add a page to the wizard with navigation controls.

        Args:
            page (BasePage): The page to add to the wizard.
            show_back_btn (bool): Whether to show the back button.
            show_next_btn (bool): Whether to show the next button.
            back_btn_txt (Optional[str]): Custom text for back button,
                uses DEFAULT_BACK_BTN_TXT if None.
            next_btn_txt (Optional[str]): Custom text for next button,
                uses DEFAULT_NEXT_BTN_TXT if None.
        """
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(page.main_widget)

        if show_back_btn or show_next_btn:
            btn_layout = QtWidgets.QHBoxLayout()
            current_page_index = len(self.pages)

            if show_back_btn:
                back_btn = QtWidgets.QPushButton(
                    back_btn_txt if back_btn_txt else self.DEFAULT_BACK_BTN_TXT
                )
                back_btn.clicked.connect(lambda: self._handle_back(current_page_index))
                btn_layout.addWidget(back_btn)

            if show_next_btn:
                next_btn = QtWidgets.QPushButton(
                    next_btn_txt if next_btn_txt else self.DEFAULT_NEXT_BTN_TXT
                )
                next_btn.clicked.connect(lambda: self._handle_next(current_page_index))
                btn_layout.addWidget(next_btn)

            layout.addLayout(btn_layout)

        page.page_widget.setLayout(layout)
        self.pages.append(page)
        self.stacked.addWidget(page.page_widget)

    def _handle_back(self, page_index: int):
        """Handle back button navigation.

        Args:
            page_index (int): Current page index.
        """
        if page_index > 0:
            self.current_index = page_index - 1
            self.stacked.setCurrentIndex(self.current_index)

    def _handle_next(self, page_index: int):
        """Handle next button navigation and data transfer.

        Gets state from current page and passes it to the next page.

        Args:
            page_index (int): Current page index.
        """
        actual_page = self.pages[page_index]
        state_of_page = actual_page.get_state()

        if page_index + 1 < len(self.pages):
            next_page = self.pages[page_index + 1]
            next_page.set_data(state_of_page)
            self.current_index = page_index + 1
            self.stacked.setCurrentIndex(self.current_index)
