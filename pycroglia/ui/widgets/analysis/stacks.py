from pathlib import Path
from typing import Optional, List

from PyQt6 import QtWidgets

from pycroglia.ui.widgets.analysis.cell_selector import CellSelector
from pycroglia.ui.widgets.common.results import ImgWithPathResults
from pycroglia.ui.widgets.segmentation.results import SegmentationResults


class CellSelectorStack(QtWidgets.QWidget):
    """Widget that manages a tabbed interface for cell selection and analysis.

    Creates and manages multiple CellSelector widgets in a tabbed interface,
    allowing users to perform cell selection and filtering on multiple segmented
    images simultaneously. Each tab corresponds to one segmentation result.

    Attributes:
        DEFAULT_HEADERS_TEXT (list[str]): Default column headers for cell lists.
        DEFAULT_REMOVE_BUTTON_TEXT (str): Default text for remove cell button.
        DEFAULT_SIZE_LABEL_TEXT (str): Default text for size filter label.
        DEFAULT_SIZE_BUTTON_TEXT (str): Default text for size filter button.
        DEFAULT_PREVIEW_BUTTON_TEXT (str): Default text for preview button.
        DEFAULT_BORDER_CHECKBOX_TEXT (str): Default text for border cell checkbox.
        tabs (QtWidgets.QTabWidget): Tab widget containing cell selectors.
        headers_text (list[str]): Text for cell list headers.
        remove_button_text (str): Text for remove cell button.
        size_label_text (str): Text for size filter label.
        size_button_text (str): Text for size filter button.
        preview_button_text (str): Text for preview button.
        border_checkbox_text (str): Text for border cell checkbox.
    """

    # UI Text Constants
    DEFAULT_HEADERS_TEXT = ["Cell number", "Cell size"]
    DEFAULT_REMOVE_BUTTON_TEXT = "Remove Cell"
    DEFAULT_SIZE_LABEL_TEXT = "Cell Size (pixels)"
    DEFAULT_SIZE_BUTTON_TEXT = "Remove smaller than"
    DEFAULT_PREVIEW_BUTTON_TEXT = "Preview"
    DEFAULT_BORDER_CHECKBOX_TEXT = "Remove border cells"

    def __init__(
        self,
        headers: Optional[list[str]] = None,
        remove_button_text: Optional[str] = None,
        size_label_text: Optional[str] = None,
        size_button_text: Optional[str] = None,
        preview_button_text: Optional[str] = None,
        border_checkbox_text: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the CellSelectorStack widget.

        Args:
            headers (Optional[list[str]]): Custom column headers for cell lists.
            remove_button_text (Optional[str]): Custom text for remove cell button.
            size_label_text (Optional[str]): Custom text for size filter label.
            size_button_text (Optional[str]): Custom text for size filter button.
            preview_button_text (Optional[str]): Custom text for preview button.
            border_checkbox_text (Optional[str]): Custom text for border cell checkbox.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent=parent)

        # Text properties
        self.headers_text = headers or self.DEFAULT_HEADERS_TEXT
        self.remove_button_text = remove_button_text or self.DEFAULT_REMOVE_BUTTON_TEXT
        self.size_label_text = size_label_text or self.DEFAULT_SIZE_LABEL_TEXT
        self.size_button_text = size_button_text or self.DEFAULT_SIZE_BUTTON_TEXT
        self.preview_button_text = (
            preview_button_text or self.DEFAULT_PREVIEW_BUTTON_TEXT
        )
        self.border_checkbox_text = (
            border_checkbox_text or self.DEFAULT_BORDER_CHECKBOX_TEXT
        )

        # Widgets
        self.tabs = QtWidgets.QTabWidget(self)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def add_tabs(self, results: list[SegmentationResults]):
        """Clear existing tabs and add a new tab for each segmentation result.

        Creates a CellSelector for each SegmentationResults object and adds it
        as a tab. The tab title is derived from the file path.

        Args:
            results (list[SegmentationResults]): List of segmentation results to create
                tabs for. Each result will get its own cell selector tab.
        """
        self.tabs.clear()

        for elem in results:
            selector = CellSelector(
                img=elem.img,
                headers=self.headers_text,
                remove_button_text=self.remove_button_text,
                size_label_text=self.size_label_text,
                size_button_text=self.size_button_text,
                preview_button_text=self.preview_button_text,
                border_checkbox_text=self.border_checkbox_text,
                parent=self,
            )
            self.tabs.addTab(selector, f"{Path(elem.file_path).name}")

    def get_results(self) -> List[ImgWithPathResults]:
        """Collect selected-cell 3D masks from each tab and return them.

        Iterates over all tabs, queries each CellSelector for its currently
        selected 3D mask (if available) and wraps it with the tab's file path
        into an ImgWithPathResults instance.

        Returns:
            List[ImgWithPathResults]: List of image+path result containers for tabs
                that expose a `get_selected_cells_3d` method.
        """
        list_of_results = []

        for i in range(self.tabs.count()):
            selector = self.tabs.widget(i)
            file_path = self.tabs.tabText(i)
            if hasattr(selector, "get_selected_cells_3d") and hasattr(
                selector, "get_cells_masks"
            ):
                list_of_results.append(
                    ImgWithPathResults(
                        file_path=file_path,
                        img=selector.get_selected_cells_3d(),
                        cells_masks=selector.get_cells_masks(),
                    )
                )

        return list_of_results
