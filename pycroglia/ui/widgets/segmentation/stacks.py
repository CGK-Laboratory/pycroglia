from pathlib import Path
from typing import Optional

from PyQt6 import QtWidgets

from pycroglia.core.enums import SkimageCellConnectivity
from pycroglia.core.labeled_cells import SkimageImgLabeling
from pycroglia.ui.widgets.imagefilters.results import FilterResults
from pycroglia.ui.widgets.segmentation.editor import SegmentationEditor


class SegmentationEditorStack(QtWidgets.QWidget):
    """Widget that manages a tabbed interface for segmentation editors.

    Attributes:
        tabs (QtWidgets.QTabWidget): Tab widget containing segmentation editors.
        headers_text (Optional[list[str]]): Text for cell list headers.
        rollback_button_text (Optional[str]): Text for rollback button.
        segmentation_button_text (Optional[str]): Text for segmentation button.
        progress_title (Optional[str]): Title for progress dialog.
        progress_cancel_text (Optional[str]): Text for progress cancel button.
    """

    def __init__(
        self,
        headers_text: Optional[list[str]] = None,
        rollback_button_text: Optional[str] = None,
        segmentation_button_text: Optional[str] = None,
        progress_title: Optional[str] = None,
        progress_cancel_text: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the SegmentationEditorStack widget.

        Args:
            headers_text (Optional[list[str]]): Text for cell list headers.
            rollback_button_text (Optional[str]): Text for rollback button.
            segmentation_button_text (Optional[str]): Text for segmentation button.
            progress_title (Optional[str]): Title for progress dialog.
            progress_cancel_text (Optional[str]): Text for progress cancel button.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent=parent)

        # Store text parameters
        self.headers_text = headers_text
        self.rollback_button_text = rollback_button_text
        self.segmentation_button_text = segmentation_button_text
        self.progress_title = progress_title
        self.progress_cancel_text = progress_cancel_text

        # Widgets
        self.tabs = QtWidgets.QTabWidget(self)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def add_tabs(self, results: list[FilterResults]):
        """Clear and add a tab for each result, each with a SegmentationEditor.

        Args:
            results (list[FilterResults]): List of filter results to add as tabs.
        """
        self.tabs.clear()

        for elem in results:
            editor = SegmentationEditor(
                img=elem.small_object_filtered_img,
                labeling_strategy=SkimageImgLabeling(SkimageCellConnectivity.CORNERS),
                min_size=elem.min_size,
                with_progress_bar=True,
                headers=self.headers_text,
                rollback_button_text=self.rollback_button_text,
                segmentation_button_text=self.segmentation_button_text,
                progress_title=self.progress_title,
                progress_cancel_text=self.progress_cancel_text,
                parent=self,
            )
            self.tabs.addTab(editor, f"{Path(elem.file_path).name}")
