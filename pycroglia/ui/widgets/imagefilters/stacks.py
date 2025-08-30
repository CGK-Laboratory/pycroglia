from pathlib import Path
from typing import Optional, List

from PyQt6 import QtWidgets

from pycroglia.ui.widgets.imagefilters.editors import MultiChannelFilterEditor
from pycroglia.ui.widgets.imagefilters.results import FilterResults


class FilterEditorStack(QtWidgets.QWidget):
    """Widget that manages a tabbed interface for multi-channel filter editors.

    Attributes:
        tabs (QtWidgets.QTabWidget): Tab widget containing filter editors.
        img_viewer_label (Optional[str]): Label text for image viewer.
        read_button_text (Optional[str]): Text for read button.
        channels_label (Optional[str]): Label for channels configurator.
        channel_of_interest_label (Optional[str]): Label for channel of interest.
        gray_filter_label (Optional[str]): Label text for gray filter editor.
        gray_filter_slider_label (Optional[str]): Label for gray filter slider.
        small_objects_filter_label (Optional[str]): Label text for small objects filter editor.
        small_objects_threshold_label (Optional[str]): Label for threshold spin box.
    """

    def __init__(
        self,
        img_viewer_label: Optional[str] = None,
        read_button_text: Optional[str] = None,
        channels_label: Optional[str] = None,
        channel_of_interest_label: Optional[str] = None,
        gray_filter_label: Optional[str] = None,
        gray_filter_slider_label: Optional[str] = None,
        small_objects_filter_label: Optional[str] = None,
        small_objects_threshold_label: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the FilterEditorStack widget.

        Args:
            img_viewer_label (Optional[str]): Label text for image viewer.
            read_button_text (Optional[str]): Text for read button.
            channels_label (Optional[str]): Label for channels configurator.
            channel_of_interest_label (Optional[str]): Label for channel of interest.
            gray_filter_label (Optional[str]): Label text for gray filter editor.
            gray_filter_slider_label (Optional[str]): Label for gray filter slider.
            small_objects_filter_label (Optional[str]): Label text for small objects filter editor.
            small_objects_threshold_label (Optional[str]): Label for threshold spin box.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        # Store text parameters
        self.img_viewer_label = img_viewer_label
        self.read_button_text = read_button_text
        self.channels_label = channels_label
        self.channel_of_interest_label = channel_of_interest_label
        self.gray_filter_label = gray_filter_label
        self.gray_filter_slider_label = gray_filter_slider_label
        self.small_objects_filter_label = small_objects_filter_label
        self.small_objects_threshold_label = small_objects_threshold_label

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
            editor = MultiChannelFilterEditor(
                file,
                img_viewer_label=self.img_viewer_label,
                read_button_text=self.read_button_text,
                channels_label=self.channels_label,
                channel_of_interest_label=self.channel_of_interest_label,
                gray_filter_label=self.gray_filter_label,
                gray_filter_slider_label=self.gray_filter_slider_label,
                small_objects_filter_label=self.small_objects_filter_label,
                small_objects_threshold_label=self.small_objects_threshold_label,
                parent=self,
            )
            self.tabs.addTab(editor, f"{Path(file).name}")

    def get_results(self) -> List[FilterResults]:
        list_of_results = []

        for i in range(self.tabs.count()):
            editor = self.tabs.widget(i)
            if hasattr(editor, 'get_filter_results'):
                list_of_results.append(editor.get_filter_results())

        return list_of_results
