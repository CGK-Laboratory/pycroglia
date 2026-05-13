from typing import Optional
from numpy.typing import NDArray

from PyQt6 import QtWidgets, QtCore

from pycroglia.ui.controllers.ch_editor import MultiChImgEditorState
from pycroglia.ui.widgets.common.img_viewer import CustomImageViewer
from pycroglia.ui.widgets.common.labeled_widgets import (
    LabeledIntSpinBox,
    FloatSliderWithSpinBox,
)
from pycroglia.ui.widgets.imagefilters.configurator import MultiChannelConfigurator
from pycroglia.ui.widgets.imagefilters.tasks import (
    ImageReaderTask,
    GrayFilterTask,
    SmallObjectFilterTask,
)


class MultiChannelImageViewer(QtWidgets.QWidget):
    """Widget for viewing multi-channel images.

    Attributes:
        state (MultiChImgEditorState): State object for image editing.
        label_text (str): Text for the main label.
        read_button_text (str): Text for the read button.
        label (QtWidgets.QLabel): Label for the viewer.
        viewer (ImageView): Image viewer widget.
        editor (MultiChannelConfigurator): Channel configuration widget.
        read_button (QtWidgets.QPushButton): Button to trigger image reading.
    """

    DEFAULT_LABEL_TEXT = "Image Viewer"
    DEFAULT_READ_BUTTON_TEXT = "Read"
    DEFAULT_CHANNELS_LABEL = "Channels"
    DEFAULT_CHANNEL_OF_INTEREST_LABEL = "Channel of interest"

    def __init__(
        self,
        state: MultiChImgEditorState,
        label_text: Optional[str] = None,
        read_button_text: Optional[str] = None,
        channels_label: Optional[str] = None,
        channel_of_interest_label: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the multi-channel image viewer.

        Args:
            state (MultiChImgEditorState): Image editor state.
            label_text (Optional[str], optional): Text for the main label.
            read_button_text (Optional[str], optional): Text for the read button.
            channels_label (Optional[str], optional): Label for channels configurator.
            channel_of_interest_label (Optional[str], optional): Label for channel of interest.
            parent (Optional[QtWidgets.QWidget], optional): Parent widget.
        """
        super().__init__(parent=parent)

        # Configurable text
        self.label_text = label_text or self.DEFAULT_LABEL_TEXT
        self.read_button_text = read_button_text or self.DEFAULT_READ_BUTTON_TEXT

        self.state = state
        self.tpool = QtCore.QThreadPool(self)

        # Widgets
        self.label = QtWidgets.QLabel(parent=self)
        self.label.setText(self.label_text)

        self.viewer = CustomImageViewer(parent=self)

        self.editor = MultiChannelConfigurator(
            channels_label=channels_label or self.DEFAULT_CHANNELS_LABEL,
            channel_of_interest_label=channel_of_interest_label
            or self.DEFAULT_CHANNEL_OF_INTEREST_LABEL,
            parent=self,
        )

        self.read_button = QtWidgets.QPushButton(parent=self)
        self.read_button.setText(self.read_button_text)

        # Connections
        self.read_button.clicked.connect(self._on_read_button_press)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.viewer)
        layout.addWidget(self.editor)
        layout.addWidget(self.read_button)
        self.setLayout(layout)

    def _on_read_button_press(self):
        """Handle the event when the read button is pressed."""
        task = ImageReaderTask(
            state=self.state,
            ch=self.editor.get_channels(),
            chi=self.editor.get_channel_of_interest(),
        )
        task.signals.finished.connect(self._on_image_ready)
        self.tpool.start(task)

    def _on_image_ready(self):
        img = self.state.get_img()
        self.viewer.set_image(self.state.get_midslice(img))


class GrayFilterEditor(QtWidgets.QWidget):
    """Widget for applying and viewing the gray filter.

    Attributes:
        state (MultiChImgEditorState): State object for image editing.
        label_text (str): Text for the main label.
        slider_label_text (str): Label for the adjustment slider.
        label (QtWidgets.QLabel): Label for the editor.
        viewer (ImageView): Image viewer widget.
        slider (FloatSliderWithSpinBox): Slider and spinbox for the Otsu value.
        GRAY_FILTER_MAX (float): Maximum slider value.
        GRAY_FILTER_MIN (float): Minimum slider value.
        GRAY_FILTER_STEP (float): Step size for the slider.
    """

    DEFAULT_LABEL_TEXT = "Gray Filter Editor"
    DEFAULT_SLIDER_LABEL_TEXT = "Adjustment"

    GRAY_FILTER_MAX = 4.0
    GRAY_FILTER_MIN = 0.1
    GRAY_FILTER_STEP = 0.1

    def __init__(
        self,
        state: MultiChImgEditorState,
        label_text: Optional[str] = None,
        slider_label_text: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the gray filter editor.

        Args:
            state (MultiChImgEditorState): Image editor state.
            label_text (Optional[str], optional): Text for the main label.
            slider_label_text (Optional[str], optional): Label for the adjustment slider.
            parent (Optional[QtWidgets.QWidget], optional): Parent widget.
        """
        super().__init__(parent=parent)

        # Configurable text
        self.label_text = label_text or self.DEFAULT_LABEL_TEXT
        self.slider_label_text = slider_label_text or self.DEFAULT_SLIDER_LABEL_TEXT

        self.state = state
        self.tpool = QtCore.QThreadPool(self)

        # Widgets
        self.label = QtWidgets.QLabel(parent=self)
        self.label.setText(self.label_text)

        self.viewer = CustomImageViewer(parent=self)

        self.slider = FloatSliderWithSpinBox(
            min_value=self.GRAY_FILTER_MIN,
            max_value=self.GRAY_FILTER_MAX,
            step_size=self.GRAY_FILTER_STEP,
            label_text=self.slider_label_text,
        )

        # Connections
        self.state.imageChanged.connect(self._on_filter_update)
        self.slider.slider.sliderReleased.connect(self._on_filter_update)
        self.slider.spin_box.valueChanged.connect(self._on_filter_update)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.viewer)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def _on_filter_update(self):
        """Update the view when the gray filter is applied."""
        task = GrayFilterTask(state=self.state, adjust_value=round(self.slider.get_value(), 2))
        task.signals.finished.connect(self._on_image_ready)
        self.tpool.start(task)

    def _on_image_ready(self):
        img = self.state.get_gray_filtered_img()
        if img is not None:
            self.viewer.set_image(self.state.get_midslice(img))


class SmallObjectsFilterEditor(QtWidgets.QWidget):
    """Widget for applying and viewing the small objects filter.

    Attributes:
        state (MultiChImgEditorState): State object for image editing.
        label_text (str): Text for the main label.
        threshold_label_text (str): Label for the threshold spin box.
        label (QtWidgets.QLabel): Label for the editor.
        viewer (ImageView): Image viewer widget.
        spin_box (LabeledIntSpinBox): Spin box for threshold value.
        FILTER_MIN_VALUE (int): Minimum value for the filter threshold.
        FILTER_MAX_VALUE (int): Maximum value for the filter threshold
    """

    DEFAULT_LABEL_TEXT = "Small objects Filter Editor"
    DEFAULT_THRESHOLD_LABEL_TEXT = "Threshold"

    FILTER_MIN_VALUE = 1
    FILTER_MAX_VALUE = 5000

    def __init__(
        self,
        state: MultiChImgEditorState,
        label_text: Optional[str] = None,
        threshold_label_text: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the small objects filter editor.

        Args:
            state (MultiChImgEditorState): Image editor state.
            label_text (Optional[str], optional): Text for the main label.
            threshold_label_text (Optional[str], optional): Label for the threshold spin box.
            parent (Optional[QtWidgets.QWidget], optional): Parent widget.
        """
        super().__init__(parent=parent)

        # Configurable text
        self.label_text = label_text or self.DEFAULT_LABEL_TEXT
        self.threshold_label_text = (
            threshold_label_text or self.DEFAULT_THRESHOLD_LABEL_TEXT
        )

        self.state = state
        self.tpool = QtCore.QThreadPool(self)

        # Widgets
        self.label = QtWidgets.QLabel(parent=self)
        self.label.setText(self.label_text)

        self.viewer = CustomImageViewer(parent=self)

        self.spin_box = LabeledIntSpinBox(
            label_text=self.threshold_label_text,
            min_value=self.FILTER_MIN_VALUE,
            max_value=self.FILTER_MAX_VALUE,
            parent=self,
        )

        # Connections
        self.state.grayImageChanged.connect(self._on_filter_update)
        self.spin_box.valueChanged.connect(self._on_filter_update)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.viewer)
        layout.addWidget(self.spin_box)
        self.setLayout(layout)

    def _on_filter_update(self):
        """Update the view when the small objects filter is applied."""
        task = SmallObjectFilterTask(
            state=self.state, threshold=self.spin_box.get_value()
        )
        task.signals.finished.connect(self._on_image_ready)
        self.tpool.start(task)

    def _on_image_ready(self):
        img = self.state.get_small_objects_img()
        if img is not None:
            self.viewer.set_image(self.state.get_midslice(img))
            self.spin_box.set_max(img.size)


class FilterResults:
    """Encapsulates the results and parameters of the filter pipeline."""

    def __init__(
        self,
        file_path: str,
        gray_filter_value: float,
        min_size: int,
        small_object_filtered_img: NDArray,
        erosion_radius: int = 3,
    ):
        """
        Args:
            file_path (str): Path to the image file.
            gray_filter_value (float): Value used for the gray filter.
            min_size (int): Minimum size for small object removal.
            small_object_filtered_img (np.ndarray): Image after small object removal.
            erosion_radius (int): Radius for the diamond erosion footprint. Default is 3.
        """
        self.file_path = file_path
        self.gray_filter_value = gray_filter_value
        self.min_size = min_size
        self.small_object_filtered_img = small_object_filtered_img
        self.erosion_radius = erosion_radius

    def as_dict(self) -> dict:
        """Returns the filter results as a dictionary.

        Returns:
            dict: Dictionary with filter values and resulting images.
        """
        return {
            "file_path": self.file_path,
            "gray_filter_value": self.gray_filter_value,
            "min_size": self.min_size,
            "small_object_filtered_img": self.small_object_filtered_img,
            "erosion_radius": self.erosion_radius,
        }


class CollapsibleSection(QtWidgets.QWidget):
    """A widget with a clickable header that shows/hides its content area.

    Use :meth:`content_layout` to add child widgets. Future advanced settings
    can be appended to the same section without touching the parent widget.

    Attributes:
        toggle_button (QToolButton): Clickable header that expands/collapses the section.
        content_area (QWidget): Container widget whose visibility is toggled.
        content_layout (QVBoxLayout): Layout inside the content area for adding children.
    """

    DEFAULT_COLLAPSED_ICON = "▶"
    DEFAULT_EXPANDED_ICON = "▼"

    def __init__(
        self,
        title: str = "Advanced Configuration",
        expanded: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ):
        """Initialize the collapsible section.

        Args:
            title (str): Text shown in the toggle header button.
            expanded (bool): Whether the section starts expanded. Default is False.
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent=parent)

        self._expanded = expanded

        # Toggle button acting as header
        self.toggle_button = QtWidgets.QToolButton(parent=self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow
        )
        self.toggle_button.setText(title)
        self.toggle_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        # Content area
        self.content_area = QtWidgets.QWidget(parent=self)
        self.content_area.setVisible(expanded)
        self.content_layout = QtWidgets.QVBoxLayout()
        self.content_layout.setContentsMargins(12, 4, 4, 4)
        self.content_area.setLayout(self.content_layout)

        # Outer layout
        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.toggle_button)
        outer.addWidget(self.content_area)
        self.setLayout(outer)

        self.toggle_button.clicked.connect(self._on_toggle)

    def _on_toggle(self, checked: bool):
        """Show or hide the content area and update the arrow direction."""
        self._expanded = checked
        self.content_area.setVisible(checked)
        self.toggle_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
        )


class MultiChannelFilterEditor(QtWidgets.QWidget):
    """Main widget for multi-channel image editing and viewing.

    Attributes:
        file_path (str): Path to the image file.
        editor_state (MultiChImgEditorState): State object for image editing.
        img_viewer (MultiChannelImageViewer): Widget for image viewing.
        gray_filter_editor (GrayFilterEditor): Widget for gray filter editing.
        small_object_filter_editor (SmallObjectsFilterEditor): Widget for small object filtering.
        erosion_spin_box (LabeledIntSpinBox): Spin box for erosion radius.
    """

    DEFAULT_EROSION_LABEL_TEXT = "Erosion"
    DEFAULT_EROSION_RADIUS = 3
    EROSION_MIN_VALUE = 1
    EROSION_MAX_VALUE = 20

    def __init__(
        self,
        file_path: str,
        img_viewer_label: Optional[str] = None,
        read_button_text: Optional[str] = None,
        channels_label: Optional[str] = None,
        channel_of_interest_label: Optional[str] = None,
        gray_filter_label: Optional[str] = None,
        gray_filter_slider_label: Optional[str] = None,
        small_objects_filter_label: Optional[str] = None,
        small_objects_threshold_label: Optional[str] = None,
        erosion_label: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the multi-channel filter editor.

        Args:
            file_path (str): Path to the image file.
            img_viewer_label (Optional[str]): Label text for image viewer.
            read_button_text (Optional[str]): Text for read button.
            channels_label (Optional[str]): Label for channels configurator.
            channel_of_interest_label (Optional[str]): Label for channel of interest.
            gray_filter_label (Optional[str]): Label text for gray filter editor.
            gray_filter_slider_label (Optional[str]): Label for gray filter slider.
            small_objects_filter_label (Optional[str]): Label text for small objects filter editor.
            small_objects_threshold_label (Optional[str]): Label for threshold spin box.
            erosion_label (Optional[str]): Label for the erosion radius spin box.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent=parent)

        self.file_path = file_path
        self.editor_state = MultiChImgEditorState(file_path=file_path)

        # Widgets
        self.img_viewer = MultiChannelImageViewer(
            state=self.editor_state,
            label_text=img_viewer_label,
            read_button_text=read_button_text,
            channels_label=channels_label,
            channel_of_interest_label=channel_of_interest_label,
            parent=self,
        )
        self.gray_filter_editor = GrayFilterEditor(
            state=self.editor_state,
            label_text=gray_filter_label,
            slider_label_text=gray_filter_slider_label,
            parent=self,
        )
        self.small_object_filter_editor = SmallObjectsFilterEditor(
            state=self.editor_state,
            label_text=small_objects_filter_label,
            threshold_label_text=small_objects_threshold_label,
            parent=self,
        )
        self.erosion_spin_box = LabeledIntSpinBox(
            label_text=erosion_label or self.DEFAULT_EROSION_LABEL_TEXT,
            min_value=self.EROSION_MIN_VALUE,
            max_value=self.EROSION_MAX_VALUE,
            parent=self,
        )
        self.erosion_spin_box.spin_box.setValue(self.DEFAULT_EROSION_RADIUS)

        # Erosion info label inside the advanced section
        self.erosion_info_label = QtWidgets.QLabel(
            "\u2139\ufe0f  This setting is used in the next window to segment cells.",
            parent=self,
        )
        self.erosion_info_label.setWordWrap(True)
        _font = self.erosion_info_label.font()
        _font.setItalic(True)
        self.erosion_info_label.setFont(_font)

        erosion_row = QtWidgets.QHBoxLayout()
        erosion_row.addWidget(self.erosion_spin_box)
        erosion_row.addWidget(self.erosion_info_label, stretch=1)
        erosion_widget = QtWidgets.QWidget(parent=self)
        erosion_widget.setLayout(erosion_row)

        # Advanced configuration collapsible section
        self.advanced_section = CollapsibleSection(
            title="Advanced Configuration", expanded=False, parent=self
        )
        self.advanced_section.content_layout.addWidget(erosion_widget)

        # Layout
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(self.img_viewer, stretch=1)
        filter_row.addWidget(self.gray_filter_editor, stretch=1)
        filter_row.addWidget(self.small_object_filter_editor, stretch=1)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(filter_row)
        layout.addWidget(self.advanced_section)
        self.setLayout(layout)

    def get_filter_results(self) -> FilterResults:
        """Get the current values and images for each filter step.

        Returns:
            FilterResults: Object containing filter values and resulting images.
        """
        gray_value = self.gray_filter_editor.slider.get_value()
        small_obj_value = self.small_object_filter_editor.spin_box.get_value()
        small_obj_img = self.editor_state.get_small_objects_img()
        erosion_radius = self.erosion_spin_box.get_value()
        return FilterResults(
            file_path=self.file_path,
            gray_filter_value=gray_value,
            min_size=small_obj_value,
            small_object_filtered_img=small_obj_img,
            erosion_radius=erosion_radius,
        )
