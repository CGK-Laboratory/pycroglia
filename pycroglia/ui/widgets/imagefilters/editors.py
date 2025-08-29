from typing import Optional
from numpy.typing import NDArray

from PyQt6 import QtWidgets, QtCore
from pyqtgraph import ImageView

from pycroglia.ui.controllers.ch_editor import MultiChImgEditorState
from pycroglia.ui.widgets.common.labeled_widgets import (
    LabeledSpinBox,
    LabeledFloatSlider,
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

        viewer = ImageView(parent=self)
        for component in (viewer.ui.histogram, viewer.ui.menuBtn, viewer.ui.roiBtn):
            component.hide()
        self.viewer = viewer

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
        self.viewer.setImage(self.state.get_midslice(img))


class GrayFilterEditor(QtWidgets.QWidget):
    """Widget for applying and viewing the gray filter.

    Attributes:
        state (MultiChImgEditorState): State object for image editing.
        label_text (str): Text for the main label.
        slider_label_text (str): Label for the adjustment slider.
        label (QtWidgets.QLabel): Label for the editor.
        viewer (ImageView): Image viewer widget.
        slider (LabeledFloatSlider): Slider for adjusting the gray filter.
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

        viewer = ImageView(parent=self)
        for component in (viewer.ui.histogram, viewer.ui.menuBtn, viewer.ui.roiBtn):
            component.hide()
        self.viewer = viewer

        self.slider = LabeledFloatSlider(
            label_text=self.slider_label_text,
            min_value=self.GRAY_FILTER_MIN,
            max_value=self.GRAY_FILTER_MAX,
            step_size=self.GRAY_FILTER_STEP,
        )

        # Connections
        self.state.imageChanged.connect(self._on_filter_update)
        self.slider.valueChanged.connect(self._on_filter_update)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.viewer)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def _on_filter_update(self):
        """Update the view when the gray filter is applied."""
        task = GrayFilterTask(state=self.state, adjust_value=self.slider.get_value())
        task.signals.finished.connect(self._on_image_ready)
        self.tpool.start(task)

    def _on_image_ready(self):
        img = self.state.get_gray_filtered_img()
        if img is not None:
            self.viewer.setImage(self.state.get_midslice(img))


class SmallObjectsFilterEditor(QtWidgets.QWidget):
    """Widget for applying and viewing the small objects filter.

    Attributes:
        state (MultiChImgEditorState): State object for image editing.
        label_text (str): Text for the main label.
        threshold_label_text (str): Label for the threshold spin box.
        label (QtWidgets.QLabel): Label for the editor.
        viewer (ImageView): Image viewer widget.
        spin_box (LabeledSpinBox): Spin box for threshold value.
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

        viewer = ImageView(parent=self)
        for component in (viewer.ui.histogram, viewer.ui.menuBtn, viewer.ui.roiBtn):
            component.hide()
        self.viewer = viewer

        self.spin_box = LabeledSpinBox(
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
            self.viewer.setImage(self.state.get_midslice(img))


class FilterResults:
    """Encapsulates the results and parameters of the filter pipeline."""

    def __init__(
        self,
        file_path: str,
        gray_filter_value: float,
        min_size: int,
        small_object_filtered_img: NDArray,
    ):
        """
        Args:
            file_path (str): Path to the image file.
            gray_filter_value (float): Value used for the gray filter.
            min_size (int): Minimum size for small object removal.
            small_object_filtered_img (np.ndarray): Image after small object removal.
        """
        self.file_path = file_path
        self.gray_filter_value = gray_filter_value
        self.min_size = min_size
        self.small_object_filtered_img = small_object_filtered_img

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
        }


class MultiChannelFilterEditor(QtWidgets.QWidget):
    """Main widget for multi-channel image editing and viewing.

    Attributes:
        file_path (str): Path to the image file.
        editor_state (MultiChImgEditorState): State object for image editing.
        img_viewer (MultiChannelImageViewer): Widget for image viewing.
        gray_filter_editor (GrayFilterEditor): Widget for gray filter editing.
        small_object_filter_editor (SmallObjectsFilterEditor): Widget for small object filtering.
    """

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

        # Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.img_viewer, stretch=1)
        layout.addWidget(self.gray_filter_editor, stretch=1)
        layout.addWidget(self.small_object_filter_editor, stretch=1)
        self.setLayout(layout)

    def get_filter_results(self) -> FilterResults:
        """Get the current values and images for each filter step.

        Returns:
            FilterResults: Object containing filter values and resulting images.
        """
        gray_value = self.gray_filter_editor.slider.get_value()
        small_obj_value = self.small_object_filter_editor.spin_box.get_value()
        small_obj_img = self.editor_state.get_small_objects_img()
        return FilterResults(
            file_path=self.file_path,
            gray_filter_value=gray_value,
            min_size=small_obj_value,
            small_object_filtered_img=small_obj_img,
        )
