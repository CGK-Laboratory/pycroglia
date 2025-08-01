from typing import Optional

from PyQt6 import QtWidgets, QtCore
from pyqtgraph import ImageView

from pycroglia.ui.controllers.ch_editor import MultiChImgEditorState
from pycroglia.ui.widgets.labeled_widgets import LabeledSpinBox, LabeledFloatSlider
from pycroglia.ui.widgets.ch_config import MultiChannelConfigurator


class TaskSignals(QtCore.QObject):
    """Signals for QRunnable tasks.

    Attributes:
        finished (QtCore.pyqtSignal): Signal emitted when the task is finished.
    """

    finished = QtCore.pyqtSignal()


class ImageReaderTask(QtCore.QRunnable):
    """QRunnable task for reading an image asynchronously.

    Attributes:
        state (MultiChImgEditorState): State object for image editing.
        ch (int): Number of channels.
        chi (int): Channel of interest.
        signals (TaskSignals): Signals for task completion.
    """

    def __init__(self, state: MultiChImgEditorState, ch: int, chi: int):
        """Initialize the image reader task.

        Args:
            state (MultiChImgEditorState): State object for image editing.
            ch (int): Number of channels.
            chi (int): Channel of interest.
        """
        super().__init__()

        self.state = state
        self.ch = ch
        self.chi = chi
        self.signals = TaskSignals()

    def run(self):
        """Run the image reading task and emit finished signal."""
        self.state.read_img(self.ch, self.chi)
        self.signals.finished.emit()


class GrayFilterTask(QtCore.QRunnable):
    """QRunnable task for applying the gray filter asynchronously.

    Attributes:
        state (MultiChImgEditorState): State object for image editing.
        adjust_value (float): Adjustment value for the gray filter.
        signals (TaskSignals): Signals for task completion.
    """

    def __init__(self, state: MultiChImgEditorState, adjust_value: float):
        """Initialize the gray filter task.

        Args:
            state (MultiChImgEditorState): State object for image editing.
            adjust_value (float): Adjustment value for the gray filter.
        """
        super().__init__()

        self.state = state
        self.adjust_value = adjust_value
        self.signals = TaskSignals()

    def run(self):
        """Run the gray filter task and emit finished signal."""
        self.state.apply_otsu_gray_filter(self.adjust_value)
        self.signals.finished.emit()


class SmallObjectFilterTask(QtCore.QRunnable):
    """QRunnable task for applying the small objects filter asynchronously.

    Attributes:
        state (MultiChImgEditorState): State object for image editing.
        threshold (int): Minimum object size threshold.
        signals (TaskSignals): Signals for task completion.
    """

    def __init__(self, state: MultiChImgEditorState, threshold: int):
        """Initialize the small objects filter task.

        Args:
            state (MultiChImgEditorState): State object for image editing.
            threshold (int): Minimum object size threshold.
        """
        super().__init__()

        self.state = state
        self.threshold = threshold
        self.signals = TaskSignals()

    def run(self):
        """Run the small objects filter task and emit finished signal."""
        self.state.apply_small_object_filter(self.threshold)
        self.signals.finished.emit()


class MultiChannelImageViewer(QtWidgets.QWidget):
    """Widget for viewing multi-channel images.

    Attributes:
        state (MultiChImgEditorState): State object for image editing.
        label (QtWidgets.QLabel): Label for the viewer.
        viewer (ImageView): Image viewer widget.
        editor (MultiChannelConfigurator): Channel configuration widget.
        read_button (QtWidgets.QPushButton): Button to trigger image reading.
    """

    def __init__(
        self, state: MultiChImgEditorState, parent: Optional[QtWidgets.QWidget] = None
    ):
        """Initialize the multi-channel image viewer.

        Args:
            state (MultiChImgEditorState): Image editor state.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent=parent)

        self.state = state
        self.tpool = QtCore.QThreadPool(self)

        # Widgets
        self.label = QtWidgets.QLabel(parent=self)
        self.label.setText("Image Viewer")

        viewer = ImageView(parent=self)
        for component in (viewer.ui.histogram, viewer.ui.menuBtn, viewer.ui.roiBtn):
            component.hide()
        self.viewer = viewer

        self.editor = MultiChannelConfigurator(parent=self)

        self.read_button = QtWidgets.QPushButton(parent=self)
        self.read_button.setText("Read")

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
        label (QtWidgets.QLabel): Label for the editor.
        viewer (ImageView): Image viewer widget.
        slider (LabeledFloatSlider): Slider for adjusting the gray filter.
        GRAY_FILTER_MAX (float): Maximum slider value.
        GRAY_FILTER_MIN (float): Minimum slider value.
        GRAY_FILTER_STEP (float): Step size for the slider.
    """

    GRAY_FILTER_MAX = 4.0
    GRAY_FILTER_MIN = 0.1
    GRAY_FILTER_STEP = 0.1

    def __init__(
        self, state: MultiChImgEditorState, parent: Optional[QtWidgets.QWidget] = None
    ):
        """Initialize the gray filter editor.

        Args:
            state (MultiChImgEditorState): Image editor state.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent=parent)

        self.state = state
        self.tpool = QtCore.QThreadPool(self)

        # Widgets
        self.label = QtWidgets.QLabel(parent=self)
        self.label.setText("Gray Filter Editor")

        viewer = ImageView(parent=self)
        for component in (viewer.ui.histogram, viewer.ui.menuBtn, viewer.ui.roiBtn):
            component.hide()
        self.viewer = viewer

        self.slider = LabeledFloatSlider(
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
        label (QtWidgets.QLabel): Label for the editor.
        viewer (ImageView): Image viewer widget.
        spin_box (LabeledSpinBox): Spin box for threshold value.
        FILTER_MIN_VALUE (int): Minimum value for the filter threshold.
        FILTER_MAX_VALUE (int): Maximum value for the filter threshold
    """

    FILTER_MIN_VALUE = 1
    FILTER_MAX_VALUE = 1000

    def __init__(
        self, state: MultiChImgEditorState, parent: Optional[QtWidgets.QWidget] = None
    ):
        """Initialize the small objects filter editor.

        Args:
            state (MultiChImgEditorState): Image editor state.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent=parent)

        self.state = state
        self.tpool = QtCore.QThreadPool(self)

        # Widgets
        self.label = QtWidgets.QLabel(parent=self)
        self.label.setText("Small objects Filter Editor")

        viewer = ImageView(parent=self)
        for component in (viewer.ui.histogram, viewer.ui.menuBtn, viewer.ui.roiBtn):
            component.hide()
        self.viewer = viewer

        self.spin_box = LabeledSpinBox(
            label_text="Threshold: ",
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


class MultiChannelFilterEditor(QtWidgets.QWidget):
    """Main widget for multi-channel image editing and viewing.

    Attributes:
        file_path (str): Path to the image file.
        editor_state (MultiChImgEditorState): State object for image editing.
        img_viewer (MultiChannelImageViewer): Widget for image viewing.
        gray_filter_editor (GrayFilterEditor): Widget for gray filter editing.
        small_object_filter_editor (SmallObjectsFilterEditor): Widget for small object filtering.
    """

    def __init__(self, file_path: str, parent: Optional[QtWidgets.QWidget] = None):
        """Initialize the multi-channel filter editor.

        Args:
            file_path (str): Path to the image file.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent=parent)

        self.file_path = file_path
        self.editor_state = MultiChImgEditorState(file_path=file_path)

        # Widgets
        self.img_viewer = MultiChannelImageViewer(state=self.editor_state, parent=self)
        self.gray_filter_editor = GrayFilterEditor(state=self.editor_state, parent=self)
        self.small_object_filter_editor = SmallObjectsFilterEditor(
            state=self.editor_state, parent=self
        )

        # Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.img_viewer, stretch=1)
        layout.addWidget(self.gray_filter_editor, stretch=1)
        layout.addWidget(self.small_object_filter_editor, stretch=1)
        self.setLayout(layout)
