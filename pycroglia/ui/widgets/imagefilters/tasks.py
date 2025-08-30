from PyQt6 import QtCore

from pycroglia.ui.controllers.ch_editor import MultiChImgEditorState


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
