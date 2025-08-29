from typing import Optional

from PyQt6 import QtWidgets, QtCore
from pyqtgraph import ImageView

from pycroglia.ui.controllers.ch_editor import MultiChImgEditorState
from pycroglia.ui.widgets.imagefilters.configurator import MultiChannelConfigurator
from pycroglia.ui.widgets.imagefilters.tasks import ImageReaderTask


class MultiChannelImageLoader(QtWidgets.QWidget):
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
