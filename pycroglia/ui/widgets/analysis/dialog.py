from typing import Optional
from numpy.typing import NDArray

from PyQt6 import QtWidgets

from pycroglia.ui.widgets.common.img_viewer import CustomImageViewer


class PreviewDialog(QtWidgets.QDialog):
    DEFAULT_WINDOW_TITLE = "Image preview"
    DEFAULT_DIALOG_SIZE = (800, 600)

    def __init__(
        self,
        img: NDArray,
        title_txt: Optional[str] = None,
        dialog_size: Optional[tuple[int, int]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)

        # Text configuration
        self.window_title = title_txt or self.DEFAULT_WINDOW_TITLE
        self.dialog_size = dialog_size or self.DEFAULT_DIALOG_SIZE

        self.setWindowTitle(self.window_title)
        self.setModal(True)

        # Widgets
        self.img_viewer = CustomImageViewer(parent=self)
        self.img_viewer.set_image(img)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.img_viewer)
        self.setLayout(layout)

        self.resize(self.dialog_size[0], self.dialog_size[1])
