from typing import Optional
from numpy.typing import NDArray

from PyQt6 import QtWidgets
from pyqtgraph import ImageView


class CustomImageViewer(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent)

        # Widget
        viewer = ImageView(parent=self)
        for component in (viewer.ui.histogram, viewer.ui.menuBtn, viewer.ui.roiBtn):
            component.hide()

        self.img_viewer = viewer

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.img_viewer)
        self.setLayout(layout)

    def set_image(self, img: NDArray):
        self.img_viewer.setImage(img)

    def set_lookup_table(self, lu: NDArray):
        self.img_viewer.getImageItem().setLookupTable(lu)
