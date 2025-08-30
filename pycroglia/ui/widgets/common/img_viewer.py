from typing import Optional
from numpy.typing import NDArray

from PyQt6 import QtWidgets
from pyqtgraph import ImageView


class CustomImageViewer(QtWidgets.QWidget):
    """Widget for displaying images using pyqtgraph's ImageView.

    Provides methods to set the displayed image and its lookup table.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """Initializes the CustomImageViewer widget.

        Args:
            parent (Optional[QtWidgets.QWidget], optional): Parent widget. Defaults to None.
        """
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
        """Sets the image to be displayed.

        Args:
            img (NDArray): Image array to display.
        """
        self.img_viewer.setImage(img)

    def set_lookup_table(self, lu: NDArray):
        """Sets the lookup table for coloring the image.

        Args:
            lu (NDArray): Lookup table array.
        """
        self.img_viewer.getImageItem().setLookupTable(lu)
