import pyqtgraph as pg

from typing import Optional
from numpy.typing import NDArray

from PyQt6 import QtWidgets


class CustomImageViewer(QtWidgets.QWidget):
    """Widget for displaying images using pyqtgraph's ImageView.

    Provides methods to set the displayed image and its lookup table.
    Includes customizable border around the image that scales with zoom.
    """

    BORDER_COLOR = "white"
    BORDER_WIDTH = 2

    def __init__(
        self,
        border_color: Optional[str] = None,
        border_width: Optional[int] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initializes the CustomImageViewer widget.

        Args:
            border_color (Optional[str], optional): Color of the image border.
                Defaults to white.
            border_width (Optional[int], optional): Width of the image border in pixels.
                Defaults to 2.
            parent (Optional[QtWidgets.QWidget], optional): Parent widget.
                Defaults to None.
        """
        super().__init__(parent=parent)

        # Configuration
        self.border_color = border_color or self.BORDER_COLOR
        self.border_width = border_width or self.BORDER_WIDTH

        # Widget
        viewer = pg.ImageView(parent=self)
        for component in (viewer.ui.histogram, viewer.ui.menuBtn, viewer.ui.roiBtn):
            component.hide()

        self.img_viewer = viewer
        self._add_image_border(self.border_color, self.border_width)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.img_viewer)
        self.setLayout(layout)

    def _add_image_border(self, color: str, width: int):
        """Adds a border to the ImageItem that remains visible when zooming.

        The border is drawn around the actual image boundaries and scales with
        zoom operations, providing a clear visual indication of the image limits.

        Args:
            color (str): Color of the border. Can be color name or hex code.
            width (int): Width of the border in pixels.
        """
        image_item = self.img_viewer.getImageItem()

        # Create an Image border
        border_pen = pg.mkPen(color, width=width)
        image_item.setBorder(border_pen)

    def set_image(self, img: NDArray):
        """Sets the image to be displayed.

        Args:
            img (NDArray): Image array to display.
        """
        self.img_viewer.setImage(img)
        self._add_image_border(self.border_color, self.border_width)

    def set_lookup_table(self, lu: NDArray):
        """Sets the lookup table for coloring the image.

        Args:
            lu (NDArray): Lookup table array.
        """
        self.img_viewer.getImageItem().setLookupTable(lu)
