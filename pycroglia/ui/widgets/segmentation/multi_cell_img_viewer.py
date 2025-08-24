from typing import Optional

import numpy as np
from PyQt6 import QtWidgets

from pycroglia.core.labeled_cells import LabeledCells
from pycroglia.ui.widgets.img_viewer import CustomImageViewer


class MultiCellImageViewer(QtWidgets.QWidget):
    DEFAULT_RGB_SEED: int = 42

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent)

        # Widgets
        self.img_viewer = CustomImageViewer(parent=self)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.img_viewer)
        self.setLayout(layout)

    def set_cells_img(self, cells: LabeledCells, rgb_seed: int = DEFAULT_RGB_SEED):
        label_2d = cells.labels_to_2d()
        n_cells = cells.len()

        rng = np.random.default_rng(rgb_seed)
        lut = np.zeros((n_cells + 1, 4), dtype=np.uint8)
        lut[0] = (0, 0, 0, 255)
        lut[1:] = np.concatenate(
            [
                rng.integers(0, 256, size=(n_cells, 3), dtype=np.uint8),
                np.full((n_cells, 1), 255, dtype=np.uint8),
            ],
            axis=1,
        )

        self.img_viewer.set_image(label_2d)
        self.img_viewer.set_lookup_table(lut)
