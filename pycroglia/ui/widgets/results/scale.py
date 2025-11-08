from typing import Optional
from PyQt6 import QtWidgets, QtCore

from pycroglia.ui.widgets.common.labeled_widgets import LabeledFloatSpinBox


class ScaleConfigWidget(QtWidgets.QWidget):
    DEFAULT_SCALE_TXT = "Scale (μm)"
    DEFAULT_Z_SCALE_TXT = "Z Scale (μm)"
    DEFAULT_BUTTON_TXT = "Calculate"

    def __init__(
        self,
        scale_txt: Optional[str] = None,
        z_scale_txt: Optional[str] = None,
        button_txt: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent=parent)

        # Text properties
        self._scale_txt = scale_txt or self.DEFAULT_SCALE_TXT
        self._z_scale_txt = z_scale_txt or self.DEFAULT_Z_SCALE_TXT
        self._button_txt = button_txt or self.DEFAULT_BUTTON_TXT

        # Widgets
        self._scale = LabeledFloatSpinBox(self._scale_txt, min_value=1.0, parent=self)
        self._z_scale = LabeledFloatSpinBox(
            self._z_scale_txt, min_value=1.0, parent=self
        )
        self._button = QtWidgets.QPushButton(self._button_txt, parent=self)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        first_row = QtWidgets.QHBoxLayout()
        first_row.addWidget(self._scale)
        first_row.addWidget(self._z_scale)

        layout.addLayout(first_row)
        layout.addWidget(self._button)

        self.setLayout(layout)

    def get_scale(self) -> float:
        return self._scale.get_value()

    def get_z_scale(self) -> float:
        return self._z_scale.get_value()

    def get_vox_scale(self) -> float:
        return self.get_scale() * self.get_scale() * self.get_scale()

    @property
    def clicked(self) -> QtCore.pyqtSignal:
        return self._button.clicked
