from typing import Optional
from PyQt6 import QtWidgets, QtCore

from pycroglia.ui.widgets.common.labeled_widgets import LabeledFloatSpinBox


class ScaleConfigWidget(QtWidgets.QWidget):
    """Widget that exposes scale controls and a calculate button.

    The widget provides two LabeledFloatSpinBox controls (scale and z-scale)
    and a QPushButton to trigger computation. Consumers can read scale values
    and enable/disable the button via the provided helpers.

    Attributes:
        _scale (LabeledFloatSpinBox): Control for primary scale.
        _z_scale (LabeledFloatSpinBox): Control for z-scale.
        _button (QPushButton): Button to trigger calculation.
    """

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
        """Return the currently selected scale value.

        Returns:
            float: The value from the scale control.
        """
        return self._scale.get_value()

    def get_z_scale(self) -> float:
        """Return the currently selected z-scale value.

        Returns:
            float: The value from the z-scale control.
        """
        return self._z_scale.get_value()

    def get_vox_scale(self) -> float:
        """Compute and return the voxel-scale approximation.

        Returns:
            float: Derived voxel scale (scale^3).
        """
        return self.get_scale() * self.get_scale() * self.get_scale()

    def disable_button(self):
        """Disable the calculate button to prevent user interaction."""
        self._button.setEnabled(False)

    def enable_button(self):
        """Enable the calculate button to allow user interaction."""
        self._button.setEnabled(True)

    @property
    def clicked(self) -> QtCore.pyqtSignal:
        """Expose the underlying button's clicked signal.

        Returns:
            QtCore.pyqtSignal: The clicked signal from the internal QPushButton.
        """
        return self._button.clicked
