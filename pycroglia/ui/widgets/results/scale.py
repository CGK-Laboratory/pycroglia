from typing import Optional
from PyQt6 import QtWidgets, QtCore

from pycroglia.ui.widgets.common.labeled_widgets import LabeledFloatSpinBox


class ScaleConfigWidget(QtWidgets.QWidget):
    skeletonizationChanged = QtCore.pyqtSignal()

    """Widget that exposes scale controls, skeletonization method, and a calculate button.

    The widget provides two LabeledFloatSpinBox controls (scale and z-scale),
    a skeletonization method radio group, and a QPushButton to trigger
    computation. Consumers can read scale values and skeletonization method
    and enable/disable the button via the provided helpers.
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

        self._is_calculating = False

        # Widgets
        self._scale = LabeledFloatSpinBox(
            self._scale_txt, min_value=0.0, decimals=3, parent=self
        )
        self._z_scale = LabeledFloatSpinBox(
            self._z_scale_txt, min_value=0.0, decimals=3, parent=self
        )
        self._button = QtWidgets.QPushButton(self._button_txt, parent=self)

        self._scale.valueChanged.connect(self._update_button_state)
        self._z_scale.valueChanged.connect(self._update_button_state)
        self._update_button_state()

        self._skeletonization_group = QtWidgets.QGroupBox("Skeletonization Method", parent=self)
        skel_layout = QtWidgets.QHBoxLayout()
        self._skel_radio_slim = QtWidgets.QRadioButton("slimskel3d", parent=self._skeletonization_group)
        self._skel_radio_slim.setChecked(True)
        self._skel_radio_slim.toggled.connect(self.skeletonizationChanged)
        self._skel_radio_scikit = QtWidgets.QRadioButton("slimskel3d_scikit", parent=self._skeletonization_group)
        self._skel_radio_scikit.toggled.connect(self.skeletonizationChanged)
        skel_layout.addWidget(self._skel_radio_slim)
        skel_layout.addWidget(self._skel_radio_scikit)
        self._skeletonization_group.setLayout(skel_layout)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        first_row = QtWidgets.QHBoxLayout()
        first_row.addWidget(self._scale)
        first_row.addWidget(self._z_scale)

        layout.addLayout(first_row)
        layout.addWidget(self._skeletonization_group)
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

    def get_skeletonization_method(self) -> str:
        if self._skel_radio_slim.isChecked():
            return "slimskel3d"
        return "slimskel3d_scikit"

    def get_vox_scale(self) -> float:
        """Compute and return the voxel-scale approximation.

        Returns:
            float: Derived voxel scale (scale^3).
        """
        return self.get_scale() * self.get_scale() * self.get_z_scale()

    def _update_button_state(self, _=None):
        """Update button state based on scale values and computing status."""
        has_valid_scales = self.get_scale() > 0.0 and self.get_z_scale() > 0.0
        self._button.setEnabled(has_valid_scales and not self._is_calculating)

    def disable_button(self):
        """Disable the calculate button to prevent user interaction."""
        self._is_calculating = True
        self._update_button_state()

    def enable_button(self):
        """Enable the calculate button to allow user interaction."""
        self._is_calculating = False
        self._update_button_state()

    @property
    def clicked(self) -> QtCore.pyqtSignal:
        """Expose the underlying button's clicked signal.

        Returns:
            QtCore.pyqtSignal: The clicked signal from the internal QPushButton.
        """
        return self._button.clicked
