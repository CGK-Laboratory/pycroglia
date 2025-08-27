from PyQt6 import QtWidgets, QtCore
from typing import Optional


class LabeledSpinBox(QtWidgets.QWidget):
    """SpinBox with label for integer values.

    Attributes:
        label (QtWidgets.QLabel): Label widget.
        spin_box (QtWidgets.QSpinBox): Spin box widget.
        valueChanged (QtCore.pyqtSignal): Signal emitted when the value changes.
    """

    # Signals
    valueChanged = QtCore.pyqtSignal(int)

    def __init__(
        self,
        label_text: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the LabeledSpinBox.

        Args:
            label_text (str): Label text.
            min_value (Optional[int]): Minimum allowed value.
            max_value (Optional[int]): Maximum allowed value.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        # Widgets
        self.label = QtWidgets.QLabel()
        self.label.setText(label_text)

        self.spin_box = QtWidgets.QSpinBox()
        if min_value:
            self.spin_box.setMinimum(min_value)
        if max_value:
            self.spin_box.setMaximum(max_value)
        self.spin_box.valueChanged.connect(self._value_changed)

        # Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.spin_box)
        self.setLayout(layout)

    def _value_changed(self):
        """Emit the signal when the value changes."""
        self.valueChanged.emit(self.spin_box.value())

    def set_max(self, max_value: int):
        """Set the maximum allowed value.

        Args:
            max_value (int): New maximum value.
        """
        current_value = self.spin_box.value()
        self.spin_box.setMaximum(max_value)

        if current_value > max_value:
            self.spin_box.setValue(current_value)

    def get_value(self) -> int:
        """Get the current value of the SpinBox.

        Returns:
            int: Current value.
        """
        return self.spin_box.value()


class LabeledIntSlider(QtWidgets.QWidget):
    """Slider with labels for integer values.

    Attributes:
        min_label (QtWidgets.QLabel): Label for minimum value.
        max_label (QtWidgets.QLabel): Label for maximum value.
        value_label (QtWidgets.QLabel): Label for current value.
        slider (QtWidgets.QSlider): Slider widget.
        valueChanged (QtCore.pyqtSignal): Signal emitted when the value changes.
    """

    valueChanged = QtCore.pyqtSignal(int)

    def __init__(
        self, min_value: int, max_value: int, parent: Optional[QtWidgets.QWidget] = None
    ):
        """Initialize the LabeledIntSlider.

        Args:
            min_value (int): Minimum value.
            max_value (int): Maximum value.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        # Widgets
        self.min_label = QtWidgets.QLabel(str(min_value))
        self.max_label = QtWidgets.QLabel(str(max_value))
        self.value_label = QtWidgets.QLabel(f"Value: {min_value}")
        self.value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_value)
        self.slider.setMaximum(max_value)
        self.slider.setValue(min_value)

        self.slider.valueChanged.connect(self._on_value_changed)

        # Layout
        slider_h_layout = QtWidgets.QHBoxLayout()
        slider_h_layout.addWidget(self.min_label)
        slider_h_layout.addWidget(self.slider)
        slider_h_layout.addWidget(self.max_label)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(slider_h_layout)
        layout.addWidget(self.value_label)
        self.setLayout(layout)

    def _on_value_changed(self, value: int):
        """Update the label and emit the signal when the value changes.

        Args:
            value (int): New value.
        """
        self.value_label.setText(f"Value: {value}")
        self.valueChanged.emit(value)

    def get_value(self):
        """Get the current value of the slider.

        Returns:
            int: Current value.
        """
        return self.slider.value()


class LabeledFloatSlider(QtWidgets.QWidget):
    """Slider with labels for float values.

    Attributes:
        min_label (QtWidgets.QLabel): Label for minimum value.
        max_label (QtWidgets.QLabel): Label for maximum value.
        value_label (QtWidgets.QLabel): Label for current value.
        label_text (str): Text for the main label.
        slider (QtWidgets.QSlider): Slider widget.
        _min (float): Minimum value.
        _max (float): Maximum value.
        _step (float): Step size.
        valueChanged (QtCore.pyqtSignal): Signal emitted when the value changes.
    """
    DEFAULT_LABEL_TEXT = "Value"

    valueChanged = QtCore.pyqtSignal(float)

    def __init__(
        self,
        min_value: float,
        max_value: float,
        step_size: float,
        label_text: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the LabeledFloatSlider.

        Args:
            min_value (float): Minimum value.
            max_value (float): Maximum value.
            step_size (float): Step size.
            label_text (Optional[str], optional): Text for the main label.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        self.label_text = label_text or self.DEFAULT_LABEL_TEXT

        self._min = min_value
        self._max = max_value
        self._step = step_size

        self._int_min = 0
        self._int_max = int(round((self._max - self._min) / self._step))

        # Widgets
        self.min_label = QtWidgets.QLabel(f"{self._min:.2f}")
        self.max_label = QtWidgets.QLabel(f"{self._max:.2f}")
        self.value_label = QtWidgets.QLabel(f"{self.label_text}: {self._min:.2f}")
        self.value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(self._int_min)
        self.slider.setMaximum(self._int_max)
        self.slider.setValue(self._int_min)

        self.slider.valueChanged.connect(self._on_value_changed)

        # Layout
        slider_h_layout = QtWidgets.QHBoxLayout()
        slider_h_layout.addWidget(self.min_label)
        slider_h_layout.addWidget(self.slider)
        slider_h_layout.addWidget(self.max_label)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(slider_h_layout)
        layout.addWidget(self.value_label)
        self.setLayout(layout)

    def _on_value_changed(self, value: int):
        """Update the label and emit the signal when the value changes.

        Args:
            value (int): Integer value from the slider.
        """
        float_value = self._min + value * self._step
        self.value_label.setText(f"{self.label_text}: {float_value:.2f}")
        self.valueChanged.emit(float_value)

    def get_value(self) -> float:
        """Get the current float value of the slider.

        Returns:
            float: Current value.
        """
        int_value = self.slider.value()
        return self._min + int_value * self._step

    def set_value(self, float_value: float):
        """Set the slider value from a float.

        Args:
            float_value (float): Value to set.
        """
        int_value = int(round((float_value - self._min) / self._step))
        self.slider.setValue(int_value)
