from PyQt6 import QtWidgets
from typing import Optional

from pycroglia.ui.widgets.labeled_widgets import LabeledSpinBox


class MultiChannelConfigurator(QtWidgets.QWidget):
    """Widget for configuring multi-channel image parameters.

    Allows selection of the number of channels and the channel of interest.

    Attributes:
        ch_box (LabeledSpinBox): Spin box for selecting the number of channels.
        chi_box (LabeledSpinBox): Spin box for selecting the channel of interest.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """Initialize the multi-channel configurator widget.

        Args:
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        # Widgets
        self.ch_box = LabeledSpinBox(
            label_text="Channels", min_value=1, max_value=None, parent=self
        )
        self.chi_box = LabeledSpinBox(
            label_text="Channel of interest", min_value=1, max_value=None, parent=self
        )
        self.ch_box.valueChanged.connect(self._update_chi_max_limit)

        # Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.ch_box)
        layout.addWidget(self.chi_box)
        self.setLayout(layout)

    def _update_chi_max_limit(self, channels: int):
        """Update the maximum limit for the channel of interest based on the number of channels.

        Args:
            channels (int): Number of selected channels.
        """
        max_channels = max(1, channels)
        self.chi_box.set_max(max_channels)

    def get_channels(self) -> int:
        """Get the selected number of channels.

        Returns:
            int: Number of channels.
        """
        return self.ch_box.get_value()

    def get_channel_of_interest(self) -> int:
        """Get the selected channel of interest.

        Returns:
            int: Channel of interest.
        """
        return self.chi_box.get_value()
