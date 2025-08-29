from PyQt6 import QtWidgets
from typing import Optional

from pycroglia.ui.widgets.common.labeled_widgets import LabeledSpinBox


class MultiChannelConfigurator(QtWidgets.QWidget):
    """Widget for configuring multi-channel image parameters.

    Allows selection of the number of channels and the channel of interest.

    Attributes:
        channels_label (str): Label text for the channels spin box.
        channel_of_interest_label (str): Label text for the channel of interest spin box.
        ch_box (LabeledSpinBox): Spin box for selecting the number of channels.
        chi_box (LabeledSpinBox): Spin box for selecting the channel of interest.
    """

    DEFAULT_CHANNELS_LABEL = "Channels"
    DEFAULT_CHANNEL_OF_INTEREST_LABEL = "Channel of interest"
    DEFAULT_MIN_CHANNELS = 1
    DEFAULT_MIN_CHANNEL_OF_INTEREST = 1

    def __init__(
        self,
        channels_label: Optional[str] = None,
        channel_of_interest_label: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the multi-channel configurator widget.

        Args:
            channels_label (Optional[str], optional): Label text for the channels spin box.
            channel_of_interest_label (Optional[str], optional): Label text for the channel of interest spin box.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        # Configurable text properties
        self.channels_label = channels_label or self.DEFAULT_CHANNELS_LABEL
        self.channel_of_interest_label = (
            channel_of_interest_label or self.DEFAULT_CHANNEL_OF_INTEREST_LABEL
        )

        # Widgets
        self.ch_box = LabeledSpinBox(
            label_text=self.channels_label,
            min_value=self.DEFAULT_MIN_CHANNELS,
            max_value=None,
            parent=self,
        )
        self.chi_box = LabeledSpinBox(
            label_text=self.channel_of_interest_label,
            min_value=self.DEFAULT_MIN_CHANNEL_OF_INTEREST,
            max_value=None,
            parent=self,
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
