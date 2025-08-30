import pytest
from pycroglia.ui.widgets.imagefilters.configurator import MultiChannelConfigurator


@pytest.fixture
def multi_channel_config(qtbot):
    """Fixture for creating a MultiChannelConfigurator widget.

    Args:
        qtbot: pytest-qt bot for widget handling.

    Returns:
        MultiChannelConfigurator: The widget instance.
    """
    widget = MultiChannelConfigurator()
    qtbot.addWidget(widget)
    return widget


def test_get_channels(multi_channel_config):
    """Test that get_channels returns the correct value after setting it."""
    new_value = 10

    multi_channel_config.ch_box.spin_box.setValue(new_value)
    assert multi_channel_config.get_channels() == new_value


def test_get_channel_of_interest(multi_channel_config):
    """Test that get_channel_of_interest returns the correct value after setting it."""
    new_value = 10

    multi_channel_config.chi_box.spin_box.setValue(new_value)
    assert multi_channel_config.get_channel_of_interest() == new_value


def test_new_ch_value_updates_chi_max_value(multi_channel_config):
    """Test that updating the channels value updates the max value for channel of interest."""
    new_ch_value = 4

    multi_channel_config.chi_box.spin_box.setValue(10)
    multi_channel_config.ch_box.spin_box.setValue(new_ch_value)

    assert multi_channel_config.chi_box.spin_box.maximum() == new_ch_value
    assert multi_channel_config.get_channel_of_interest() == new_ch_value
