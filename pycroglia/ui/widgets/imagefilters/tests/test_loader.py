import pytest
import numpy as np
from numpy.typing import NDArray
from unittest.mock import MagicMock, patch

from pycroglia.ui.widgets.imagefilters.loader import MultiChannelImageLoader


@pytest.fixture
def fake_image() -> NDArray:
    """Fixture for a fake 3D numpy image.

    Returns:
        NDArray: 3D array of ones.
    """
    return np.ones((5, 5, 5), dtype=np.uint8)


@pytest.fixture
def mock_editor_state(fake_image):
    """Fixture for a mocked MultiChImgEditorState.

    Args:
        fake_image (NDArray): The fake image fixture.

    Returns:
        MagicMock: Mocked editor state.
    """
    state = MagicMock()
    state.get_img.return_value = fake_image
    state.get_gray_filtered_img.return_value = fake_image
    state.get_small_objects_img.return_value = fake_image
    state.get_midslice.side_effect = lambda img: img[:, :, img.shape[2] // 2]
    return state


@pytest.fixture
def multi_channel_image_viewer(qtbot, mock_editor_state):
    """Fixture for MultiChannelImageViewer widget.

    Args:
        qtbot: pytest-qt bot.
        mock_editor_state: Mocked editor state.

    Returns:
        MultiChannelImageLoader: The widget instance.
    """
    widget = MultiChannelImageLoader(state=mock_editor_state)
    qtbot.addWidget(widget)
    return widget


def test_image_viewer_reads_and_displays_image(
    multi_channel_image_viewer, mock_editor_state
):
    """Test that the image viewer reads and displays the image correctly.

    Args:
        multi_channel_image_viewer (MultiChannelImageLoader): The image viewer widget.
        mock_editor_state (MagicMock): Mocked editor state.
    """
    multi_channel_image_viewer.editor.get_channels = lambda: 1
    multi_channel_image_viewer.editor.get_channel_of_interest = lambda: 0

    with patch.object(
        multi_channel_image_viewer.viewer,
        "setImage",
        lambda img: setattr(multi_channel_image_viewer.viewer, "image", img),
    ):

        def fake_read_img(ch, chi):
            multi_channel_image_viewer._on_image_ready()

        mock_editor_state.read_img.side_effect = fake_read_img

        multi_channel_image_viewer.read_button.click()

    assert hasattr(multi_channel_image_viewer.viewer, "image")
