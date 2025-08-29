import numpy as np
import pytest

from numpy.typing import NDArray
from unittest.mock import MagicMock


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
