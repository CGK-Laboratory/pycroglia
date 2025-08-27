import pytest
from unittest.mock import MagicMock, patch
from numpy.typing import NDArray
import numpy as np

from PyQt6 import QtWidgets

from pycroglia.ui.widgets.ch_img import (
    MultiChannelImageViewer,
    GrayFilterEditor,
    SmallObjectsFilterEditor,
    MultiChannelFilterEditor,
)


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
        MultiChannelImageViewer: The widget instance.
    """
    widget = MultiChannelImageViewer(state=mock_editor_state)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def gray_filter_editor(qtbot, mock_editor_state):
    """Fixture for GrayFilterEditor widget.

    Args:
        qtbot: pytest-qt bot.
        mock_editor_state: Mocked editor state.

    Returns:
        GrayFilterEditor: The widget instance.
    """
    widget = GrayFilterEditor(state=mock_editor_state)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def small_objects_filter_editor(qtbot, mock_editor_state):
    """Fixture for SmallObjectsFilterEditor widget.

    Args:
        qtbot: pytest-qt bot.
        mock_editor_state: Mocked editor state.

    Returns:
        SmallObjectsFilterEditor: The widget instance.
    """
    widget = SmallObjectsFilterEditor(state=mock_editor_state)
    qtbot.addWidget(widget)
    return widget


def test_image_viewer_reads_and_displays_image(
    multi_channel_image_viewer, mock_editor_state, fake_image
):
    """Test that the image viewer reads and displays the image correctly.

    Args:
        multi_channel_image_viewer (MultiChannelImageViewer): The image viewer widget.
        mock_editor_state (MagicMock): Mocked editor state.
        fake_image (NDArray): The fake image.
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


def test_gray_filter_updates_on_slider(
    gray_filter_editor, mock_editor_state, fake_image
):
    """Test that the gray filter editor updates the image when the slider changes.

    Args:
        gray_filter_editor (GrayFilterEditor): The gray filter editor widget.
        mock_editor_state (MagicMock): Mocked editor state.
        fake_image (NDArray): The fake image.
    """
    with patch.object(
        gray_filter_editor.viewer,
        "setImage",
        lambda img: setattr(gray_filter_editor.viewer, "image", img),
    ):

        def fake_apply_otsu_gray_filter(val):
            gray_filter_editor._on_image_ready()

        mock_editor_state.apply_otsu_gray_filter.side_effect = (
            fake_apply_otsu_gray_filter
        )

        gray_filter_editor.slider.set_value(1.0)

    assert hasattr(gray_filter_editor.viewer, "image")


def test_small_filter_updates_on_spinbox(
    small_objects_filter_editor, mock_editor_state, fake_image
):
    """Test that the small objects filter editor updates the image when the spinbox changes.

    Args:
        small_objects_filter_editor (SmallObjectsFilterEditor): The small objects filter editor widget.
        mock_editor_state (MagicMock): Mocked editor state.
        fake_image (NDArray): The fake image.
    """
    with patch.object(
        small_objects_filter_editor.viewer,
        "setImage",
        lambda img: setattr(small_objects_filter_editor.viewer, "image", img),
    ):

        def fake_apply_small_object_filter(val):
            small_objects_filter_editor._on_image_ready()

        mock_editor_state.apply_small_object_filter.side_effect = (
            fake_apply_small_object_filter
        )

        small_objects_filter_editor.spin_box.spin_box.setValue(5)

    assert hasattr(small_objects_filter_editor.viewer, "image")


def test_multichannel_filter_editor_init(qtbot, monkeypatch):
    """Test initialization of MultiChannelFilterEditor and its subwidgets.

    Args:
        qtbot: pytest-qt bot.
        monkeypatch: pytest monkeypatch fixture.
    """
    mock_state = MagicMock()
    monkeypatch.setattr(
        "pycroglia.ui.widgets.ch_img.MultiChImgEditorState",
        lambda file_path: mock_state,
    )

    widget = MultiChannelFilterEditor(file_path="dummy.tif")
    qtbot.addWidget(widget)

    assert isinstance(widget.img_viewer, QtWidgets.QWidget)
    assert isinstance(widget.gray_filter_editor, QtWidgets.QWidget)
    assert isinstance(widget.small_object_filter_editor, QtWidgets.QWidget)


def test_filter_results_as_dict():
    """Test that FilterResults.as_dict returns the correct dictionary."""
    from pycroglia.ui.widgets.ch_img import FilterResults

    dummy_img = np.ones((2, 2, 2), dtype=np.uint8)
    fr = FilterResults(
        file_path="file.tif",
        gray_filter_value=1.5,
        min_size=10,
        small_object_filtered_img=dummy_img,
    )
    result = fr.as_dict()
    assert result["file_path"] == "file.tif"
    assert result["gray_filter_value"] == 1.5
    assert result["min_size"] == 10
    assert np.array_equal(result["small_object_filtered_img"], dummy_img)


def test_multichannel_filter_editor_get_filter_results(qtbot, monkeypatch):
    """Test that MultiChannelFilterEditor.get_filter_results returns a FilterResults object with correct values."""
    dummy_img = np.ones((2, 2, 2), dtype=np.uint8)

    class DummySlider:
        def get_value(self):
            return 2.0

    class DummySpinBox:
        def get_value(self):
            return 7

    class DummyGrayFilterEditor:
        slider = DummySlider()

    class DummySmallObjectFilterEditor:
        spin_box = DummySpinBox()

    class DummyEditorState:
        imageChanged = MagicMock()
        grayImageChanged = MagicMock()

        def get_small_objects_img(self):
            return dummy_img

    monkeypatch.setattr(
        "pycroglia.ui.widgets.ch_img.MultiChImgEditorState",
        lambda file_path: DummyEditorState(),
    )

    widget = MultiChannelFilterEditor(file_path="dummy.tif")
    widget.gray_filter_editor = DummyGrayFilterEditor()
    widget.small_object_filter_editor = DummySmallObjectFilterEditor()
    widget.editor_state = DummyEditorState()

    result = widget.get_filter_results()
    assert result.file_path == "dummy.tif"
    assert result.gray_filter_value == 2.0
    assert result.min_size == 7
    assert np.array_equal(result.small_object_filtered_img, dummy_img)
