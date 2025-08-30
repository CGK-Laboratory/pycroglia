import pytest
import numpy as np

from numpy.typing import NDArray
from PyQt6 import QtWidgets
from unittest.mock import MagicMock, patch

from pycroglia.ui.widgets.imagefilters.editors import (
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
        "pycroglia.ui.widgets.imagefilters.editors.MultiChImgEditorState",
        lambda file_path: mock_state,
    )

    widget = MultiChannelFilterEditor(file_path="dummy.tif")
    qtbot.addWidget(widget)

    assert isinstance(widget.img_viewer, QtWidgets.QWidget)
    assert isinstance(widget.gray_filter_editor, QtWidgets.QWidget)
    assert isinstance(widget.small_object_filter_editor, QtWidgets.QWidget)


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
        "pycroglia.ui.widgets.imagefilters.editors.MultiChImgEditorState",
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
