import pytest
import numpy as np

from PyQt6 import QtWidgets
from pycroglia.ui.widgets.analysis.dialog import PreviewDialog


@pytest.fixture
def sample_image():
    """Fixture for sample 2D image."""
    return np.random.randint(0, 255, (50, 50), dtype=np.uint8)


@pytest.fixture
def preview_dialog(qtbot, sample_image):
    """Fixture for PreviewDialog widget."""
    dialog = PreviewDialog(sample_image)
    qtbot.addWidget(dialog)
    return dialog


def test_preview_dialog_initialization_default_values(preview_dialog, sample_image):
    """Test PreviewDialog initialization with default values."""
    assert preview_dialog.windowTitle() == PreviewDialog.DEFAULT_WINDOW_TITLE
    assert preview_dialog.isModal()
    assert preview_dialog.img_viewer is not None

    # Check dialog size
    expected_size = PreviewDialog.DEFAULT_DIALOG_SIZE
    actual_size = (preview_dialog.width(), preview_dialog.height())
    assert actual_size == expected_size


def test_preview_dialog_initialization_all_custom(qtbot, sample_image):
    """Test PreviewDialog initialization with all custom parameters."""
    custom_title = "Full Custom Dialog"
    custom_size = (1200, 900)
    parent = QtWidgets.QWidget()

    dialog = PreviewDialog(
        sample_image, title_txt=custom_title, dialog_size=custom_size, parent=parent
    )
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == custom_title
    assert (dialog.width(), dialog.height()) == custom_size
    assert dialog.parent() == parent


def test_preview_dialog_empty_image(qtbot):
    """Test with empty image."""
    empty_image = np.zeros((10, 10), dtype=np.uint8)
    dialog = PreviewDialog(empty_image)
    qtbot.addWidget(dialog)

    assert dialog.img_viewer.img_viewer.image is not None
    assert dialog.img_viewer.img_viewer.image.shape == (10, 10)


def test_preview_dialog_none_parameters_use_defaults(qtbot, sample_image):
    """Test that None parameters use default values."""
    dialog = PreviewDialog(sample_image, title_txt=None, dialog_size=None)
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == PreviewDialog.DEFAULT_WINDOW_TITLE
    assert (dialog.width(), dialog.height()) == PreviewDialog.DEFAULT_DIALOG_SIZE
