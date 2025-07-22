import pytest

from unittest.mock import patch
from PyQt6 import QtCore
from PyQt6.QtTest import QSignalSpy

from pycroglia.ui.widgets.file_selector import FileSelector


@pytest.fixture
def file_selector(qtbot):
    """Fixture for FileSelector widget.

    Args:
        qtbot: pytest-qt bot for widget handling.

    Returns:
        FileSelector: The widget instance.
    """
    widget = FileSelector(
        label_text="Test Label",
        button_text="Test Button",
        dialog_title="Test Dialog",
        file_filters="Test Files (*.test);;All Files (*)",
    )
    qtbot.addWidget(widget)
    return widget


@patch("PyQt6.QtWidgets.QFileDialog.getOpenFileName")
def test_button_click_with_selection(mock_dialog, file_selector, qtbot):
    """Test that clicking the button emits the fileSelected signal when a file is chosen."""
    test_path = "/path/example.txt"
    mock_dialog.return_value = (test_path, "Test Files (*.test)")

    spy = QSignalSpy(file_selector.fileSelected)
    qtbot.mouseClick(file_selector.button, QtCore.Qt.MouseButton.LeftButton)

    assert len(spy) == 1
    mock_dialog.assert_called_once_with(
        parent=file_selector,
        caption="Test Dialog",
        filter="Test Files (*.test);;All Files (*)",
        directory=QtCore.QDir.homePath(),
    )


@patch("PyQt6.QtWidgets.QFileDialog.getOpenFileName")
def test_button_click_without_selection(mock_dialog, file_selector, qtbot):
    """Test that clicking the button does not emit the fileSelected signal if no file is chosen."""
    mock_dialog.return_value = ("", "")

    spy = QSignalSpy(file_selector.fileSelected)
    qtbot.mouseClick(file_selector.button, QtCore.Qt.MouseButton.LeftButton)

    mock_dialog.assert_called_once()
    assert len(spy) == 0
