import pytest
from unittest.mock import Mock, patch
from PyQt6 import QtWidgets

from pycroglia.ui.widgets.imagefilters.stacks import FilterEditorStack
from pycroglia.ui.widgets.imagefilters.results import FilterResults


@pytest.fixture
def filter_editor_stack():
    """Create a FilterEditorStack instance."""
    return FilterEditorStack(
        img_viewer_label="Test Viewer",
        read_button_text="Load",
        channels_label="Channels:",
        channel_of_interest_label="Channel:",
        gray_filter_label="Gray Filter",
        gray_filter_slider_label="Threshold:",
        small_objects_filter_label="Small Objects",
        small_objects_threshold_label="Min Size:",
    )


def test_filter_editor_stack__init__(filter_editor_stack):
    """Test FilterEditorStack initialization.

    Asserts:
        All text parameters are stored and tabs widget is created.
    """
    assert filter_editor_stack.img_viewer_label == "Test Viewer"
    assert filter_editor_stack.read_button_text == "Load"
    assert filter_editor_stack.channels_label == "Channels:"
    assert filter_editor_stack.channel_of_interest_label == "Channel:"
    assert filter_editor_stack.gray_filter_label == "Gray Filter"
    assert filter_editor_stack.gray_filter_slider_label == "Threshold:"
    assert filter_editor_stack.small_objects_filter_label == "Small Objects"
    assert filter_editor_stack.small_objects_threshold_label == "Min Size:"
    assert isinstance(filter_editor_stack.tabs, QtWidgets.QTabWidget)


def test_filter_editor_stack_default__init__():
    """Test FilterEditorStack initialization with default values.

    Asserts:
        All text parameters are None when not provided.
    """
    filter_stack = FilterEditorStack()

    assert filter_stack.img_viewer_label is None
    assert filter_stack.read_button_text is None
    assert filter_stack.channels_label is None
    assert filter_stack.channel_of_interest_label is None
    assert filter_stack.gray_filter_label is None
    assert filter_stack.gray_filter_slider_label is None
    assert filter_stack.small_objects_filter_label is None
    assert filter_stack.small_objects_threshold_label is None


@patch("pycroglia.ui.widgets.imagefilters.stacks.MultiChannelFilterEditor")
def test_filter_editor_stack_add_tabs_creates_editors(
    mock_editor_class, filter_editor_stack
):
    """Test add_tabs creates MultiChannelFilterEditor for each file.

    Asserts:
        Editors are created with correct parameters and added as tabs.
    """
    files = ["/path/to/file1.tif", "/path/to/file2.tif"]
    mock_editor1 = Mock()
    mock_editor2 = Mock()
    mock_editor_class.side_effect = [mock_editor1, mock_editor2]

    # Mock the tabs.addTab method to avoid Qt call
    filter_editor_stack.tabs.addTab = Mock()

    filter_editor_stack.add_tabs(files)

    assert mock_editor_class.call_count == 2
    mock_editor_class.assert_any_call(
        files[0],
        img_viewer_label="Test Viewer",
        read_button_text="Load",
        channels_label="Channels:",
        channel_of_interest_label="Channel:",
        gray_filter_label="Gray Filter",
        gray_filter_slider_label="Threshold:",
        small_objects_filter_label="Small Objects",
        small_objects_threshold_label="Min Size:",
        parent=filter_editor_stack,
    )


def test_filter_editor_stack_add_tabs_clears_existing_tabs(filter_editor_stack):
    """Test add_tabs clears existing tabs before adding new ones.

    Asserts:
        Tabs are cleared when adding new files.
    """
    filter_editor_stack.tabs.clear = Mock()
    filter_editor_stack.tabs.addTab = Mock()

    with patch("pycroglia.ui.widgets.imagefilters.stacks.MultiChannelFilterEditor"):
        filter_editor_stack.add_tabs(["/path/to/file.tif"])

    filter_editor_stack.tabs.clear.assert_called_once()


def test_filter_editor_stack_get_results_returns_filter_results(filter_editor_stack):
    """Test get_results returns FilterResults from all editors.

    Asserts:
        Results are collected from all MultiChannelFilterEditor tabs.
    """
    # Create mock editors with get_filter_results method
    mock_editor1 = Mock()
    mock_editor1.get_filter_results.return_value = Mock(spec=FilterResults)
    mock_editor2 = Mock()
    mock_editor2.get_filter_results.return_value = Mock(spec=FilterResults)

    filter_editor_stack.tabs.count = Mock(return_value=2)
    filter_editor_stack.tabs.widget = Mock(side_effect=[mock_editor1, mock_editor2])

    # Mock hasattr to simulate MultiChannelFilterEditor detection
    with patch("builtins.hasattr") as mock_hasattr:
        mock_hasattr.return_value = True

        results = filter_editor_stack.get_results()

    assert len(results) == 2
    mock_editor1.get_filter_results.assert_called_once()
    mock_editor2.get_filter_results.assert_called_once()
