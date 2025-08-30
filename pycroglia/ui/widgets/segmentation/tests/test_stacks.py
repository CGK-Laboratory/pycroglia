import pytest
from unittest.mock import Mock, patch
from PyQt6 import QtWidgets

from pycroglia.ui.widgets.segmentation.stacks import SegmentationEditorStack
from pycroglia.ui.widgets.imagefilters.results import FilterResults

@pytest.fixture
def segmentation_editor_stack():
    """Create a SegmentationEditorStack instance."""
    return SegmentationEditorStack(
        headers_text=["Cell Number", "Cell Size"],
        rollback_button_text="Rollback",
        segmentation_button_text="Segment",
        progress_title="Processing...",
        progress_cancel_text="Cancel"
    )


def test_segmentation_editor_stack__init__(segmentation_editor_stack):
    """Test SegmentationEditorStack initialization.

    Asserts:
        All text parameters are stored and tabs widget is created.
    """
    assert segmentation_editor_stack.headers_text == ["Cell Number", "Cell Size"]
    assert segmentation_editor_stack.rollback_button_text == "Rollback"
    assert segmentation_editor_stack.segmentation_button_text == "Segment"
    assert segmentation_editor_stack.progress_title == "Processing..."
    assert segmentation_editor_stack.progress_cancel_text == "Cancel"
    assert isinstance(segmentation_editor_stack.tabs, QtWidgets.QTabWidget)


def test_segmentation_editor_stack_default__init__():
    """Test SegmentationEditorStack initialization with default values.

    Asserts:
        All text parameters are None when not provided.
    """
    segmentation_stack = SegmentationEditorStack()

    assert segmentation_stack.headers_text is None
    assert segmentation_stack.rollback_button_text is None
    assert segmentation_stack.segmentation_button_text is None
    assert segmentation_stack.progress_title is None
    assert segmentation_stack.progress_cancel_text is None


@patch('pycroglia.ui.widgets.segmentation.stacks.SegmentationEditor')
@patch('pycroglia.ui.widgets.segmentation.stacks.SkimageImgLabeling')
def test_segmentation_editor_stack_add_tabs_creates_editors(mock_labeling_class, mock_editor_class, segmentation_editor_stack):
    """Test add_tabs creates SegmentationEditor for each result.

    Asserts:
        Editors are created with correct parameters and added as tabs.
    """
    mock_result1 = Mock(spec=FilterResults)
    mock_result1.small_object_filtered_img = Mock()
    mock_result1.min_size = 100
    mock_result1.file_path = "/path/to/file1.tif"

    mock_result2 = Mock(spec=FilterResults)
    mock_result2.small_object_filtered_img = Mock()
    mock_result2.min_size = 150
    mock_result2.file_path = "/path/to/file2.tif"

    results = [mock_result1, mock_result2]

    mock_editor1 = Mock()
    mock_editor2 = Mock()
    mock_editor_class.side_effect = [mock_editor1, mock_editor2]

    mock_labeling_strategy = Mock()
    mock_labeling_class.return_value = mock_labeling_strategy

    # Mock the tabs.addTab method to avoid Qt call
    segmentation_editor_stack.tabs.addTab = Mock()

    segmentation_editor_stack.add_tabs(results)

    assert mock_editor_class.call_count == 2
    mock_editor_class.assert_any_call(
        img=mock_result1.small_object_filtered_img,
        labeling_strategy=mock_labeling_strategy,
        min_size=mock_result1.min_size,
        with_progress_bar=True,
        headers=["Cell Number", "Cell Size"],
        rollback_button_text="Rollback",
        segmentation_button_text="Segment",
        progress_title="Processing...",
        progress_cancel_text="Cancel",
        parent=segmentation_editor_stack
    )


def test_segmentation_editor_stack_add_tabs_clears_existing_tabs(segmentation_editor_stack):
    """Test add_tabs clears existing tabs before adding new ones.

    Asserts:
        Tabs are cleared when adding new results.
    """
    segmentation_editor_stack.tabs.clear = Mock()
    segmentation_editor_stack.tabs.addTab = Mock()

    with patch('pycroglia.ui.widgets.segmentation.stacks.SegmentationEditor'), \
            patch('pycroglia.ui.widgets.segmentation.stacks.SkimageImgLabeling'):
        mock_result = Mock(spec=FilterResults)
        mock_result.small_object_filtered_img = Mock()
        mock_result.min_size = 100
        mock_result.file_path = "/path/to/file.tif"

        segmentation_editor_stack.add_tabs([mock_result])

    segmentation_editor_stack.tabs.clear.assert_called_once()


@patch('pycroglia.ui.widgets.segmentation.stacks.Path')
def test_segmentation_editor_stack_add_tabs_sets_correct_tab_names(mock_path_class, segmentation_editor_stack):
    """Test add_tabs sets correct tab names from file paths.

    Asserts:
        Tab names are set using file names from paths.
    """
    mock_result = Mock(spec=FilterResults)
    mock_result.small_object_filtered_img = Mock()
    mock_result.min_size = 100
    mock_result.file_path = "/path/to/test_file.tif"

    mock_path = Mock()
    mock_path.name = "test_file.tif"
    mock_path_class.return_value = mock_path

    segmentation_editor_stack.tabs.addTab = Mock()

    with patch('pycroglia.ui.widgets.segmentation.stacks.SegmentationEditor') as mock_editor_class, \
            patch('pycroglia.ui.widgets.segmentation.stacks.SkimageImgLabeling'):
        mock_editor = Mock()
        mock_editor_class.return_value = mock_editor

        segmentation_editor_stack.add_tabs([mock_result])

    mock_path_class.assert_called_with(mock_result.file_path)
    segmentation_editor_stack.tabs.addTab.assert_called_with(mock_editor, "test_file.tif")
