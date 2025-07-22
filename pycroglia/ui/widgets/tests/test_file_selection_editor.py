import pytest
from pycroglia.ui.widgets.file_selection_editor import FileSelectionEditor


@pytest.fixture
def file_selection_editor(qtbot):
    """Fixture for FileSelectionEditor widget.

    Args:
        qtbot: pytest-qt bot for widget handling.

    Returns:
        FileSelectionEditor: The widget instance.
    """
    widget = FileSelectionEditor(
        headers=["Type", "Path"],
        delete_button_text="Delete",
        open_file_text="Open file:",
        open_button_text="Browse",
        open_dialog_title="Open File",
        open_dialog_default_path="/tmp",
        file_filters="All Files (*)",
    )
    qtbot.addWidget(widget)
    return widget


def test_file_selector_emits_signal(file_selection_editor, qtbot):
    """Test that the file selector emits the dataChanged signal and updates the file list."""
    test_path = "/tmp/data.json"

    with qtbot.waitSignal(file_selection_editor.dataChanged, timeout=1000):
        file_selection_editor.file_selector.fileSelected.emit(test_path)

    assert file_selection_editor.file_list.model.rowCount() == 1
    assert file_selection_editor.file_list.model.item(0, 0).text() == ".json"
    assert file_selection_editor.file_list.model.item(0, 1).text() == test_path


def test_on_file_added(file_selection_editor, qtbot):
    """Test that _on_file_added adds a file and emits the dataChanged signal."""
    test_path = "/tmp/test_file.csv"

    with qtbot.waitSignal(file_selection_editor.dataChanged, timeout=1000):
        file_selection_editor._on_file_added(test_path)

    assert file_selection_editor.file_list.model.rowCount() == 1
    assert file_selection_editor.file_list.model.item(0, 0).text() == ".csv"
    assert file_selection_editor.file_list.model.item(0, 1).text() == test_path


def test_on_file_added_empty_path(file_selection_editor):
    """Test that _on_file_added does not add a file when the path is empty."""
    initial_rows = file_selection_editor.file_list.model.rowCount()
    file_selection_editor._on_file_added("")
    assert file_selection_editor.file_list.model.rowCount() == initial_rows


def test_get_files_returns_file_paths(file_selection_editor):
    """Test that get_files returns the correct list of file paths.

    Asserts:
        The returned list matches the file paths added to the file list.
    """
    file_selection_editor.file_list.add_item(".csv", "/tmp/test1.csv")
    file_selection_editor.file_list.add_item(".json", "/tmp/test2.json")
    assert file_selection_editor.get_files() == ["/tmp/test1.csv", "/tmp/test2.json"]


def test_get_files_empty(file_selection_editor):
    """Test that get_files returns an empty list when no files are present.

    Asserts:
        The returned list is empty.
    """
    assert file_selection_editor.get_files() == []
