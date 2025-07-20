import pytest

from pycroglia.ui.widgets.two_column_list import TwoColumnList


@pytest.fixture
def two_column_list(qtbot):
    """Fixture for TwoColumnList widget.

    Args:
        qtbot: pytest-qt bot for widget handling.

    Returns:
        TwoColumnList: The widget instance.
    """
    widget = TwoColumnList(
        headers=["Header 1", "Header 2"], delete_button_text="Test button"
    )
    qtbot.addWidget(widget)
    return widget


def test_add_item(two_column_list, qtbot):
    """Test that add_item adds a row to the model and emits dataChanged."""
    with qtbot.waitSignal(two_column_list.dataChanged, timeout=100):
        two_column_list.add_item("PDF", "/path/to/file.pdf")

    assert two_column_list.model.rowCount() == 1
    assert two_column_list.model.item(0, 0).text() == "PDF"
    assert two_column_list.model.item(0, 1).text() == "/path/to/file.pdf"


def test_selection_enables_delete(two_column_list, qtbot):
    """Test that selecting a row enables the delete button."""
    two_column_list.add_item("PDF", "/path/to/file.pdf")
    index = two_column_list.model.index(0, 0)
    two_column_list.table_view.selectRow(index.row())

    qtbot.waitUntil(lambda: two_column_list.delete_button.isEnabled(), timeout=1000)

    assert two_column_list.delete_button.isEnabled()


def test_delete_with_selection(two_column_list, qtbot):
    """Test that deleting a selected row removes it from the model and emits dataChanged."""
    two_column_list.add_item("PDF", "/file.pdf")
    two_column_list.add_item("DOCX", "/file.docx")

    two_column_list.table_view.selectRow(0)

    with qtbot.waitSignal(two_column_list.dataChanged, timeout=1000):
        two_column_list._remove_selected_item()

    assert two_column_list.model.rowCount() == 1
    assert two_column_list.model.item(0, 0).text() == "DOCX"


def test_no_delete_without_selection(two_column_list):
    """Test that the delete button is not enabled when no row is selected."""
    two_column_list.add_item("PDF", "/file.pdf")
    assert not two_column_list.delete_button.isEnabled()


def test_get_column_returns_correct_values(two_column_list):
    """Test that get_column returns the correct values for each column.

    Asserts:
        The returned list matches the expected values for each column.
    """
    two_column_list.add_item("PDF", "/path/to/file.pdf")
    two_column_list.add_item("DOCX", "/path/to/file.docx")

    assert two_column_list.get_column(0) == ["PDF", "DOCX"]
    assert two_column_list.get_column(1) == ["/path/to/file.pdf", "/path/to/file.docx"]


def test_get_column_invalid_index_raises(two_column_list):
    """Test that get_column raises ValueError for invalid column index.

    Asserts:
        ValueError is raised when column index is not 0 or 1.
    """
    with pytest.raises(ValueError):
        two_column_list.get_column(2)
