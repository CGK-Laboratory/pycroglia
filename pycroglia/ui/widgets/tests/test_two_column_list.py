import pytest

from pycroglia.ui.widgets.two_column_list import (
    TwoColumnListWithDelete,
    TwoColumnList,
)


@pytest.fixture
def two_column_list(qtbot):
    """Fixture for TwoColumnList widget.

    Args:
        qtbot: pytest-qt bot for widget handling.

    Returns:
        TwoColumnList: The widget instance.
    """
    widget = TwoColumnList(headers=["Header 1", "Header 2"])
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def two_column_list_with_delete(qtbot):
    """Fixture for TwoColumnListWithDelete widget.

    Args:
        qtbot: pytest-qt bot for widget handling.

    Returns:
        TwoColumnListWithDelete: The widget instance.
    """
    widget = TwoColumnListWithDelete(
        headers=["Header 1", "Header 2"], delete_button_text="Test button"
    )
    qtbot.addWidget(widget)
    return widget


# --- TwoColumnList tests ---


def test_two_column_list_add_item(two_column_list, qtbot):
    """Test that add_item adds a row to the model and emits dataChanged (TwoColumnList)."""
    with qtbot.waitSignal(two_column_list.dataChanged, timeout=100):
        two_column_list.add_item("PDF", "/path/to/file.pdf")

    assert two_column_list.model.rowCount() == 1
    assert two_column_list.model.item(0, 0).text() == "PDF"
    assert two_column_list.model.item(0, 1).text() == "/path/to/file.pdf"


def test_two_column_list_get_column_returns_correct_values(two_column_list):
    """Test that get_column returns the correct values for each column (TwoColumnList).

    Asserts:
        The returned list matches the expected values for each column.
    """
    two_column_list.add_item("PDF", "/path/to/file.pdf")
    two_column_list.add_item("DOCX", "/path/to/file.docx")

    assert two_column_list.get_column(0) == ["PDF", "DOCX"]
    assert two_column_list.get_column(1) == ["/path/to/file.pdf", "/path/to/file.docx"]


def test_two_column_list_get_column_invalid_index_raises(two_column_list):
    """Test that get_column raises ValueError for invalid column index (TwoColumnList).

    Asserts:
        ValueError is raised when column index is not 0 or 1.
    """
    with pytest.raises(ValueError):
        two_column_list.get_column(2)


def test_two_column_list_get_selected_item(two_column_list, qtbot):
    """Test that get_selected_item returns the correct tuple when a row is selected (TwoColumnList)."""
    two_column_list.add_item("PDF", "/file.pdf")
    two_column_list.add_item("DOCX", "/file.docx")
    index = two_column_list.model.index(1, 0)
    two_column_list.table_view.selectRow(index.row())
    result = two_column_list.get_selected_item()
    assert result == ("DOCX", "/file.docx")


def test_two_column_list_get_selected_item_none(two_column_list):
    """Test that get_selected_item returns None when nothing is selected (TwoColumnList)."""
    two_column_list.add_item("PDF", "/file.pdf")
    assert two_column_list.get_selected_item() is None


# --- TwoColumnListWithDelete tests ---


def test_two_column_list_with_delete_selection_enables_delete(
    two_column_list_with_delete, qtbot
):
    """Test that selecting a row enables the delete button (TwoColumnListWithDelete)."""
    two_column_list_with_delete.add_item("PDF", "/path/to/file.pdf")
    index = two_column_list_with_delete.model.index(0, 0)
    two_column_list_with_delete.table_view.selectRow(index.row())

    qtbot.waitUntil(
        lambda: two_column_list_with_delete.delete_button.isEnabled(), timeout=1000
    )

    assert two_column_list_with_delete.delete_button.isEnabled()


def test_two_column_list_with_delete_delete_with_selection(
    two_column_list_with_delete, qtbot
):
    """Test that deleting a selected row removes it from the model and emits dataChanged (TwoColumnListWithDelete)."""
    two_column_list_with_delete.add_item("PDF", "/file.pdf")
    two_column_list_with_delete.add_item("DOCX", "/file.docx")

    two_column_list_with_delete.table_view.selectRow(0)

    with qtbot.waitSignal(two_column_list_with_delete.dataChanged, timeout=1000):
        two_column_list_with_delete._remove_selected_item()

    assert two_column_list_with_delete.model.rowCount() == 1
    assert two_column_list_with_delete.model.item(0, 0).text() == "DOCX"


def test_two_column_list_with_delete_no_delete_without_selection(
    two_column_list_with_delete,
):
    """Test that the delete button is not enabled when no row is selected (TwoColumnListWithDelete)."""
    two_column_list_with_delete.add_item("PDF", "/file.pdf")
    assert not two_column_list_with_delete.delete_button.isEnabled()
