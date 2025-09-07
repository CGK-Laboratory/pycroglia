import pytest
import numpy as np
from unittest.mock import patch

from PyQt6 import QtCore
from pycroglia.ui.widgets.analysis.cell_selector import (
    CellSelector,
    CellSelectorControlPanel,
    ColorType,
)
from pycroglia.core.labeled_cells import LabeledCells, SkimageImgLabeling
from pycroglia.core.enums import SkimageCellConnectivity


@pytest.fixture
def labeled_cells_with_borders():
    """Fixture for LabeledCells with border cells."""
    img = np.zeros((2, 5, 5), dtype=np.uint8)
    # Border cell
    img[0, 0, 0] = 1
    img[0, 0, 1] = 1
    # Center cell
    img[0, 2, 2] = 1
    img[1, 2, 2] = 1
    # Another border cell
    img[1, 4, 4] = 1

    labeling = SkimageImgLabeling(SkimageCellConnectivity.EDGES)
    return LabeledCells(img, labeling)


@pytest.fixture
def cell_selector(qtbot, labeled_cells_with_borders):
    """Fixture for CellSelector widget."""
    widget = CellSelector(labeled_cells_with_borders)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def control_panel(qtbot):
    """Fixture for CellSelectorControlPanel widget."""
    widget = CellSelectorControlPanel(max_cell_size=100)
    qtbot.addWidget(widget)
    return widget


# CellSelectorControlPanel tests
def test_cell_selector_control_panel_initialization_default_texts(control_panel):
    """Test control panel initialization with default texts."""
    assert (
        control_panel.remove_btn.text()
        == CellSelectorControlPanel.DEFAULT_REMOVE_BUTTON_TEXT
    )
    assert (
        control_panel.size_btn.text()
        == CellSelectorControlPanel.DEFAULT_SIZE_BUTTON_TEXT
    )
    assert (
        control_panel.preview_btn.text()
        == CellSelectorControlPanel.DEFAULT_PREVIEW_BUTTON_TEXT
    )
    assert (
        control_panel.border_checkbox.text()
        == CellSelectorControlPanel.DEFAULT_BORDER_CHECKBOX_TEXT
    )


def test_cell_selector_control_panel_initialization_custom_texts(qtbot):
    """Test control panel initialization with custom texts."""
    custom_texts = {
        "remove_button_text": "Delete Cell",
        "size_button_text": "Filter Small",
        "preview_button_text": "Show Preview",
        "border_checkbox_text": "Exclude Border",
    }

    panel = CellSelectorControlPanel(max_cell_size=50, **custom_texts)
    qtbot.addWidget(panel)

    assert panel.remove_btn.text() == custom_texts["remove_button_text"]
    assert panel.size_btn.text() == custom_texts["size_button_text"]
    assert panel.preview_btn.text() == custom_texts["preview_button_text"]
    assert panel.border_checkbox.text() == custom_texts["border_checkbox_text"]


def test_cell_selector_control_panel_remove_button_initially_disabled(control_panel):
    """Test that remove button is initially disabled."""
    assert not control_panel.remove_btn.isEnabled()


def test_color_type_enum_values():
    """Test ColorType enum values."""
    assert ColorType.SELECTED.value == "selected"
    assert ColorType.UNSELECTED.value == "unselected"


def test_cell_selector_initialization_basic(cell_selector, labeled_cells_with_borders):
    """Test CellSelector initialization."""
    assert cell_selector.img == labeled_cells_with_borders
    assert len(cell_selector.border_cells) > 0
    assert len(cell_selector.unselected_cells) == 0
    assert cell_selector.headers_text == CellSelector.DEFAULT_HEADERS_TEXT


def test_cell_selector_initialization_custom_headers(qtbot, labeled_cells_with_borders):
    """Test CellSelector initialization with custom headers."""
    custom_headers = ["ID", "Volume", "Area"]
    selector = CellSelector(labeled_cells_with_borders, headers=custom_headers)
    qtbot.addWidget(selector)

    assert selector.headers_text == custom_headers


def test_cell_selector_get_border_cells_detection(cell_selector):
    """Test that border cells are correctly detected."""
    border_cells = cell_selector.get_border_cells()
    assert len(border_cells) > 0

    # Check that detected border cells are valid
    for cell_id in border_cells:
        assert 1 <= cell_id <= cell_selector.img.len()


def test_cell_selector_build_cell_to_row_cache_completeness(cell_selector):
    """Test building of cell-to-row cache."""
    assert len(cell_selector._cell_to_row_cache) == cell_selector.img.len()

    for cell_id in range(1, cell_selector.img.len() + 1):
        assert cell_id in cell_selector._cell_to_row_cache


def test_cell_selector_on_remove_button_clicked_unselect(cell_selector, qtbot):
    """Test remove button click to unselect a cell."""
    # Select first cell
    index = cell_selector.viewer.cell_list.list.model.index(0, 0)
    cell_selector.viewer.cell_list.list.table_view.selectRow(index.row())

    # Get selected cell ID
    selected_cell = cell_selector.viewer.cell_list.get_selected_cell_id()

    # Click remove button
    qtbot.mouseClick(
        cell_selector.control_panel.remove_btn, QtCore.Qt.MouseButton.LeftButton
    )

    assert selected_cell in cell_selector.unselected_cells


def test_cell_selector_on_remove_button_clicked_reselect(cell_selector, qtbot):
    """Test remove button click to reselect a previously unselected cell."""
    # Select and unselect first cell
    index = cell_selector.viewer.cell_list.list.model.index(0, 0)
    cell_selector.viewer.cell_list.list.table_view.selectRow(index.row())
    selected_cell = cell_selector.viewer.cell_list.get_selected_cell_id()

    # First click - unselect
    qtbot.mouseClick(
        cell_selector.control_panel.remove_btn, QtCore.Qt.MouseButton.LeftButton
    )
    assert selected_cell in cell_selector.unselected_cells

    # Second click - reselect
    qtbot.mouseClick(
        cell_selector.control_panel.remove_btn, QtCore.Qt.MouseButton.LeftButton
    )
    assert selected_cell not in cell_selector.unselected_cells


def test_cell_selector_on_size_button_clicked_filtering(cell_selector, qtbot):
    """Test size-based filtering."""
    threshold = 2
    with patch.object(
        cell_selector.control_panel.size_input, "get_value", return_value=threshold
    ):
        qtbot.mouseClick(
            cell_selector.control_panel.size_btn, QtCore.Qt.MouseButton.LeftButton
        )

        # Check that cells smaller than threshold are unselected
        for cell_id in range(1, cell_selector.img.len() + 1):
            cell_size = cell_selector.img.get_cell_size(cell_id)
            if cell_size < threshold:
                assert cell_id in cell_selector.unselected_cells


def test_cell_selector_on_border_checkbox_toggled_check(cell_selector, qtbot):
    """Test border checkbox when checked."""
    cell_selector.control_panel.border_checkbox.setChecked(True)

    # All border cells should be unselected
    for border_cell in cell_selector.border_cells:
        assert border_cell in cell_selector.unselected_cells


@patch("pycroglia.ui.widgets.analysis.dialog.PreviewDialog.exec")
def test_cell_selector_on_preview_button_clicked_with_selected_cells(
    mock_exec, cell_selector, qtbot
):
    """Test preview button with selected cells."""
    qtbot.mouseClick(
        cell_selector.control_panel.preview_btn, QtCore.Qt.MouseButton.LeftButton
    )
    mock_exec.assert_called_once()


@patch("pycroglia.ui.widgets.analysis.dialog.PreviewDialog.exec")
def test_cell_selector_on_preview_button_clicked_no_selected_cells(
    mock_exec, cell_selector, qtbot
):
    """Test preview button when no cells are selected."""
    # Unselect all cells
    cell_selector.unselected_cells = set(range(1, cell_selector.img.len() + 1))

    qtbot.mouseClick(
        cell_selector.control_panel.preview_btn, QtCore.Qt.MouseButton.LeftButton
    )
    mock_exec.assert_called_once()


def test_cell_selector_get_selected_cells_basic(cell_selector):
    """Test getting selected cells."""
    # Initially all cells should be selected
    selected = cell_selector.get_selected_cells()
    assert len(selected) == cell_selector.img.len()

    # Unselect one cell
    cell_selector.unselected_cells.add(1)
    selected = cell_selector.get_selected_cells()
    assert len(selected) == cell_selector.img.len() - 1
    assert 1 not in selected


def test_cell_selector_get_unselected_cells_basic(cell_selector):
    """Test getting unselected cells."""
    unselected = cell_selector.get_unselected_cells()
    assert len(unselected) == 0

    cell_selector.unselected_cells.add(1)
    unselected = cell_selector.get_unselected_cells()
    assert len(unselected) == 1
    assert 1 in unselected


def test_cell_selector_get_border_cells_basic(cell_selector):
    """Test getting border cells."""
    border_cells = cell_selector.get_border_cells()
    assert isinstance(border_cells, set)
    assert len(border_cells) > 0


def test_cell_selector_set_row_color_valid_cell(cell_selector):
    """Test setting row color."""
    cell_id = 1

    # Test unselected color
    cell_selector._set_row_color(cell_id, ColorType.UNSELECTED)
    row = cell_selector._cell_to_row_cache[cell_id]
    model = cell_selector.viewer.cell_list.list.model
    item = model.item(row, 0)
    assert item.background().color() == CellSelector.UNSELECTED_COLOR

    # Test selected color
    cell_selector._set_row_color(cell_id, ColorType.SELECTED)
    assert item.background() == CellSelector.SELECTED_COLOR


def test_cell_selector_set_row_color_invalid_cell(cell_selector):
    """Test setting row color for invalid cell ID."""
    # Should not crash
    cell_selector._set_row_color(999, ColorType.UNSELECTED)


def test_cell_selector_update_colors_batch_multiple_cells(cell_selector):
    """Test batch color updates."""
    cell_ids = {1, 2}
    cell_selector._update_colors_batch(cell_ids, ColorType.UNSELECTED)

    model = cell_selector.viewer.cell_list.list.model
    for cell_id in cell_ids:
        if cell_id in cell_selector._cell_to_row_cache:
            row = cell_selector._cell_to_row_cache[cell_id]
            item = model.item(row, 0)
            assert item.background().color() == CellSelector.UNSELECTED_COLOR


def test_cell_selector_on_cell_selection_changed_enables_remove_button(
    cell_selector, qtbot
):
    """Test that selecting a cell enables the remove button."""
    # Initially remove button should be disabled
    assert not cell_selector.control_panel.remove_btn.isEnabled()

    # Select a cell
    index = cell_selector.viewer.cell_list.list.model.index(0, 0)
    cell_selector.viewer.cell_list.list.table_view.selectRow(index.row())

    # Remove button should now be enabled
    assert cell_selector.control_panel.remove_btn.isEnabled()


def test_cell_selector_load_data_updates_cache(cell_selector):
    """Test that loading data updates the cache."""
    original_cache_size = len(cell_selector._cell_to_row_cache)
    cell_selector._load_data(cell_selector.img)

    # Cache should be rebuilt with same size
    assert len(cell_selector._cell_to_row_cache) == original_cache_size
