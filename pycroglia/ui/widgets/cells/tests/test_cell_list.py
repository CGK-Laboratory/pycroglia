import pytest
import numpy as np

from pycroglia.ui.widgets.cells.cell_list import CellList
from pycroglia.core.labeled_cells import LabeledCells, SkimageImgLabeling
from pycroglia.core.enums import SkimageCellConnectivity


@pytest.fixture
def labeled_cells_simple():
    """Fixture for a simple LabeledCells object with two cells."""
    img = np.zeros((1, 4, 4), dtype=np.uint8)
    img[0, 0, 0] = 1
    img[0, 0, 1] = 1
    img[0, 1, 0] = 1
    img[0, 3, 3] = 1
    labeling = SkimageImgLabeling(SkimageCellConnectivity.EDGES)
    return LabeledCells(img, labeling)


@pytest.fixture
def cell_list(qtbot):
    """Fixture for CellList widget."""
    widget = CellList(headers=["Cell", "Size"])
    qtbot.addWidget(widget)
    return widget


def test_add_cells(cell_list, labeled_cells_simple):
    """Test adding cells to the list."""
    cell_list.add_cells(labeled_cells_simple)
    assert cell_list.list.model.rowCount() == 2

    sizes = [int(cell_list.list.model.item(i, 1).text()) for i in range(2)]
    assert sizes == sorted(sizes, reverse=True)


def test_clear_cells(cell_list, labeled_cells_simple):
    """Test clearing cells from the list."""
    cell_list.add_cells(labeled_cells_simple)
    cell_list.clear_cells()
    assert cell_list.list.model.rowCount() == 0


def test_get_selected_cell_id(cell_list, labeled_cells_simple):
    """Test getting selected cell id."""
    cell_list.add_cells(labeled_cells_simple)

    index = cell_list.list.model.index(0, 0)
    cell_list.list.table_view.selectRow(index.row())
    assert cell_list.get_selected_cell_id() == int(
        cell_list.list.model.item(0, 0).text()
    )


def test_get_selected_cell_info(cell_list, labeled_cells_simple):
    """Test getting selected cell info."""
    cell_list.add_cells(labeled_cells_simple)

    index = cell_list.list.model.index(0, 0)
    cell_list.list.table_view.selectRow(index.row())
    info = cell_list.get_selected_cell_info()
    assert info[0] == int(cell_list.list.model.item(0, 0).text())
    assert info[1] == int(cell_list.list.model.item(0, 1).text())


def test_get_selected_cell_id_none(cell_list):
    """Test get_selected_cell_id returns None if nothing is selected."""
    assert cell_list.get_selected_cell_id() is None


def test_get_selected_cell_info_none(cell_list):
    """Test get_selected_cell_info returns None if nothing is selected."""
    assert cell_list.get_selected_cell_info() is None
