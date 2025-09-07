import pytest
import numpy as np

from PyQt6 import QtWidgets
from pycroglia.ui.widgets.cells.cells_panel import CellsPanel
from pycroglia.core.labeled_cells import LabeledCells, SkimageImgLabeling
from pycroglia.core.enums import SkimageCellConnectivity


@pytest.fixture
def labeled_cells_simple():
    """Fixture for a simple LabeledCells object with two cells."""
    img = np.zeros((2, 4, 4), dtype=np.uint8)
    img[0, 0, 0] = 1
    img[0, 0, 1] = 1
    img[0, 1, 0] = 1
    img[1, 3, 3] = 1
    img[1, 2, 2] = 1
    labeling = SkimageImgLabeling(SkimageCellConnectivity.EDGES)
    return LabeledCells(img, labeling)


@pytest.fixture
def cells_panel(qtbot, labeled_cells_simple):
    """Fixture for CellsPanel widget."""
    widget = CellsPanel(labeled_cells_simple)
    qtbot.addWidget(widget)
    return widget


def test_cells_panel_initialization(cells_panel, labeled_cells_simple):
    """Test CellsPanel initialization."""
    assert cells_panel.img == labeled_cells_simple
    assert cells_panel.headers == CellsPanel.DEFAULT_HEADERS_TEXT
    assert cells_panel.cell_list is not None
    assert cells_panel.multi_viewer is not None
    assert cells_panel.cell_viewer is not None


def test_cells_panel_initialization_with_custom_headers(qtbot, labeled_cells_simple):
    """Test CellsPanel initialization with custom headers."""
    custom_headers = ["ID", "Volume"]
    panel = CellsPanel(labeled_cells_simple, headers=custom_headers)
    qtbot.addWidget(panel)

    assert panel.headers == custom_headers


def test_cells_panel_initialization_with_control_panel(qtbot, labeled_cells_simple):
    """Test CellsPanel initialization with control panel."""
    control_panel = QtWidgets.QWidget()
    panel = CellsPanel(labeled_cells_simple, control_panel=control_panel)
    qtbot.addWidget(panel)

    assert panel.control_panel == control_panel


def test_cells_panel_load_data(cells_panel):
    """Test loading new data into CellsPanel."""
    # Create new labeled cells
    img = np.zeros((1, 3, 3), dtype=np.uint8)
    img[0, 0, 0] = 1
    img[0, 1, 1] = 1
    labeling = SkimageImgLabeling(SkimageCellConnectivity.EDGES)
    new_cells = LabeledCells(img, labeling)

    cells_panel.load_data(new_cells)

    assert cells_panel.img == new_cells
    assert cells_panel.cell_list.list.model.rowCount() == new_cells.len()


def test_cells_panel_cell_selection_handler(cells_panel, qtbot):
    """Test cell selection handling."""
    # Select first cell
    index = cells_panel.cell_list.list.model.index(0, 0)
    cells_panel.cell_list.list.table_view.selectRow(index.row())

    # Check if cell viewer is updated
    assert cells_panel.cell_viewer.img_viewer.image is not None


def test_control_panel_integration(qtbot, labeled_cells_simple):
    """Test integration with control panel."""
    control_panel = QtWidgets.QPushButton("Test Button")
    panel = CellsPanel(labeled_cells_simple, control_panel=control_panel)
    qtbot.addWidget(panel)

    # Check if control panel is properly integrated in layout
    assert panel.control_panel == control_panel

    # Find the control panel in the widget hierarchy
    left_widget = panel.layout().itemAt(0).widget()
    left_layout = left_widget.layout()
    assert left_layout.itemAt(1).widget() == control_panel
