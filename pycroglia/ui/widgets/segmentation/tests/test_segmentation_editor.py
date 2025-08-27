import pytest
import numpy as np

from pycroglia.ui.widgets.segmentation.segmentation_editor import SegmentationEditor
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
def segmentation_editor(qtbot, labeled_cells_simple):
    """Fixture for SegmentationEditor widget."""
    img = labeled_cells_simple.labels > 0
    labeling_strategy = SkimageImgLabeling(SkimageCellConnectivity.EDGES)
    widget = SegmentationEditor(img, labeling_strategy, min_size=1)
    qtbot.addWidget(widget)
    return widget


def test_load_data_populates_list(segmentation_editor):
    """Test that _load_data populates the cell list."""
    segmentation_editor._load_data()
    assert segmentation_editor.list.list.model.rowCount() > 0


def test_on_cell_selection_enables_segment_button(segmentation_editor, qtbot):
    """Test that selecting a cell enables the segment button."""
    segmentation_editor._load_data()

    index = segmentation_editor.list.list.model.index(0, 0)
    segmentation_editor.list.list.table_view.selectRow(index.row())
    segmentation_editor._on_cell_selection_changed()

    assert segmentation_editor.segment_button.isEnabled()


def test_on_cell_selection_updates_cell_viewer(segmentation_editor, qtbot):
    """Test that selecting a cell updates the cell viewer image."""
    segmentation_editor._load_data()

    index = segmentation_editor.list.list.model.index(0, 0)
    segmentation_editor.list.list.table_view.selectRow(index.row())
    segmentation_editor._on_cell_selection_changed()

    assert segmentation_editor.cell_viewer.img_viewer.image is not None


def test_on_rollback_request_restores_state(segmentation_editor):
    """Test that rollback restores the previous segmentation state."""
    segmentation_editor._load_data()
    # Simulate a previous state
    segmentation_editor.state._prev_state = segmentation_editor.state._actual_state
    segmentation_editor._on_rollback_request()
    assert segmentation_editor.state._prev_state is None
