import pytest
import numpy as np

from pycroglia.ui.widgets.segmentation.multi_cell_img_viewer import MultiCellImageViewer
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
def multi_cell_img_viewer(qtbot):
    """Fixture for MultiCellImageViewer widget."""
    widget = MultiCellImageViewer()
    qtbot.addWidget(widget)
    return widget


def test_set_cells_img(multi_cell_img_viewer, labeled_cells_simple):
    """Test that set_cells_img sets the image and lookup table correctly."""
    multi_cell_img_viewer.set_cells_img(labeled_cells_simple)

    label_2d = labeled_cells_simple.labels_to_2d()
    assert multi_cell_img_viewer.img_viewer.img_viewer.image.shape == label_2d.shape

    n_cells = labeled_cells_simple.len()
    assert (
        multi_cell_img_viewer.img_viewer.img_viewer.imageItem.lut.shape[0]
        == n_cells + 1
    )
