import pytest
import numpy as np

from pycroglia.ui.widgets.img_viewer import CustomImageViewer

@pytest.fixture
def custom_img_viewer(qtbot):
    """Fixture for CustomImageViewer widget."""
    widget = CustomImageViewer()
    qtbot.addWidget(widget)
    return widget

def test_set_image_sets_image(custom_img_viewer):
    """Test that set_image sets the image in the viewer."""
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    custom_img_viewer.set_image(img)

    assert custom_img_viewer.img_viewer.image.shape == img.shape

def test_set_lookup_table_sets_lut(custom_img_viewer):
    """Test that set_lookup_table sets the lookup table in the viewer."""
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)

    lut = np.zeros((256, 4), dtype=np.uint8)
    lut[:, :3] = np.arange(256).reshape(-1, 1)
    lut[:, 3] = 255
    custom_img_viewer.set_image(img)
    custom_img_viewer.set_lookup_table(lut)

    assert np.array_equal(
        custom_img_viewer.img_viewer.getImageItem().lut, lut
    )

