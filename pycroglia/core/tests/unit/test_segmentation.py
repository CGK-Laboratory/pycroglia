import numpy as np

from numpy.typing import NDArray
from pycroglia.core.segmentation import (
    segment_cell,
    segment_single_cell,
    SegmentationConfig,
)
from pycroglia.core.erosion import Rectangle2DFootprint, Rectangle3DFootprint
from pycroglia.core.labeled_cells import (
    LabeledCells,
    SkimageImgLabeling,
)
from pycroglia.core.enums import SkimageCellConnectivity


def simple_cells_img() -> NDArray:
    """Returns a 3D binary image with two separate cells.

    Returns:
        NDArray: 3D binary image.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[0, 0, 0] = 1
    img[2, 2, 2] = 1
    return img


def test_segmentation_config_defaults():
    """Test SegmentationConfig default values.

    Asserts:
        The default values are set as expected.
    """
    config = SegmentationConfig(
        cut_off_size=10, noise=2, connectivity=SkimageCellConnectivity.FACES
    )
    assert config.cut_off_size == 10
    assert config.noise == 2
    assert (
        config.min_nucleus_fraction == SegmentationConfig.DEFAULT_MIN_NUCLEUS_FRACTION
    )
    assert config.gmm_n_init == SegmentationConfig.DEFAULT_GMM_N_INIT
    assert config.connectivity == SkimageCellConnectivity.FACES


def test_segment_cell_basic():
    """Test segment_cell returns correct number of segments for simple input.

    Asserts:
        The number of returned segments matches the number of cells.
    """
    img = simple_cells_img()
    cells = LabeledCells(img, SkimageImgLabeling(SkimageCellConnectivity.FACES))
    config = SegmentationConfig(
        cut_off_size=1, noise=1, connectivity=SkimageCellConnectivity.FACES
    )
    footprint = Rectangle2DFootprint(x=1, y=1)
    segments = segment_cell(cells, footprint, config)
    assert isinstance(segments, list)
    assert len(segments) == 2
    for seg in segments:
        assert seg.shape == img.shape
        assert np.sum(seg) == 1
        assert set(np.unique(seg)).issubset({0, 1})


def test_segment_cell_small_cell_not_split():
    """Test segment_cell does not split small cells below cut_off_size.

    Asserts:
        The returned segment is the same as the input cell.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[1, 1, 1] = 1
    cells = LabeledCells(img, SkimageImgLabeling(SkimageCellConnectivity.FACES))
    config = SegmentationConfig(
        cut_off_size=2, noise=1, connectivity=SkimageCellConnectivity.FACES
    )
    footprint = Rectangle2DFootprint(x=1, y=1)
    segments = segment_cell(cells, footprint, config)
    assert len(segments) == 1
    assert np.sum(segments[0]) == 1
    assert segments[0][1, 1, 1] == 1


def test_segment_single_cell_split():
    """Test segment_single_cell splits a large cell into multiple segments.

    Asserts:
        The number of returned segments matches the number of nuclei.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[0, 0, 0] = 1
    img[2, 2, 2] = 1
    img[1, 1, 1] = 1
    config = SegmentationConfig(
        cut_off_size=1, noise=1, connectivity=SkimageCellConnectivity.FACES
    )
    # Use a minimal footprint to avoid eroding away the nuclei
    footprint = Rectangle3DFootprint(x=0, y=0, z=0)
    segments = segment_single_cell(img, footprint, config)

    assert isinstance(segments, list)
    assert len(segments) == 3
    for seg in segments:
        assert seg.shape == img.shape
        assert np.sum(seg) == 1
        assert set(np.unique(seg)).issubset({0, 1})
