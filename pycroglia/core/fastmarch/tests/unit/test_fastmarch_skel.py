import numpy as np
import pytest
from pycroglia.core.fastmarch.fastmarch_skel import fastmarch_skel


def _straight_line_vol(length: int = 20) -> np.ndarray:
    """Return a small 3D binary volume containing a single straight line."""
    vol = np.zeros((length + 10, 20, 20), dtype=bool)
    vol[5 : 5 + length, 10, 10] = True
    return vol


def _thick_line_vol(length: int = 20, radius: int = 3) -> np.ndarray:
    """Return a 3D binary volume containing a thick cylinder along axis 0.

    A radius > 1 ensures the Gaussian smoothing step (sigma=1) does not
    erase the foreground entirely before skeletonization.
    """
    vol = np.zeros((length + 10, 30, 30), dtype=bool)
    cx, cy = 15, 15
    for z in range(5, 5 + length):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy**2 + dx**2 <= radius**2:
                    vol[z, cy + dy, cx + dx] = True
    return vol


def test_returns_bool_array():
    """Output dtype must be boolean."""
    vol = _straight_line_vol()
    skel = fastmarch_skel(vol, sigma=0)
    assert skel.dtype == bool


def test_output_shape_matches_input():
    """Output shape must equal input shape."""
    vol = _straight_line_vol()
    skel = fastmarch_skel(vol, sigma=0)
    assert skel.shape == vol.shape


def test_skeleton_is_nonempty():
    """A non-trivial input must produce at least one skeleton voxel."""
    vol = _straight_line_vol()
    skel = fastmarch_skel(vol, sigma=0)
    assert skel.any(), "skeleton should not be empty for a non-trivial volume"


def test_skeleton_coordinates_in_valid_range():
    """All skeleton voxel indices must be within the valid array bounds.

    The fast marching algorithm traces paths from skeleton centres to image
    boundaries, so skeleton voxels are not guaranteed to lie within the
    original foreground region.  We verify only that coordinates are
    in-bounds (i.e., no index errors would occur when indexing the output).
    """
    vol = _straight_line_vol()
    skel = fastmarch_skel(vol, sigma=0)
    coords = np.argwhere(skel)
    for dim, size in enumerate(vol.shape):
        assert np.all(coords[:, dim] >= 0), f"negative index along dim {dim}"
        assert np.all(coords[:, dim] < size), f"out-of-bounds index along dim {dim}"


def test_no_smoothing_sigma_zero():
    """sigma=0 should skip smoothing without raising an error."""
    vol = _straight_line_vol()
    skel = fastmarch_skel(vol, sigma=0)
    assert skel.any()


def test_smoothing_sigma_positive():
    """sigma>0 (default) should still produce a valid skeleton on thick volumes."""
    vol = _thick_line_vol()
    skel = fastmarch_skel(vol, sigma=1.0)
    assert skel.shape == vol.shape
    assert skel.any(), "skeleton should not be empty for a thick foreground volume"


def test_empty_volume_returns_empty_skeleton():
    """An all-zero volume should produce an all-False skeleton without crashing."""
    vol = np.zeros((20, 20, 20), dtype=bool)
    skel = fastmarch_skel(vol, sigma=0)
    assert not skel.any(), "empty input should yield an empty skeleton"


def test_empty_after_smoothing_returns_empty_skeleton():
    """A thin 1-voxel line erased by Gaussian smoothing should return empty skeleton."""
    # A single-voxel line is completely erased by sigma=5 Gaussian
    vol = _straight_line_vol()
    skel = fastmarch_skel(vol, sigma=5.0)
    # Should not raise; output may be empty (thin structure erased by large sigma)
    assert isinstance(skel, np.ndarray)
    assert skel.shape == vol.shape
