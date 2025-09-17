import numpy as np
from pathlib import Path
from pycroglia.core.slimskel3d import graph2skel
from pycroglia.core.slimskel3d.skel2graph import skel2graph

TEST_DIR = Path(__file__).parent  # folder where this test lives
FILES_DIR = TEST_DIR / "files"  # adjust if files/ is elsewhere


def test_graph2skel():
    """Test Graph2Skel3D correctly reconstructs skeleton from graph.

    Asserts:
        - Converting a skeleton volume to graph (`skel2graph`) and back to
          a voxel skeleton (`graph2skel`) matches the original volume
          except for 9 known mismatched voxels.
        - The number of mismatches is exactly 9.
        - The mismatched voxel coordinates match the expected set.
    """
    with np.load(FILES_DIR / "skel_test.npz") as data:
        expected = data["arr_0"].astype(np.uint8)

    _, nodes, links = skel2graph(expected, threshold=0)

    got = graph2skel.graph2skel(nodes, links, expected.shape)
    mismatches = np.argwhere(got != expected)
    assert len(mismatches == 9)
    expected_mismatches = np.array(
        [
            [21, 5, 4],
            [21, 5, 6],
            [22, 4, 7],
            [22, 6, 4],
            [22, 7, 4],
            [23, 5, 6],
            [23, 8, 4],
            [24, 6, 5],
            [24, 7, 4],
        ],
        dtype=int,
    )
    np.testing.assert_equal(mismatches, expected_mismatches)
