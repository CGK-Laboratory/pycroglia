import numpy as np
import scipy.sparse as sparse
from pathlib import Path
from pycroglia.core.slimskel3d.skel2graph import skel2graph

TEST_DIR = Path(__file__).parent  # folder where this test lives
FILES_DIR = TEST_DIR / "files"  # adjust if files/ is elsewhere


# TODO(jab227): Make this test more robust
def test_skel2graph():
    """Test Skel2Graph3D correctly converts a skeleton volume to a graph.

    Asserts:
        - The number of detected nodes equals 88.
        - The number of detected links equals 97.
        - The computed adjacency matrix matches the expected adjacency
          matrix (loaded from file) element-wise.
    """
    with np.load(FILES_DIR / "skel_test.npz") as data:
        skel = data["arr_0"]
    expected_adjacency_matrix = sparse.load_npz(FILES_DIR / "adjacency_test.npz")

    got, nodes, links = skel2graph(skel, threshold=0)
    assert len(nodes) == 88
    assert len(links) == 97
    np.testing.assert_allclose(got.toarray(), expected_adjacency_matrix.toarray())
