import numpy as np
import scipy.sparse as sparse

from pycroglia.core.slimSkel3D.skel2graph import skel2graph


# TODO(jab227): Make this test more robust
def test_skel2graph():
    """Test Skel2Graph3D correctly converts a skeleton volume to a graph.

    Asserts:
        - The number of detected nodes equals 88.
        - The number of detected links equals 97.
        - The computed adjacency matrix matches the expected adjacency
          matrix (loaded from file) element-wise.
    """
    with np.load("./files/skel_test.npz") as data:
        skel = data["arr_0"]
    expected_adjacency_matrix = sparse.load_npz("./files/adjacency_test.npz")

    got, nodes, links = skel2graph(skel, threshold=0)
    assert len(nodes) == 88
    assert len(links) == 97
    np.testing.assert_allclose(got.toarray(), expected_adjacency_matrix.toarray())
