from numpy.typing import NDArray
from pycroglia.core.slimskel3d.skeleton3D import skeleton3D
from pycroglia.core.slimskel3d.skel2graph import skel2graph
from pycroglia.core.slimskel3d.graph2skel import graph2skel


def slimskel3d(vol: NDArray, threshold: int) -> NDArray:
    """Skeletonize and iteratively slim a 3D binary image.

    This function mirrors the behavior of the MATLAB `SlimSkel3D`:
    it skeletonizes a binary volume, converts it into a graph
    representation, prunes spurious branches, and iterates until
    the network length stabilizes.

    Args:
        cell_to_skel (np.ndarray):
            3D binary array representing the object (True/1 = foreground).
        thr (int):
            Minimum branch length (in voxels). Branches shorter than `thr`
            are removed during skeleton-to-graph conversion.

    Returns:
        np.ndarray:
            A 3D binary array of the slimmed skeleton, same shape as input.

    Notes:
        - Uses `skeletonize_3d` (equivalent to MATLAB `Skeleton3D`).
        - Uses `skel2graph` to build graph representation (nodes + links).
        - Uses `graph2skel` to reconstruct skeleton from graph.
        - Iterates until the total skeleton link length no longer changes.

    Example:
        >>> import numpy as np
        >>> from pycroglia.core.slimSkel3D.slim_skel import slim_skel_3d
        >>> vol = np.zeros((30, 30, 30), dtype=ool)
        >>> vol[5:25, 15, 15] = 1   # simple line
        >>> slim = slim_skel_3d(vol, thr=5)
        >>> slim.sum()  # number of skeleton voxels
        20
    """
    # Step 1: Skeletonization
    skel = skeleton3D(vol)
    # Step 2: Convert to graph
    _, nodes, links = skel2graph(skel, threshold=threshold)
    wl = sum(len(node.links) for node in nodes)  # total link length
    # Step 3: Reconstruct skeleton
    slim_skel = graph2skel(nodes, links, skel.shape)
    # Step 4: Recompute graph
    _, nodes2, links2 = skel2graph(slim_skel, threshold=0)
    wl_new = sum(len(node.links) for node in nodes2)

    # Step 5: Iterate until stable
    while wl != wl_new:
        wl = wl_new
        slim_skel = graph2skel(nodes2, links2, skel.shape)
        _, nodes2, links2 = skel2graph(slim_skel, threshold=0)
        wl_new = sum(len(node.links) for node in nodes2)
    return slim_skel
