from numpy.typing import NDArray
from pycroglia.core.slimskel3d.skel2graph import skel2graph
from pycroglia.core.slimskel3d.graph2skel import graph2skel

from skimage.morphology import skeletonize
def slimskel3d_scikit(vol: NDArray, threshold: int) -> NDArray:
    """Skeletonize and iteratively slim a 3D binary image.

    This function merge the behavior of the MATLAB `SlimSkel3D`, but uses `skimage.morphology.skeletonize` for skeletonization instead of a custom implementation. The rest of the process remains the same:
    it skeletonizes a binary volume, converts it into a graph
    representation, prunes spurious branches, and iterates until
    the network length stabilizes.

    Args:
        vol (np.ndarray):
            3D binary array representing the object (True/1 = foreground).
        threshold (int):
            Minimum branch length (in voxels). Branches shorter than `threshold`
            are removed during skeleton-to-graph conversion.

    Returns:
        np.ndarray:
            A 3D binary array of the slimmed skeleton, same shape as input.

    Notes:
        - Uses `skeletonize` from `skimage.morphology`.
        - Uses `skel2graph` to build graph representation (nodes + links).
        - Uses `graph2skel` to reconstruct skeleton from graph.
        - Iterates until the total skeleton link length no longer changes.
    """
    # Step 1: Skeletonization
    skel = skeletonize(vol)
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
