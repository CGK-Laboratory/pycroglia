import numpy as np
from numpy.typing import NDArray
from pycroglia.core.slimskel3d.skel2graph import Node, Link


def graph2skel(
    nodes: list[Node], links: list[Link], shape: tuple[int, int, int]
) -> NDArray:
    """Reconstruct a 3D skeleton (binary mask) from node/link
    graph. It's not an exact recreation of the original skeleton

    Args:
        node (list[dict]): List of node dicts with keys:
            - "idx": list/array of voxel indices (flattened)
            - "links": list of link indices connected to this node
        link (list[dict]): List of link dicts with key:
            - "point": list/array of voxel indices (flattened) for edges
        w, l, h (int): Dimensions of the output volume (z, y, x)

    Returns:
        np.ndarray: 3D boolean skeleton array.

    """
    skel = np.zeros(shape, dtype=np.uint8)

    for node in nodes:
        if node.links:  # node has links
            # Paint node voxels
            skel.ravel()[node.indices] = 1

            # Paint link voxels
            for link in node.links:
                skel.ravel()[link.points] = 1

    return skel.astype(np.uint8)
