from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from skimage.measure import label as sklabel
from scipy.sparse import csr_matrix


@dataclass
class CenterOfMass:
    """Represents the geometric center of mass (mean voxel position) of a node.

    Attributes:
        x (float): Mean x-coordinate of all voxels belonging to the node.
        y (float): Mean y-coordinate of all voxels belonging to the node.
        z (float): Mean z-coordinate of all voxels belonging to the node.
    """

    x: float = 0
    y: float = 0
    z: float = 0


@dataclass
class Node:
    """Graph node corresponding to a cluster of voxels in a skeletonized volume.

    A node is created from one or more skeleton voxels that represent either
    a branching point or an endpoint in the skeleton.

    Attributes:
        id (int): Unique identifier of the node in the graph.
        indices (NDArray): 1D array of voxel indices (flattened) belonging to the node.
        center_of_mass (CenterOfMass): Average (x, y, z) location of node voxels.
        label (int): Connected-component label of the skeleton the node belongs to.
        links (list[Link]): List of Link objects connecting this node to others.
        connections (list[Node]): List of Node objects directly connected to this node.
        is_endpoint (bool): True if this node is an endpoint (degree 1), False otherwise.
    """

    id: int
    indices: NDArray
    center_of_mass: CenterOfMass
    label: int
    links: list["Link"] = field(default_factory=list)
    connections: list["Node"] = field(default_factory=list)
    is_endpoint: bool = False


@dataclass
class Link:
    """Graph edge corresponding to a sequence of canal voxels between two nodes.

    A link represents a continuous path through the skeleton, connecting
    two nodes (branch points or endpoints). It stores both voxel indices
    and references to its endpoints.

    Attributes:
        start (Node): Starting node of the link.
        end (Node): Ending node of the link.
        points (NDArray): 1D array of voxel indices (flattened) along the link path.
        label (int): Connected-component label of the skeleton the link belongs to.
    """

    start: Node
    end: Node
    points: NDArray
    label: int


def _follow_link(
    skel: np.ndarray,
    nodes: list[Node],
    node_idx: int,
    voxel_idx: int,
    start_voxel: int,
    canals: np.ndarray,
    canal_index_map: np.ndarray,
) -> tuple[NDArray, int, bool]:
    """Follow a canal voxel sequence until a node is reached.

    Args:
        skel (NDArray): Skeleton label volume (uint16).
        nodes (list[Node]): Node objects.
        node_idx (int): Index of the current node in `nodes`.
        voxel_idx (int): Index into `nodes[node_idx].indices`.
        start_voxel (int): Linear index of the starting canal voxel.
        canals (NDArray): Canal voxel array, shape (N, 3),
            with each row [canal, neighbor1, neighbor2].
        canal_index_map (NDArray): Map from voxel index → canal row index.

    Returns:
        tuple[np.ndarray, int, bool]:
            - voxels (np.ndarray): Linear indices of the followed voxel chain.
            - neighbor_node_idx (int): Index of the reached node.
            - is_endpoint (bool): Whether the reached node is an endpoint.
    """
    voxels: list[int] = []
    neighbor_node_idx = -1
    is_endpoint = False

    # assign start node voxel
    start_node_voxel = nodes[node_idx].indices[voxel_idx]
    voxels.append(int(start_node_voxel))

    idx = int(start_voxel)
    done = False
    while not done:
        # find canal row for current voxel
        next_cand = canal_index_map[idx]
        if next_cand == -1:
            # open-ended canal → endpoint spur
            is_endpoint = True
            break

        cand1, cand2 = int(canals[next_cand, 1]), int(canals[next_cand, 2])

        # Avoid self-reference or backtracking
        cand = cand1 if cand1 != voxels[-1] else cand2
        if cand == idx or cand == voxels[-1]:
            # Dead end or loop
            break

        if skel.ravel()[cand] > 1:  # node found
            voxels.append(idx)
            voxels.append(cand)
            neighbor_node_idx = (
                int(skel.ravel()[cand]) - 1
            )  # MATLAB 1-based → Python 0-based

            if nodes[neighbor_node_idx].is_endpoint:
                is_endpoint = True

            done = True
        else:
            voxels.append(idx)
            idx = cand

    if neighbor_node_idx < 0 or len(voxels) <= 2:
        return np.array([], dtype=int), -1, True

    return np.array(voxels, dtype=int), neighbor_node_idx, is_endpoint


# skeleton is the result of calling skeleton3D
def skel2graph(
    skeleton: NDArray, threshold: float = 0.0
) -> tuple[csr_matrix, list[Node], list[Link]]:
    """ "Convert a 3D skeletonized volume into a graph representation.

    This function implements the logic of the MATLAB function `Skel2Graph3D`,
    producing a network graph from a binary skeleton volume. The skeleton is
    represented as a set of `Node` objects (branching points and endpoints) and
    `Link` objects (paths connecting nodes). An adjacency matrix is also returned.

    Args:
        skeleton (NDArray):
            Binary 3D skeleton volume (z, y, x). Typically the output of
            `skeleton3D`. Foreground voxels (True/1) are treated as skeleton voxels.
        threshold (float, optional):
            Minimum length of a branch to be included. If `0`, short branches
            that terminate in endpoints are also kept. Default is `100.0`.

    Returns:
        adjacency (csr_matrix):
            Sparse adjacency matrix (n_nodes × n_nodes), where entry (i, j)
            contains the number of voxels along the link between nodes i and j.
        nodes (list[Node]):
            List of `Node` objects describing graph nodes. Each node contains:
              - `indices`: Linear voxel indices belonging to the node.
              - `center_of_mass`: Average (x, y, z) position of voxels.
              - `label`: Connected-component label from the original skeleton.
              - `links`: List of `Link` objects connected to this node.
              - `connections`: Neighboring `Node` objects.
              - `is_endpoint`: True if this node is a terminal endpoint.
        links (list[Link]):
            List of `Link` objects describing graph edges. Each link contains:
              - `start`: Starting `Node`.
              - `end`: Ending `Node`.
              - `points`: Linear voxel indices along the path.
              - `label`: Connected-component label from the original skeleton.

    Processing Steps:
        1. Pad the skeleton by 1 voxel to avoid boundary issues.
        2. Compute 26-neighborhoods for all skeleton voxels.
        3. Classify voxels as:
            - Nodes: >2 neighbors
            - Endpoints: exactly 1 neighbor
            - Canals: exactly 2 neighbors (forming links)
        4. Cluster adjacent node voxels into connected components (`Node`s).
        5. Follow canal voxels to connect nodes into `Link`s.
        6. Remove interior voxels of links to avoid re-traversal.
        7. Optionally discard links shorter than `threshold` unless they
           connect to endpoints and `threshold == 0`.
        8. Mark 1-degree nodes as endpoints.
        9. Build the sparse adjacency matrix.
        10. Convert voxel indices and centers of mass back to non-padded coordinates.

    Notes:
        - The voxel indexing is 0-based and follows NumPy’s (z, y, x) order.
        - Padding ensures correct neighborhood extraction near boundaries.
        - The graph is undirected: adjacency[i, j] = adjacency[j, i].
        - The function is designed to mirror the behavior of the original
          MATLAB implementation for reproducibility.
    """

    skel = np.pad(skeleton, pad_width=1, mode="constant", constant_values=0)
    skel2 = skel.astype(np.uint16, copy=True)
    lm = sklabel(skel, connectivity=3)  # lm = label matrix, num = number of components
    list_canal = np.flatnonzero(skel)

    # 26-neighborhood of all canal voxels (foreground voxels)
    neighbourhood = _get_neighbourhood(skel, list_canal).astype(bool)

    # 26-neighborhood indices of all canal voxels
    neighbourhood_indices = _get_neighbourhood_indices(skel, list_canal)

    # number of 26-neighbors of each skeleton voxel + center
    neighbourhood_sum = neighbourhood.sum(axis=1)

    # all canal voxels with >2 neighbors are nodes
    node_voxels = list_canal[neighbourhood_sum > 3]

    # all canal voxels with exactly one neighbor are endpoints
    endpoints = list_canal[neighbourhood_sum == 2]

    # all canal voxels with exactly 2 neighbors (canal voxels)
    canals = list_canal[neighbourhood_sum == 3]

    # Nx3 matrix with the 2 neighbors of each canal voxel
    canal_neighbourhood_indices = _get_neighbourhood_indices(skel, canals)
    canals_neighbourhood = _get_neighbourhood(skel, canals)

    # remove center of 3x3x3 cube (col=13 in Python 0-based)
    canal_neighbourhood_indices = np.delete(canal_neighbourhood_indices, 13, axis=1)
    canals_neighbourhood = np.delete(canals_neighbourhood, 13, axis=1)

    # keep only the two existing foreground voxels
    canals_nb = np.sort(
        canals_neighbourhood.astype(int) * canal_neighbourhood_indices, axis=1
    )

    # remove zeros: keep last two nonzero entries per row
    canals_nb = canals_nb[:, -2:]

    # add neighbors to canal voxel list (shape: (N, 3))
    canals = np.column_stack([canals, canals_nb])

    nodes: list[Node] = []
    links: list[Link] = []

    # group clusters of node voxels into nodes
    tmp = np.zeros(skel.shape, dtype=bool)
    if len(node_voxels) > 0:
        tmp[np.unravel_index(np.array(node_voxels, dtype=np.intp), skel.shape)] = True

    labeled_nodes, num_realnodes = sklabel(tmp, connectivity=3, return_num=True)

    for i in range(1, num_realnodes + 1):
        indices = np.flatnonzero(labeled_nodes == i)
        z, y, x = np.unravel_index(indices, skel.shape)
        node = Node(
            id=len(nodes),
            indices=indices,
            center_of_mass=CenterOfMass(
                x=float(np.mean(x)), y=float(np.mean(y)), z=float(np.mean(z))
            ),
            label=int(lm.ravel()[indices[0]]),
        )
        nodes.append(node)
        skel2.ravel()[indices] = len(nodes)

    # group endpoint voxels into nodes
    tmp = np.zeros(skel.shape, dtype=bool)
    if len(endpoints) > 0:
        tmp[np.unravel_index(np.array(endpoints, dtype=np.intp), skel.shape)] = True

    labeled_endpoints, num_endpoints = sklabel(tmp, connectivity=3, return_num=True)

    for i in range(1, num_endpoints + 1):
        indices = np.flatnonzero(labeled_endpoints == i)
        z, y, x = np.unravel_index(indices, skel.shape)
        node = Node(
            id=len(nodes),
            indices=indices,
            center_of_mass=CenterOfMass(
                x=float(np.mean(x)), y=float(np.mean(y)), z=float(np.mean(z))
            ),
            label=int(lm.ravel()[indices[0]]),
            is_endpoint=True,
        )
        nodes.append(node)
        skel2.ravel()[indices] = len(nodes)

    # Map: linear index of a canal voxel → row index in `canals`
    canal_index_map = np.full(skel.size, -1, dtype=int)
    if len(canals) > 0:
        canal_index_map[canals[:, 0]] = np.arange(len(canals), dtype=int)

    # Map: linear index of a skeleton voxel → row index in `neighbourhood_indices`
    skeleton_index_map = np.zeros(skel.size, dtype=int)
    skeleton_index_map[neighbourhood_indices[:, 13]] = np.arange(
        len(neighbourhood_indices), dtype=int
    )

    # Follow links between nodes
    for node in nodes:
        link_indices = skeleton_index_map[node.indices]

        for j, li in enumerate(link_indices):
            link_candidates = neighbourhood_indices[li, neighbourhood[li, :] == 1]
            endpoint_candidates = np.intersect1d(link_candidates, endpoints)

            link_candidates = link_candidates[skel2.ravel()[link_candidates] == 1]
            link_candidates = np.intersect1d(link_candidates, canals[:, 0])

            for candidate in link_candidates:
                voxels, target_node_idx, is_endpoint = _follow_link(
                    skel2, nodes, node.id, j, int(candidate), canals, canal_index_map
                )
                if len(voxels) == 0:
                    continue

                skel2.ravel()[voxels[1:-1]] = 0

                if (is_endpoint and len(voxels) > threshold) or (
                    not is_endpoint and node.id != target_node_idx
                ):
                    target_node = nodes[target_node_idx]
                    link = Link(
                        start=node,
                        end=target_node,
                        points=voxels,
                        label=int(lm.ravel()[voxels[0]]),
                    )
                    links.append(link)
                    node.links.append(link)
                    node.connections.append(target_node)
                    target_node.links.append(link)
                    target_node.connections.append(node)

            if threshold == 0:
                for ep_cand in endpoint_candidates:
                    val = int(skel2.ravel()[ep_cand])
                    if val > 1:  # must be a valid node
                        n_idx = val - 1
                        if n_idx < len(nodes) and n_idx != node.id:
                            skel2.ravel()[ep_cand] = 0
                            target_node = nodes[n_idx]
                            link = Link(
                                start=node,
                                end=target_node,
                                points=np.array([ep_cand]),
                                label=int(lm.ravel()[ep_cand]),
                            )
                            links.append(link)
                            node.links.append(link)
                            node.connections.append(target_node)
                            target_node.links.append(link)
                            target_node.connections.append(node)

    for node in nodes:
        if len(node.links) == 1:
            node.is_endpoint = True

    n_nodes = len(nodes)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, node in enumerate(nodes):
        for link in node.links:
            if link.start is node:
                j = link.end.id
            else:
                j = link.start.id
            adjacency[i, j] = len(link.points)
            adjacency[j, i] = len(link.points)
    adjacency = csr_matrix(adjacency)

    depth, height, width = skel.shape
    pad = 1
    unpad_shape = (depth - 2 * pad, height - 2 * pad, width - 2 * pad)

    for node in nodes:
        z, y, x = np.unravel_index(node.indices, skel.shape)
        node.indices = np.ravel_multi_index((z - pad, y - pad, x - pad), unpad_shape)
        node.center_of_mass.z -= pad
        node.center_of_mass.y -= pad
        node.center_of_mass.x -= pad

    for link in links:
        z, y, x = np.unravel_index(link.points, skel.shape)
        link.points = np.ravel_multi_index((z - pad, y - pad, x - pad), unpad_shape)

    return adjacency, nodes, links


def _get_neighbourhood(img: NDArray, indices: NDArray) -> NDArray:
    """Return the 3x3x3 neighborhood of given voxels in a 3D binary image.
    This function mimics the MATLAB `pk_get_nh` behavior, collecting the values
    of all 27 neighbors (including the voxel itself) around each input voxel
    index. Out-of-bounds neighbors are treated as `False`.
    Args:
        img (NDArray):
            A 3D binary image (bool or int).
        indices (NDArray):
            Linear indices (0-based, flattened order, i.e. as from `img.ravel()`).
    Returns:
        NDArray:
            A boolean array of shape ``(len(indices), 27)``, where each row
            corresponds to the 27-neighborhood of a voxel in row-major order.
    """
    shape = img.shape  # (z, y, x)
    z, y, x = np.unravel_index(indices, shape)

    neighbourhood = np.zeros((len(indices), 27), dtype=bool)
    w = 0
    for dz in range(3):
        for dy in range(3):
            for dx in range(3):
                zn = z + dz - 1
                yn = y + dy - 1
                xn = x + dx - 1
                inside = (
                    (zn >= 0)
                    & (zn < shape[0])
                    & (yn >= 0)
                    & (yn < shape[1])
                    & (xn >= 0)
                    & (xn < shape[2])
                )
                vals = np.zeros(len(indices), dtype=bool)
                if np.any(inside):
                    vals[inside] = img[zn[inside], yn[inside], xn[inside]]
                neighbourhood[:, w] = vals
                w += 1
    return neighbourhood


def _get_neighbourhood_indices(img: NDArray, indices: NDArray) -> NDArray:
    """Return linear indices of the 3x3x3 neighborhood around given voxels.

    This is the Python equivalent of the MATLAB function `pk_get_nh_idx`.

    Args:
        img (NDArray):
            Input 3D image, shape (z, y, x).
        indices (NDArray):
            1D array of linear indices (0-based) of voxels in `img`.

    Returns:
        NDArray:
            Array of shape (len(indices), 27) with the linear indices of each
            voxel’s 3x3x3 neighborhood, in row-major order.
    """
    shape = img.shape  # (z, y, x)
    z, y, x = np.unravel_index(indices, shape)

    neighbourhood = np.zeros((len(indices), 27), dtype=int)

    w = 0
    for dz in range(3):
        for dy in range(3):
            for dx in range(3):
                zn = z + dz - 1
                yn = y + dy - 1
                xn = x + dx - 1
                # convert (zn,yn,xn) → linear index
                neighbourhood[:, w] = np.ravel_multi_index(
                    (zn, yn, xn), shape, mode="clip"
                )
                w += 1

    return neighbourhood
