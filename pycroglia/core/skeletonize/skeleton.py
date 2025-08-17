import numpy as np
from scipy.ndimage import binary_dilation
from .msfm import msfm2d
from .shortest_path import ShortestPath
from scipy.sparse import lil_matrix


def _get_boundary_distance(image: np.ndarray, is3d: bool) -> np.ndarray:
    """
    Calculates the distance from all foreground voxels/pixels to the boundary of a binary object.

    Args:
        image (np.ndarray): Binary image or volume (2D or 3D).
        is3d (bool): True if input is 3D, False if 2D.

    Returns:
        np.ndarray: Distance map from boundary voxels/pixels.
    """
    structure = np.ones((3, 3, 3) if is3d else (3, 3), dtype=bool)
    B = np.logical_xor(image, binary_dilation(image, structure=structure))

    source_indices = np.argwhere(B).T  # shape: (2, N) or (3, N)

    speed_image = np.ones_like(image, dtype=np.float64)
    boundary_distance = msfm2d(
        speed_image, source_indices, use_second=False, use_cross=True
    )
    if isinstance(boundary_distance, np.ndarray):
        boundary_distance[np.logical_not(image)] = 0
        return boundary_distance
    else:
        assert False, "Shouldn't happen"


def _max_distance_point(
    boundary_distance: np.ndarray, image: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Finds the coordinates of the maximum distance value in the masked volume.

    Args:
        boundary_distance (np.ndarray): Distance map.
    Returns:
        tuple[np.ndarray, float]: Coordinates of the maximum distance point and its value.
    """
    boundary_distance[np.logical_not(image)] = 0
    max_d = np.max(boundary_distance)
    if not np.isfinite(max_d):
        raise ValueError("Maximum from MSFM is infinite!")

    ind = np.argmax(boundary_distance)
    coords = np.unravel_index(ind, boundary_distance.shape)
    pos_d = np.array(coords).reshape((-1, 1))  # shape: (2, 1) or (3, 1)

    return pos_d, max_d


def _get_line_length(L: np.ndarray, is_3d: bool) -> np.float64:
    """
    Computes the total length of a polyline in 2D or 3D.

    Args:
        L (np.ndarray): An (N, 2) or (N, 3) array of line points.
        is_3d (bool): Whether the line is in 3D (True) or 2D (False).

    Returns:
        float: Total line length.
    """
    diff = np.diff(L, axis=0)
    if is_3d:
        dist = np.sqrt(np.sum(diff[:, :3] ** 2, axis=1))
    else:
        dist = np.sqrt(np.sum(diff[:, :2] ** 2, axis=1))
    return np.sum(dist)


def _organize_skeleton(
    skeleton_segments: list[np.ndarray], is_3d: bool
) -> list[np.ndarray]:
    """
    Breaks skeleton segments into subsegments based on proximity of endpoints.

    Args:
        skeleton_segments (List[np.ndarray]): List of skeleton segments (each (N, 2) or (N, 3)).
        is_3d (bool): Whether the segments are in 3D (True) or 2D (False).

    Returns:
        List[np.ndarray]: List of broken segments.
    """
    length = len(skeleton_segments)
    dims = 3 if is_3d else 2
    endpoints = np.zeros((length * 2, dims))
    max_len = 1

    for w, segment in enumerate(skeleton_segments):
        max_len = max(max_len, len(segment))
        endpoints[2 * w] = segment[0]
        endpoints[2 * w + 1] = segment[-1]

    cut_skel = lil_matrix((len(endpoints), max_len), dtype=bool)
    connect_distance_sq = 4  # squared distance threshold

    for w, segment in enumerate(skeleton_segments):
        ep_t = endpoints[:, None, :]  # shape: (2n, 1, D)
        ss_exp = segment[None, :, :]  # shape: (1, L, D)

        dists_sq = np.sum((ep_t - ss_exp) ** 2, axis=2)  # shape: (2n, L)

        check = np.min(dists_sq, axis=1) < connect_distance_sq
        check[2 * w] = False
        check[2 * w + 1] = False

        if np.any(check):
            j_indices = np.flatnonzero(check)
            for j in j_indices:
                line = dists_sq[j]
                k = np.argmin(line)
                if 2 < k < len(line) - 2:
                    cut_skel[w, k] = True

    cell_array = []
    for w, segment in enumerate(skeleton_segments):
        cut_row = cut_skel.getrow(w).toarray().ravel()
        cut_indices = np.flatnonzero(cut_row)
        r = np.concatenate(([0], cut_indices, [len(segment) - 1]))
        for i in range(len(r) - 1):
            cell_array.append(segment[r[i] : r[i + 1] + 1])

    return cell_array


def skeleton(image: np.ndarray) -> list[np.ndarray]:
    """
     Computes the skeleton (centerlines) of a 2D or 3D binary object using
    the Multistencil Fast Marching (MSFM) distance transform.

    This function returns subvoxel-accurate centerlines of the object by
    repeatedly tracing shortest paths from medial points to the object's boundary.

    Args:
        binary_image (np.ndarray): A 2D or 3D binary image or volume representing the object.
        verbose (bool, optional): If True, prints debugging information. Defaults to True.

    Returns:
        list[np.ndarray]: A list of (N x D) arrays, each representing one skeleton branch.
                          D is 2 for 2D input and 3 for 3D input.
    """
    assert image.ndim in (2, 3), "Image should be 2D or 3D"
    is_3d = image.ndim == 3
    boundary_distance = _get_boundary_distance(image, is_3d)
    source_point, max_distance = _max_distance_point(boundary_distance, image)
    # Make a fast marching speed image from the distance image
    speed_image = (boundary_distance / max_distance) ** 4
    speed_image[speed_image == 0] = 1e-10  # Avoid zero speed (non-traversable)

    # Initialize list for skeleton segments (preallocated to 1000 entries)
    skeleton_segments = []

    while True:
        # Do fast marching using the maximum distance value in the image
        # and the points describing all found branches are sourcepoints.
        output_distance_image, euclidean_distance_image = msfm2d(
            speed_image,
            source_point,
            use_second=False,
            use_cross=False,
            skeletonize=True,
        )
        # Trace a branch back to the used sourcepoints
        start_point, _ = _max_distance_point(euclidean_distance_image, image)
        shortest_path = ShortestPath(step_size=1.0)
        shortest_line, _ = shortest_path.calculate(
            output_distance_image, start_point, source_point
        )
        # Calculate the length of the new skeleton segment
        line_length = _get_line_length(shortest_line, is_3d)
        # Stop finding branches, if the lenght of the new branch is smaller
        # then the diameter of the largest vessel
        if line_length < max_distance * 2:
            break
        # Store the found branch skeleton
        skeleton_segments.append(shortest_line)
        # Add found branch to the list of fast marching source points
        source_point = np.hstack([source_point, shortest_line.T])
    return _organize_skeleton(skeleton_segments, is_3d)
