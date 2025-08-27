import numpy as np
from scipy.ndimage import binary_dilation
from pycroglia.core.skeletonize.raytracing import factory
from pycroglia.core.skeletonize.msfm import msfm2d
from pycroglia.core.skeletonize.shortest_path import ShortestPath
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

    source_indices = np.argwhere(B).astype(np.int64)  # shape: (2, N) or (3, N)

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
    boundary_distance: np.ndarray, image: np.ndarray, is_3d: bool
) -> tuple[np.ndarray, float]:
    """
    Finds the coordinates of the maximum distance value in the masked volume.

    Args:
        boundary_distance (np.ndarray): Distance map.
        image (np.ndarray): Binary image mask.
        is_3d (bool): Whether the image is 3D.

    Returns:
        tuple[np.ndarray, float]: Coordinates of the maximum distance point and its value.
    """
    # Create a copy to avoid modifying the original
    masked_distance = boundary_distance.copy()
    masked_distance[np.logical_not(image)] = 0

    max_d = np.max(masked_distance)
    if not np.isfinite(max_d):
        raise ValueError("Maximum from MSFM is infinite!")

    ind = np.argmax(masked_distance)
    pos_d = np.array(np.unravel_index(ind, image.shape))

    # Return coordinates as row vector for consistency with msfm2d expectations
    return pos_d.reshape(1, -1), max_d


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

    This matches the MATLAB OrganizeSkeleton function behavior.

    Args:
        skeleton_segments (List[np.ndarray]): List of skeleton segments (each (N, 2) or (N, 3)).
        is_3d (bool): Whether the segments are in 3D (True) or 2D (False).

    Returns:
        List[np.ndarray]: List of broken segments.
    """
    n = len(skeleton_segments)
    if n == 0:
        return []

    dims = 3 if is_3d else 2
    endpoints = np.zeros((n * 2, dims))
    max_len = 1

    # Build endpoints array - using MATLAB indexing convention
    for w, segment in enumerate(skeleton_segments):
        max_len = max(max_len, len(segment))
        endpoints[w * 2, :] = segment[0, :]  # w*2-1 in MATLAB (0-indexed here)
        endpoints[w * 2 + 1, :] = segment[-1, :]  # w*2 in MATLAB (1-indexed here)

    cut_skel = lil_matrix((n, max_len), dtype=bool)
    connect_distance_sq = 4  # 2^2 to match MATLAB ConnectDistance

    for w, ss in enumerate(skeleton_segments):
        ex = np.tile(endpoints[:, 0][:, None], (1, ss.shape[0]))
        sx = np.tile(ss[:, 0][None, :], (endpoints.shape[0], 1))
        ey = np.tile(endpoints[:, 1][:, None], (1, ss.shape[0]))
        sy = np.tile(ss[:, 1][None, :], (endpoints.shape[0], 1))
        if is_3d:
            ez = np.tile(endpoints[:, 2][:, None], (1, ss.shape[0]))
            sz = np.tile(ss[:, 2][None, :], (endpoints.shape[0], 1))
            D = (ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2
        else:
            D = (ex - sx) ** 2 + (ey - sy) ** 2

        # Find endpoints that are close to this segment
        check = np.min(D, axis=1) < connect_distance_sq
        # Exclude endpoints of current segment (MATLAB: check(w*2-1)=false; check(w*2)=false;)
        check[w * 2] = False
        check[w * 2 + 1] = False

        if np.any(check):
            j = np.where(check)[0]
            for jj in j:
                line = D[jj, :]
                k = np.argmin(line)
                # Only cut if not too close to endpoints (MATLAB: k>2 && k<length(line)-2)
                if 1 < k < (len(line) - 1):  # Adjusted for 0-indexing
                    cut_skel[w, k] = True

    # Build final segments
    cell_array = []
    for w, ss in enumerate(skeleton_segments):
        # Get cut points (MATLAB: r=[1 find(CutSkel(w,:)) length(ss)])
        cut_points = np.where(cut_skel[w, :].toarray()[0])[0]
        r = [0] + list(cut_points) + [len(ss) - 1]

        for i in range(len(r) - 1):
            start_idx = r[i]
            end_idx = r[i + 1]
            if start_idx <= end_idx:  # Ensure valid range
                cell_array.append(ss[start_idx : end_idx + 1, :])

    return cell_array


def skeleton(
    image: np.ndarray, stepper_type: factory.StepperType = factory.StepperType.RK4
) -> list[np.ndarray]:
    """
    Computes the skeleton (centerlines) of a 2D or 3D binary object using
    the Multistencil Fast Marching (MSFM) distance transform.

    This function returns subvoxel-accurate centerlines of the object by
    repeatedly tracing shortest paths from medial points to the object's boundary.

    Args:
        binary_image (np.ndarray): A 2D or 3D binary image or volume representing the object.
        stepper_type (factory.StepperType): Algorithm used for raytracing.
    Returns:
        list[np.ndarray]: A list of (N x D) arrays, each representing one skeleton branch.
                          D is 2 for 2D input and 3 for 3D input.
    """
    raise NotImplementedError
    assert image.ndim in (2, 3), "Image should be 2D or 3D"
    is_3d = image.ndim == 3
    boundary_distance = _get_boundary_distance(image, is_3d)
    source_point, max_distance = _max_distance_point(boundary_distance, image, is_3d)
    # Make a fast marching speed image from the distance image
    speed_image = (boundary_distance / max_distance) ** 4
    speed_image[speed_image == 0] = 1e-10  # Avoid zero speed (non-traversable)

    # Initialize list for skeleton segments (preallocated to 1000 entries)
    skeleton_segments = []
    shortest_path = ShortestPath(step_size=1.0, stepper_type=stepper_type)
    iteration = 0

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
        start_point, _ = _max_distance_point(euclidean_distance_image, image, is_3d)

        # Calculate shortest path (match MATLAB: shortestpath(T,StartPoint,SourcePoint,1,'rk4'))
        shortest_line, _ = shortest_path.calculate(
            output_distance_image, start_point.flatten(), source_point
        )

        # Calculate the length of the new skeleton segment
        line_length = _get_line_length(shortest_line, is_3d)

        # Stop finding branches, if the length of the new branch is smaller
        # than the diameter of the largest vessel
        if line_length < max_distance * 2:
            break

        # Store the found branch skeleton
        skeleton_segments.append(shortest_line)
        iteration += 1

        # Add found branch to the list of fast marching source points
        # Match MATLAB: SourcePoint=[SourcePoint ShortestLine'];
        # Convert shortest_line to source points format (N, 2)
        if source_point.size == 0:
            source_point = shortest_line.astype(int)
        else:
            source_point = np.vstack([source_point, shortest_line.astype(int)])

    # Organize skeleton segments (matches MATLAB OrganizeSkeleton)
    result = _organize_skeleton(skeleton_segments, is_3d)

    return result
