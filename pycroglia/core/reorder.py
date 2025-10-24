import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


def reorder_pixel_list(
    pixel_indices: NDArray,
    shape: tuple[int, int, int],
    endpoint: NDArray,
    centroid: NDArray,
) -> NDArray:
    """
    Order the voxels of a 3D branch mask from the endpoint toward the centroid.

    This function takes a list of flattened voxel indices belonging to a single
    3D branch (a connected set of True voxels) and reconstructs an ordered path
    from a known endpoint toward a given centroid. The ordering is computed by
    iteratively selecting the nearest unvisited voxel to the current one,
    ensuring spatial continuity along the branch.

    The resulting list starts with the endpoint voxel and progresses through
    adjacent voxels until reaching (or approximating) the centroid region.

    Args:
        pixel_indices (NDArray):
            1D array of flattened voxel indices (0-based) representing the branch.
        shape (tuple[int, int, int]):
            Shape of the 3D binary mask `(Z, Y, X)` from which the indices were drawn.
        endpoint (NDArray):
            Array of shape `(3,)` giving the endpoint coordinates `(z, y, x)`.
            Must be one of the voxels in the branch.
        centroid (NDArray):
            Array of shape `(3,)` giving the centroid coordinates `(z, y, x)`
            used as the stopping target for ordering.

    Returns:
        NDArray:
            Array of shape `(N, 3)` containing voxel coordinates ordered by
            connectivity from endpoint to centroid. The first row equals
            `endpoint`, and the last row corresponds to the voxel nearest
            to `centroid`.

    Raises:
        AssertionError:
            If input shapes are invalid, or if the endpoint voxel is not
            present in the given branch.
    """
    assert len(shape) == 3, f"Expected shape of length 3, got {shape}"
    assert pixel_indices.ndim == 1, "pixel_indices must be a 1D array"
    assert np.array(endpoint).shape == (3,), "endpoint must have shape (3,)"
    assert np.array(centroid).shape == (3,), "centroid must have shape (3,)"

    coords = np.column_stack(np.unravel_index(pixel_indices, shape)).astype(int)
    if coords.shape[0] == 0:
        return coords

    ep_mask = np.all(coords == endpoint, axis=1)
    assert np.any(ep_mask), f"Endpoint {endpoint.tolist()} not found in pixel list"
    ep_index = np.argmax(ep_mask)

    coords[[0, ep_index]] = coords[[ep_index, 0]]

    i = 0
    while i < len(coords) - 1 and not np.all(coords[i] == centroid):
        sub_coords = coords[i:]  # Remaining points to consider
        distances = cdist([coords[i]], sub_coords)[0]
        distances[distances == 0] = np.nan  # ignore self
        nearest_idx = np.nanargmin(distances) + i  # absolute index

        # Swap next voxel with nearest
        coords[[i + 1, nearest_idx]] = coords[[nearest_idx, i + 1]]
        i += 1

    return coords[: i + 1]
