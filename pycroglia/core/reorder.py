import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


def reorder_pixel_list(
    pixel_indices: NDArray,
    shape: tuple[int, int, int],
    endpoint: NDArray,
    centroid: NDArray,
) -> NDArray:
    """Reorder voxel coordinates by connectivity from endpoint to centroid.

    This function receives the list of voxels that form a 3D branch and
    orders them so that the first voxel corresponds to the endpoint and
    subsequent voxels follow the nearest connected voxel until reaching
    the centroid.

    Args:
        pixel_indices (NDArray):
            Flat voxel indices (1D array) corresponding to `True` voxels in
            a 3D branch mask.
        shape (tuple[int, int, int]):
            Shape of the 3D mask volume (Z, Y, X).
        endpoint (NDArray):
            Array of shape (3,) with (z, y, x) coordinates of the endpoint.
        centroid (NDArray):
            Array of shape (3,) with (z, y, x) coordinates of the centroid.

    Returns:
        NDArray:
            (N, 3) array of voxel coordinates ordered by connectivity from
            endpoint to centroid. The first row is `endpoint`, and the last
            row is the centroid.

    Raises:
        AssertionError: If input arrays have invalid shapes or contain
            inconsistent indices.
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
