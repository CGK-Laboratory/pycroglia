import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, binary_dilation

from eiko_skelfm.skeleton import skeleton


def fastmarch_skel(vol: NDArray, sigma: float = 1.0, verbose: bool = False) -> NDArray:
    """Skeletonize a 3D binary image using the Fast Marching Method.

    This function mirrors the MATLAB fast marching skeleton workflow:
    it optionally Gaussian-smooths the input volume before calling the
    ``eiko_skelfm`` ``skeleton`` function, then assembles all returned
    branch coordinates into a single binary output array.

    If the largest dimension of the volume exceeds 512 voxels the volume
    is first downsampled by a factor of 2 (via nearest-neighbour resizing)
    before skeletonization, and the resulting skeleton is then upsampled
    back to the original shape — matching the MATLAB behaviour of
    converting large cells to 512×512 to speed up processing.

    Args:
        vol (NDArray):
            3D binary array representing the object (True/1 = foreground).
        sigma (float):
            Standard deviation for the Gaussian smoothing applied before
            skeletonization.  When ``sigma > 0`` a Gaussian blur is applied
            and only voxels that remain above the 0.5 threshold are kept as
            foreground; this removes thin surface hairs before the fast
            marching step (mirrors ``imgaussfilt3`` in MATLAB).  For very
            thin structures (e.g. 1-voxel-wide lines) set ``sigma=0`` to
            avoid washing away the foreground entirely.  Defaults to 1.0.
        verbose (bool):
            Whether to print progress messages from the underlying
            ``skeleton`` function.  Defaults to ``False``.

    Returns:
        NDArray:
            A 3D boolean array of the same shape as *vol* where ``True``
            marks skeleton voxels.

    Notes:
        - ``eiko_skelfm.skeleton`` receives a binary boolean mask (not the
          raw float smoothed image).  The smoothing step only determines
          which voxels belong to the foreground mask passed to the library.
        - Branch coordinates returned by ``skeleton()`` are rounded to the
          nearest integer and clipped to the valid index range before being
          written into the output array.
        - The downsampling/upsampling step preserves skeleton topology while
          substantially reducing computation time for large volumes.
        - If the volume contains no foreground voxels after optional
          smoothing an all-False array is returned without calling the
          underlying solver (which would raise a ``RuntimeError``).

    Example:
        >>> import numpy as np
        >>> from pycroglia.core.fastmarch.fastmarch_skel import fastmarch_skel
        >>> vol = np.zeros((30, 30, 30), dtype=bool)
        >>> vol[5:25, 15, 15] = True   # simple straight line
        >>> skel = fastmarch_skel(vol, sigma=0)
        >>> skel.any()
        True
    """
    original_shape = vol.shape
    working_vol = vol.astype(bool)

    # --- Downsample large volumes (mirrors MATLAB: if s(1) > 512) -----------
    downsampled = False
    if max(original_shape) > 512:
        from skimage.transform import resize

        new_shape = tuple(max(1, s // 2) for s in original_shape)
        working_vol = resize(
            working_vol, new_shape, order=0, anti_aliasing=False, preserve_range=True
        ).astype(bool)
        downsampled = True

    # --- Gaussian smoothing (mirrors MATLAB: imgaussfilt3) ------------------
    # Apply smoothing to remove thin surface hairs; the result is thresholded
    # back to a binary mask before being passed to the fast marching solver.
    if sigma > 0:
        smoothed_mask = gaussian_filter(working_vol.astype(float), sigma=sigma) > 0.5
    else:
        smoothed_mask = working_vol

    # --- Early exit for empty foreground ------------------------------------
    # eiko_skelfm raises RuntimeError when there are no boundary source points
    # (i.e. the foreground is empty). Return a blank skeleton in that case.
    if not smoothed_mask.any():
        return np.zeros(original_shape, dtype=bool)

    # --- Fast Marching skeleton ---------------------------------------------
    branches: list[NDArray] = skeleton(smoothed_mask.astype(bool), verbose=verbose)

    # --- Assemble binary skeleton image (mirrors MATLAB: WholeSkel) ---------
    skel = np.zeros(working_vol.shape, dtype=bool)
    if branches:
        coords = np.round(np.vstack(branches)).astype(int)
        ndim = working_vol.ndim
        for dim in range(ndim):
            coords[:, dim] = np.clip(coords[:, dim], 0, working_vol.shape[dim] - 1)
        skel[tuple(coords[:, d] for d in range(ndim))] = True

    # --- Upsample skeleton back to original shape ---------------------------
    if downsampled:
        from skimage.transform import resize

        skel = resize(
            skel, original_shape, order=0, anti_aliasing=False, preserve_range=True
        ).astype(bool)

    return skel
