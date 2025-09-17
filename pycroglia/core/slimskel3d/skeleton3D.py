from numpy.typing import NDArray
import numpy as np


def _border_candidates(skel: NDArray, current_border: int) -> NDArray:
    """Compute border-deletion candidates for one sweep direction.

    A voxel is considered a candidate if it lies on the "border"
    between foreground and background when sweeping along one of
    the six Cartesian directions (+/-x, +/-y, +/-z). This matches
    the conditions in the Lee–Kashyap–Chu thinning algorithm.

    Args:
        skel (NDArray):
            3D binary skeleton volume of shape ``(z, y, x)``.
        current_border (int):
            Index of the current border direction:
            - 1: +y (south)
            - 2: -y (north)
            - 3: -x (west)
            - 4: +x (east)
            - 5: -z (bottom)
            - 6: +z (top)

    Returns:
        NDArray:
            Boolean mask of the same shape as `skel`, with True
            at voxels that are candidates for deletion in this
            sweep direction.
    """
    c = np.zeros_like(skel, dtype=bool)

    if current_border == 4:
        c[..., 1:] = skel[..., 1:] & (~skel[..., :-1])  # +x
    elif current_border == 3:
        c[..., :-1] = skel[..., :-1] & (~skel[..., 1:])  # -x
    elif current_border == 1:
        c[:, 1:, :] = skel[:, 1:, :] & (~skel[:, :-1, :])  # +y
    elif current_border == 2:
        c[:, :-1, :] = skel[:, :-1, :] & (~skel[:, 1:, :])  # -y
    elif current_border == 6:
        c[1:, :, :] = skel[1:, :, :] & (~skel[:-1, :, :])  # +z
    elif current_border == 5:
        c[:-1, :, :] = skel[:-1, :, :] & (~skel[1:, :, :])  # -z

    # Ensure only foreground voxels are candidates
    return c & skel


def _fill_euler_lut() -> NDArray:
    """Construct the Euler characteristic look-up table (LUT).

    This LUT encodes the Euler characteristic contributions of all
    possible 2×2×2 binary voxel configurations (indexed as an 8-bit
    integer from 0 to 255). It is used during skeletonization to
    ensure voxel deletions preserve topology.

    The table was originally derived from the MATLAB function
    ``FillEulerLUT`` (Kollmannsberger, 2013) and implements the
    same assignment rules. Only specific entries are non-zero,
    corresponding to configurations that affect the Euler number.

    Returns:
        NDArray:
            A 1D array of shape (256,) with dtype ``int8``.
            Index ``lut[k]`` contains the Euler contribution
            for configuration ``k``.

    Notes:
        - Index 0 (all voxels empty) is unused and remains 0.
        - Negative values indicate topology-breaking configurations.
        - This LUT is referenced inside ``_compute_euler_invariant_mask``.
    """
    lut = np.zeros(256, dtype=np.int8)  # index 0 unused (kept 0)
    assignments = {
        1: 1,
        3: -1,
        5: -1,
        7: 1,
        9: -3,
        11: -1,
        13: -1,
        15: 1,
        17: -1,
        19: 1,
        21: 1,
        23: -1,
        25: 3,
        27: 1,
        29: 1,
        31: -1,
        33: -3,
        35: -1,
        37: 3,
        39: 1,
        41: 1,
        43: -1,
        45: 3,
        47: 1,
        49: -1,
        51: 1,
        53: 1,
        55: -1,
        57: 3,
        59: 1,
        61: 1,
        63: -1,
        65: -3,
        67: 3,
        69: -1,
        71: 1,
        73: 1,
        75: 3,
        77: -1,
        79: 1,
        81: -1,
        83: 1,
        85: 1,
        87: -1,
        89: 3,
        91: 1,
        93: 1,
        95: -1,
        97: 1,
        99: 3,
        101: 3,
        103: 1,
        105: 5,
        107: 3,
        109: 3,
        111: 1,
        113: -1,
        115: 1,
        117: 1,
        119: -1,
        121: 3,
        123: 1,
        125: 1,
        127: -1,
        129: -7,
        131: -1,
        133: -1,
        135: 1,
        137: -3,
        139: -1,
        141: -1,
        143: 1,
        145: -1,
        147: 1,
        149: 1,
        151: -1,
        153: 3,
        155: 1,
        157: 1,
        159: -1,
        161: -3,
        163: -1,
        165: 3,
        167: 1,
        169: 1,
        171: -1,
        173: 3,
        175: 1,
        177: -1,
        179: 1,
        181: 1,
        183: -1,
        185: 3,
        187: 1,
        189: 1,
        191: -1,
        193: -3,
        195: 3,
        197: -1,
        199: 1,
        201: 1,
        203: 3,
        205: -1,
        207: 1,
        209: -1,
        211: 1,
        213: 1,
        215: -1,
        217: 3,
        219: 1,
        221: 1,
        223: -1,
        225: 1,
        227: 3,
        229: 3,
        231: 1,
        233: 5,
        235: 3,
        237: 3,
        239: 1,
        241: -1,
        243: 1,
        245: 1,
        247: -1,
        249: 3,
        251: 1,
        253: 1,
        255: -1,
    }
    for k, v in assignments.items():
        lut[k] = v
    return lut


EULER_LUT: NDArray = _fill_euler_lut()


def skeleton3D(volume: NDArray) -> NDArray:
    """3D skeletonization of a binary volume using the
    Lee–Kashyap–Chu medial axis thinning algorithm.

    This implementation follows the algorithm from:

        Lee, T.C., Kashyap, R.L., & Chu, C.N. (1994).
        "Building skeleton models via 3-D medial surface/axis
        thinning algorithms." Computer Vision, Graphics, and
        Image Processing, 56(6): 462–478.

    The algorithm iteratively deletes border voxels from the
    foreground volume, ensuring that deletions do not alter
    the topology of the object. Candidate voxels are removed
    only if they satisfy three conditions:

    1. **Not endpoints**: The voxel has more than one neighbor.
    2. **Euler invariant**: Removal preserves the Euler characteristic,
       verified with a precomputed look-up table (LUT).
    3. **Simple point**: Removal does not break connectivity, verified
       using recursive octant labeling.

    The thinning continues until no voxels can be removed from
    any of the six sweep directions.

    Args:
        volume (NDArray):
            Input 3D binary volume of shape ``(z, y, x)``.
            Must be boolean or convertible to boolean.

    Returns:
        NDArray:
            Skeletonized 3D binary volume with the same shape
            and dtype as the input.

    Notes:
        - A 1-voxel border of zeros is temporarily padded around
          the input to avoid edge effects; this is removed before
          returning the result.
        - Neighborhoods are evaluated in 3×3×3 cubes flattened
          in (z, y, x) order.
        - The Euler LUT is precomputed once via :func:`_fill_euler_lut`.
        - Simple point detection is performed with
          :func:`_is_simple_point`, which recursively labels
          connected components of the neighborhood.

    References:
        - Lee, T.C., Kashyap, R.L., & Chu, C.N. (1994).
          *Building skeleton models via 3-D medial surface/axis
          thinning algorithms*. CVGIP: Image Understanding,
          56(6): 462–478.
    """
    orig_dtype = volume.dtype
    skel = np.pad(volume.astype(bool, copy=False), 1, mode="constant")

    # Shape in (z, y, x)
    zdim, ydim, xdim = skel.shape

    unchanged_borders = 0
    while unchanged_borders < 6:
        unchanged_borders = 0

        for current_border in (1, 2, 3, 4, 5, 6):
            cands_mask = _border_candidates(skel, current_border)
            no_change_this_border = True

            if cands_mask.any():
                cands_indices = np.flatnonzero(cands_mask)
                nhood = _get_neighbourhood(skel, cands_indices)

                # Remove endpoints (exactly one neighbor → sum==2 incl. center)
                di1 = nhood.sum(axis=1) == 2
                if np.any(di1):
                    keep = ~di1
                    nhood = nhood[keep]
                    cands_indices = cands_indices[keep]

                if cands_indices.size > 0:
                    # Euler invariant filter
                    di2 = ~_compute_euler_invariant_mask(nhood, EULER_LUT)
                    if np.any(di2):
                        keep = ~di2
                        nhood = nhood[keep]
                        cands_indices = cands_indices[keep]

                if cands_indices.size > 0:
                    # Simple point filter
                    di3 = ~_is_simple_point(nhood)
                    if np.any(di3):
                        keep = ~di3
                        cands_indices = cands_indices[keep]
                        nhood = nhood[keep]

                if cands_indices.size > 0:
                    # Partition into 8 interleaved subvolumes
                    zc, yc, xc = np.unravel_index(cands_indices, (zdim, ydim, xdim))
                    x1, y1, z1 = (xc + 1) % 2 == 1, (yc + 1) % 2 == 1, (zc + 1) % 2 == 1
                    x2, y2, z2 = ~x1, ~y1, ~z1

                    submasks = [
                        (x1 & y1 & z1),
                        (x2 & y1 & z1),
                        (x1 & y2 & z1),
                        (x2 & y2 & z1),
                        (x1 & y1 & z2),
                        (x2 & y1 & z2),
                        (x1 & y2 & z2),
                        (x2 & y2 & z2),
                    ]

                    for submask in submasks:
                        if not np.any(submask):
                            continue
                        sel_lin = cands_indices[submask]

                        # Tentatively delete
                        skel.ravel()[sel_lin] = False

                        neighbourhood_recheck = _get_neighbourhood(skel, sel_lin)
                        bad_euler = ~_compute_euler_invariant_mask(
                            neighbourhood_recheck, EULER_LUT
                        )
                        bad_simple = ~_is_simple_point(neighbourhood_recheck)

                        # Revert violations
                        if np.any(bad_euler):
                            skel.ravel()[sel_lin[bad_euler]] = True
                        if np.any(bad_simple):
                            skel.ravel()[sel_lin[bad_simple]] = True

                        survived = ~(bad_euler | bad_simple)
                        if np.any(survived):
                            no_change_this_border = False

            if no_change_this_border:
                unchanged_borders += 1

    # Remove padding and restore dtype
    skel = skel[1:-1, 1:-1, 1:-1]
    return skel.astype(orig_dtype, copy=False)


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


def _compute_euler_invariant_mask(img: NDArray, lut: NDArray) -> NDArray:
    """Check Euler characteristic preservation for voxel neighborhoods.

    This function evaluates whether candidate voxels can be removed
    without altering the Euler characteristic of the 3D object. It does
    so by computing the Euler contribution of each voxel’s 3×3×3
    neighborhood across all 8 octants, and summing these contributions
    using the precomputed Euler look-up table (LUT).

    Args:
        img (NDArray):
            Boolean array of shape (N, 27). Each row encodes the 3×3×3
            neighborhood of a voxel, flattened in (z, y, x) order, where
            column 13 is the center voxel.
        lut (NDArray):
            Precomputed Euler LUT of shape (256,) with dtype ``int8``.
            The LUT maps a 7-bit configuration index (0–255) to its
            Euler characteristic contribution. Must have length 256.

    Returns:
        NDArray:
            Boolean mask of shape (N,). ``True`` if the voxel’s deletion
            preserves the Euler characteristic (i.e., total contribution
            equals 0), ``False`` otherwise.

    Notes:
        - Each octant corresponds to a 2×2×2 sub-cube around the voxel.
        - Only 7 neighbors of the sub-cube are considered per octant,
          yielding a 7-bit configuration index.
        - These 7 bits are combined into an index via bitwise OR
          (weights: [128, 64, 32, 16, 8, 4, 2]).
        - The LUT provides the Euler contribution for each configuration.
        - If the sum of contributions from all 8 octants is 0, the voxel
          is Euler invariant.

    References:
        - P. Kollmannsberger, *Skeleton3D* (MATLAB implementation).
        - Lee, Kashyap, and Chu (1994): "Building skeleton models via
          3-D medial surface/axis thinning algorithms."
    """

    assert lut.size == 256
    N = img.shape[0]
    euler_char = np.zeros(N, dtype=np.int32)

    bitor_table = np.array([128, 64, 32, 16, 8, 4, 2], dtype=np.uint8)

    # (z,y,x) → col index mapping for your neighbourhood order
    coord_to_col = {
        (dz - 1, dy - 1, dx - 1): dz * 9 + dy * 3 + dx
        for dz in range(3)
        for dy in range(3)
        for dx in range(3)
    }

    octants = [
        # SWU
        [
            (+1, -1, -1),
            (+1, -1, 0),
            (0, -1, -1),
            (0, -1, 0),
            (+1, 0, -1),
            (+1, 0, 0),
            (0, 0, -1),
        ],
        # SEU
        [
            (+1, -1, +1),
            (+1, -1, 0),
            (0, -1, +1),
            (0, -1, 0),
            (+1, 0, +1),
            (+1, 0, 0),
            (0, 0, +1),
        ],
        # NWU
        [
            (+1, +1, -1),
            (+1, +1, 0),
            (0, +1, -1),
            (0, +1, 0),
            (+1, 0, -1),
            (+1, 0, 0),
            (0, 0, -1),
        ],
        # NEU
        [
            (+1, +1, +1),
            (+1, +1, 0),
            (0, +1, +1),
            (0, +1, 0),
            (+1, 0, +1),
            (+1, 0, 0),
            (0, 0, +1),
        ],
        # SWB
        [
            (-1, -1, -1),
            (-1, -1, 0),
            (0, -1, -1),
            (0, -1, 0),
            (-1, 0, -1),
            (-1, 0, 0),
            (0, 0, -1),
        ],
        # SEB
        [
            (-1, -1, +1),
            (-1, -1, 0),
            (0, -1, +1),
            (0, -1, 0),
            (-1, 0, +1),
            (-1, 0, 0),
            (0, 0, +1),
        ],
        # NWB
        [
            (-1, +1, -1),
            (-1, +1, 0),
            (0, +1, -1),
            (0, +1, 0),
            (-1, 0, -1),
            (-1, 0, 0),
            (0, 0, -1),
        ],
        # NEB
        [
            (-1, +1, +1),
            (-1, +1, 0),
            (0, +1, +1),
            (0, +1, 0),
            (-1, 0, +1),
            (-1, 0, 0),
            (0, 0, +1),
        ],
    ]

    def accumulate(coords):
        # IMPORTANT: start from 1 to match MATLAB’s LUT addressing
        n = np.ones(N, dtype=np.uint8)
        for bit, coord in zip(bitor_table, coords):
            col = coord_to_col[coord]
            mask = img[:, col]
            # img is bool; where True, OR in the bit
            n[mask] = np.bitwise_or(n[mask], bit)
        return lut[n]

    for octant in octants:
        euler_char += accumulate(octant)

    return euler_char == 0


OCTANT_RULES: dict[int, list[tuple[int, list[int]]]] = {
    1: [
        (0, []),  # MATLAB 1
        (1, [2]),  # 2
        (3, [3]),  # 4
        (4, [2, 3, 4]),  # 5
        (9, [5]),  # 10
        (10, [2, 5, 6]),  # 11
        (12, [3, 5, 7]),  # 13
    ],
    2: [
        (1, [1]),  # 2
        (4, [1, 3, 4]),  # 5
        (10, [1, 5, 6]),  # 11
        (2, []),  # 3
        (5, [4]),  # 6
        (11, [6]),  # 12
        (13, [4, 6, 8]),  # 14
    ],
    3: [
        (3, [1]),  # 4
        (4, [1, 2, 4]),  # 5
        (12, [1, 5, 7]),  # 13
        (6, []),  # 7
        (7, [4]),  # 8
        (14, [7]),  # 15
        (15, [4, 7, 8]),  # 16
    ],
    4: [
        (4, [1, 2, 3]),  # 5
        (5, [2]),  # 6
        (13, [2, 6, 8]),  # 14
        (7, [3]),  # 8
        (15, [3, 7, 8]),  # 16
        (8, []),  # 9
        (16, [8]),  # 17
    ],
    5: [
        (9, [1]),  # 10
        (10, [1, 2, 6]),  # 11
        (12, [1, 3, 7]),  # 13
        (17, []),  # 18
        (18, [6]),  # 19
        (20, [7]),  # 21
        (21, [6, 7, 8]),  # 22
    ],
    6: [
        (10, [1, 2, 5]),  # 11
        (11, [2]),  # 12
        (13, [2, 4, 8]),  # 14
        (18, [5]),  # 19
        (21, [5, 7, 8]),  # 22
        (19, []),  # 20
        (22, [8]),  # 23
    ],
    7: [
        (12, [1, 3, 5]),  # 13
        (14, [3]),  # 15
        (15, [3, 4, 8]),  # 16
        (20, [5]),  # 21
        (21, [5, 6, 8]),  # 22
        (23, []),  # 24
        (24, [8]),  # 25
    ],
    8: [
        (13, [2, 4, 6]),  # 14
        (15, [3, 4, 7]),  # 16
        (16, [4]),  # 17
        (21, [5, 6, 7]),  # 22
        (22, [6]),  # 23
        (24, [7]),  # 25
        (25, []),  # 26
    ],
}


def _label_octant(octant: int, label: int, cube: NDArray) -> NDArray:
    """Recursively propagate labels across a voxel’s octant neighborhood.

    This function operates on the 26-neighbor configuration of a voxel’s
    3×3×3 neighborhood (excluding the center). The neighborhood is
    represented as a 1D array ``cube`` of length 26, where each entry
    corresponds to a neighbor voxel state (1 = foreground, 0 = background,
    >1 = already labeled).

    The function starts at the given ``octant`` (1..8), identifies which
    neighbors belong to that octant, and assigns them the current ``label``.
    If a neighbor is relabeled, recursive calls are made into the next
    connected octants specified by ``OCTANT_RULES``.

    Args:
        octant (int):
            Octant ID (1–8). Defines which subset of neighbors to process.
        label (int):
            Label value to assign to the connected voxels in this octant.
        cube (NDArray):
            1D array of length 26 (uint8). The neighborhood configuration
            of one voxel, excluding the center voxel.

    Returns:
        NDArray:
            Updated ``cube`` with connected voxels labeled.

    Notes:
        - Uses the global constant ``OCTANT_RULES`` to determine which
          neighbor indices belong to each octant, and which other octants
          should be visited recursively.
        - The recursion stops once all connected voxels in the octant are
          relabeled or when no new voxels remain.
    """
    for col, next_octants in OCTANT_RULES[octant]:
        if cube[col] == 1:
            cube[col] = label
            for nxt in next_octants:
                cube = _label_octant(nxt, label, cube)
    return cube


def _is_simple_point(N: NDArray) -> NDArray:
    """Check whether voxels are 'simple points' (can be safely removed).

    A voxel is considered *simple* if removing it does not alter the
    topological structure of the object. This check is used in thinning /
    skeletonization algorithms to preserve connectivity.

    Each voxel's 26-neighborhood (3×3×3 minus the center) is analyzed.
    The neighborhood is partitioned into octants, and connected components
    are recursively labeled using :func:`_label_octant`. If labeling
    requires more than 3 distinct labels, the voxel is **not simple**.

    Args:
        N (NDArray):
            Neighborhood table of shape ``(n_points, 27)``. Each row is the
            flattened 3×3×3 neighborhood (row-major, z–y–x order), with the
            center voxel at index 13.

    Returns:
        NDArray:
            Boolean mask of shape ``(n_points,)`` where ``True`` means the
            voxel is simple (removable) and ``False`` means the voxel must
            be preserved.
    """
    n_points = N.shape[0]
    is_simple = np.ones(n_points, dtype=bool)

    # Build cube without center voxel
    cube = np.zeros((n_points, 26), dtype=np.uint8)
    cube[:, :13] = N[:, :13]
    cube[:, 13:] = N[:, 14:]

    labels = np.full(n_points, 2, dtype=np.uint8)

    for row in range(n_points):
        if not is_simple[row]:
            continue

        for i in range(26):
            if cube[row, i] != 1:
                continue

            if i in {0, 1, 3, 4, 9, 10, 12}:
                cube[row, :] = _label_octant(1, labels[row], cube[row, :])
            elif i in {2, 5, 11, 13}:
                cube[row, :] = _label_octant(2, labels[row], cube[row, :])
            elif i in {6, 7, 14, 15}:
                cube[row, :] = _label_octant(3, labels[row], cube[row, :])
            elif i in {8, 16}:
                cube[row, :] = _label_octant(4, labels[row], cube[row, :])
            elif i in {17, 18, 20, 21}:
                cube[row, :] = _label_octant(5, labels[row], cube[row, :])
            elif i in {19, 22}:
                cube[row, :] = _label_octant(6, labels[row], cube[row, :])
            elif i in {23, 24}:
                cube[row, :] = _label_octant(7, labels[row], cube[row, :])
            elif i == 25:
                cube[row, :] = _label_octant(8, labels[row], cube[row, :])

            labels[row] += 1
            if labels[row] >= 4:
                is_simple[row] = False
                break

    return is_simple
