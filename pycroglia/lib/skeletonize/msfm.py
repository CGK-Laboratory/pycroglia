import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Tuple
import heapq

F64 = np.float64
F64Array = npt.NDArray[F64]
BoolArray = npt.NDArray[np.uint8]
IntArray = npt.NDArray[np.int64]

# NOTE(juan): Arbitrary value from the original matlab implementation
EPS: F64 = F64(2.2204460492503131e-16)


def bounds_check(dims: Tuple[int, ...], i: int, j: int) -> bool:
    assert len(dims) == 2
    return i >= 0 and j >= 0 and i < dims[0] and j < dims[1]


def is_frozen2d(i: int, j: int, frozen: BoolArray) -> bool:
    """
    Checks if the pixel at frozen[i, j] is frozen
    """
    return frozen[i, j] == 1


def roots(coeffs: np.ndarray) -> np.ndarray:
    """Compute roots of a quadratic polynomial a*x^2 + b*x + c = 0
    Matching the custom MATLAB logic in `msfm2d`.
    """
    a, b, c = coeffs
    d = max((b * b) - 4.0 * a * c, 0.0)  # Discriminant, clamped to non-negative

    roots = np.zeros(2, dtype=np.float64)
    if a != 0.0:
        sqrt_d = np.sqrt(d)
        roots[0] = (-b - sqrt_d) / (2.0 * a)
        roots[1] = (-b + sqrt_d) / (2.0 * a)
    else:
        # Degenerate quadratic (actually linear): a == 0
        sqrt_d = np.sqrt(d)
        denominator1 = -b - sqrt_d
        denominator2 = -b + sqrt_d
        if denominator1 != 0.0:
            roots[0] = (2.0 * c) / denominator1
        else:
            roots[0] = np.inf
        if denominator2 != 0.0:
            roots[1] = (2.0 * c) / denominator2
        else:
            roots[1] = np.inf
    return roots


@dataclass
class DerivativeResult:
    tm: F64Array = np.zeros(4)
    order: BoolArray = np.zeros(4, dtype=np.uint8)


def _tpatch(t: F64Array, i: int, j: int, frozen: BoolArray) -> F64:
    tpatch = np.float64(np.inf)
    if bounds_check(frozen.shape, i, j) and is_frozen2d(i, j, frozen):
        tpatch = t[i, j]
    return tpatch


def _compute_second_order(
    first: Tuple[F64, F64], second: Tuple[F64, F64]
) -> Tuple[F64, np.uint8 | None]:
    value = F64(0.0)
    ch1 = (first[0] < first[1]) and np.isfinite(first[1])
    ch2 = (second[0] < second[1]) and np.isfinite(second[1])

    if ch1 and ch2:
        value = np.min(
            [
                (4.0 * first[1] - first[0]) / 3.0,
                (4.0 * second[1] - second[0]) / 3.0,
            ]
        )
        return (value, np.uint8(2))
    elif ch1:
        value = (4.0 * first[1] - first[0]) / 3.0
        return (F64(value), np.uint8(2))
    elif ch2:
        value = (4.0 * second[1] - second[0]) / 3.0
        return (F64(value), np.uint8(2))
    else:
        return (F64(value), None)


class FirstOrderTPatch:
    def __init__(self, t: F64Array, i: int, j: int, frozen: BoolArray) -> None:
        self.idx_2_3: F64 = _tpatch(t, i - 1, j, frozen)
        self.idx_3_2: F64 = _tpatch(t, i, j - 1, frozen)
        self.idx_3_4: F64 = _tpatch(t, i, j + 1, frozen)
        self.idx_4_3: F64 = _tpatch(t, i + 1, j, frozen)
        self.idx_2_2: F64 = _tpatch(t, i - 1, j - 1, frozen)
        self.idx_2_4: F64 = _tpatch(t, i - 1, j + 1, frozen)
        self.idx_4_2: F64 = _tpatch(t, i + 1, j - 1, frozen)
        self.idx_4_4: F64 = _tpatch(t, i + 1, j + 1, frozen)

    def calculate_derivative(self) -> DerivativeResult:
        order = np.zeros(4, dtype=np.uint8)
        tm = np.zeros(4)

        tm[0] = np.min([self.idx_2_3, self.idx_4_3])
        if np.isfinite(tm[0]):
            order[0] = 1

        tm[1] = np.min([self.idx_3_2, self.idx_3_4])
        if np.isfinite(tm[1]):
            order[1] = 1

        tm[2] = np.min([self.idx_2_2, self.idx_4_4])
        if np.isfinite(tm[2]):
            order[2] = 1

        tm[3] = np.min([self.idx_2_4, self.idx_4_2])
        if np.isfinite(tm[3]):
            order[3] = 1

        return DerivativeResult(tm=tm, order=order)


class SecondOrderTPatch:
    def __init__(
        self,
        first_order: FirstOrderTPatch,
        order: BoolArray,
        t: F64Array,
        i: int,
        j: int,
        frozen: BoolArray,
    ) -> None:
        self.txm2 = _tpatch(t, i - 2, j, frozen)
        self.txp2 = _tpatch(t, i + 2, j, frozen)
        self.tym2 = _tpatch(t, i, j - 2, frozen)
        self.typ2 = _tpatch(t, i, j + 2, frozen)
        self.tr1m2 = _tpatch(t, i - 2, j - 2, frozen)
        self.tr2m2 = _tpatch(t, i - 2, j + 2, frozen)
        self.tr2p2 = _tpatch(t, i + 2, j - 2, frozen)
        self.tr1p2 = _tpatch(t, i + 2, j + 2, frozen)
        self.first_order = first_order
        self.order = order

    def calculate_derivative(self) -> DerivativeResult:
        tm2 = np.zeros(4)
        tm2[0], order0 = _compute_second_order(
            (self.txm2, self.first_order.idx_2_3), (self.txp2, self.first_order.idx_4_3)
        )
        if order0 is not None:
            self.order[0] = order0

        tm2[1], order1 = _compute_second_order(
            (self.tym2, self.first_order.idx_3_2), (self.typ2, self.first_order.idx_3_4)
        )
        if order1 is not None:
            self.order[1] = order1

        tm2[2], order2 = _compute_second_order(
            (self.tr1m2, self.first_order.idx_2_2),
            (self.tr1p2, self.first_order.idx_4_4),
        )
        if order2 is not None:
            self.order[2] = order2

        tm2[3], order3 = _compute_second_order(
            (self.tr2m2, self.first_order.idx_2_4),
            (self.tr2p2, self.first_order.idx_4_2),
        )
        if order3 is not None:
            self.order[3] = order3
        return DerivativeResult(tm=tm2, order=self.order)


# NOTE(juan): removed cross option for computing tpatch, compute all directly
def _calculate_distance(
    image: F64Array,
    fij: F64,
    i: int,
    j: int,
    frozen: BoolArray,
    use_second: bool = False,
    use_cross: bool = False,
):
    # Constants
    CROSS_TERM = 0.5

    first_order_tpatch = FirstOrderTPatch(image, i, j, frozen)
    result = first_order_tpatch.calculate_derivative()
    second_result = DerivativeResult()
    if use_second:
        second_order_tpatch = SecondOrderTPatch(
            first_order_tpatch, result.order, image, i, j, frozen
        )
        second_result = second_order_tpatch.calculate_derivative()

    coefficients = np.array([0.0, 0.0, -1.0 / np.max([fij * fij, EPS])], dtype=F64)

    for i in range(0, 2):
        if result.order[i] == 1:
            coefficients[0] += 1.0
            coefficients[1] += -2.0 * result.tm[i]
            coefficients[2] += result.tm[i] * result.tm[i]
        elif result.order[i] == 2:
            coefficients[0] += 2.2500
            coefficients[1] += -2.0 * second_result.tm[i] * (2.2500)
            coefficients[2] += second_result.tm[i] * second_result.tm[i] * (2.2500)

    ansroot = roots(coefficients)
    tt = np.max(ansroot)

    if use_cross:
        coefficients[2] += -1.0 / np.max([fij * fij, EPS])
        for i in range(2, 4):
            if result.order[i] == 1:
                coefficients[0] += CROSS_TERM
                coefficients[1] += -2.0 * CROSS_TERM * result.tm[i]
                coefficients[2] += CROSS_TERM * result.tm[i] * result.tm[i]
            elif result.order[i] == 2:
                coefficients[0] += CROSS_TERM * 2.25
                coefficients[1] += -2 * CROSS_TERM * second_result.tm[i] * (2.25)
                coefficients[2] += (
                    second_result.tm[i] * second_result.tm[i] * CROSS_TERM * 2.25
                )

        if coefficients[0] > 0.0:
            ansroot = roots(coefficients)
            tt2 = np.max(ansroot)
            tt = np.min([tt, tt2])

        for i in range(0, 4):
            direct_neighbors = result.tm[np.isfinite(result.tm)]
            if np.any(direct_neighbors >= tt):
                tt = np.min(direct_neighbors) + (1.0 / np.max([fij, EPS]))
    else:
        for i in range(0, 2):
            direct_neighbors = result.tm[np.isfinite(result.tm)]
            if np.any(direct_neighbors >= tt):
                tt = np.min(direct_neighbors) + (1.0 / np.max([fij, EPS]))

    return tt


def msfm2d(
    speed_image: F64Array,
    source_points: IntArray,
    use_second: bool = True,
    use_cross: bool = True,
    skeletonize: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculates the shortest distance from a list of points to all other pixels in an image using the Multistencil Fast Marching Method (MSFM).

    This function implements the MSFM algorithm, which computes distance maps more accurately than the standard Fast Marching Method by incorporating second-order derivatives and cross-neighbor stencils.

    Args:
        speed_image (np.ndarray): The speed function image. All values must be strictly greater than zero (minimum value is 1e-8). If any values are zero or negative, those regions will never be reached (i.e., infinite time).
        source_points (np.ndarray): A list of starting points with shape (N, 2). These are the grid coordinates where the distance is initialized to zero.
        use_second (bool, optional): If True, the method will use both first- and second-order derivatives. Defaults to True.
        use_cross (bool, optional): If True, the method will include diagonal (cross) neighbors in derivative computations. Defaults to True.
        skeletonize (bool, optional): If True, also returns a Euclidean distance image (used in skeletonization contexts). Defaults to False.

    Returns:
        np.ndarray or Tuple[np.ndarray, np.ndarray]: A distance image `T` of the same shape as `speed_image`, where each pixel holds the shortest distance from any source point. If `skeletonize` is True, also returns a second image with Euclidean distances.
    """
    assert speed_image.ndim == 2, "Image should be 2D"

    dims = speed_image.shape
    output_distance_image = np.full(dims, -1.0)
    frozen = np.zeros(dims, dtype=np.uint8)
    euclidean_distance_image = np.full(dims, -1.0, dtype=F64) if skeletonize else None
    heap = []  # Min-heap for narrow band
    # Initialize source points
    for x, y in source_points:
        if 0 <= x < dims[0] and 0 <= y < dims[1]:
            frozen[x, y] = 1
            output_distance_image[x, y] = 0.0
            if euclidean_distance_image:
                euclidean_distance_image[x, y] = 0.0

    # Neighbors: 4-connected
    ne = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Add neighbors of source points
    for x, y in source_points:
        for dx, dy in ne:
            i, j = x + dx, y + dy
            if bounds_check(frozen.shape, i, j) and not is_frozen2d(i, j, frozen):
                t_val = 1.0 / np.max([speed_image[i, j], EPS])
                if (
                    output_distance_image[i, j] == -1
                    or t_val < output_distance_image[i, j]
                ):
                    output_distance_image[i, j] = t_val
                    heapq.heappush(heap, (t_val, i, j))
                    if euclidean_distance_image:
                        euclidean_distance_image[i, j] = 1.0

    # Fast marching loop
    while heap:
        t_val, x, y = heapq.heappop(heap)
        if frozen[x, y]:
            continue
        frozen[x, y] = 1
        output_distance_image[x, y] = t_val

        if euclidean_distance_image:
            # Assume distance 1 for uniform skeleton metric
            euclidean_distance_image[x, y] = euclidean_distance_image[x, y]

        for dx, dy in ne:
            i, j = x + dx, y + dy
            if bounds_check(frozen.shape, i, j) and not is_frozen2d(i, j, frozen):
                t_new = _calculate_distance(
                    output_distance_image,
                    F64(speed_image[i, j]),
                    i,
                    j,
                    frozen,
                    use_second,
                    use_cross,
                )
                if (
                    t_new < output_distance_image[i, j]
                    or output_distance_image[i, j] == -1
                ):
                    output_distance_image[i, j] = t_new
                    heapq.heappush(heap, (t_new, i, j))
                    if euclidean_distance_image:
                        y_new = _calculate_distance(
                            euclidean_distance_image,
                            F64(1.0),
                            i,
                            j,
                            frozen,
                            use_second,
                            use_cross,
                        )
                        euclidean_distance_image[i, j] = y_new
    return (
        (output_distance_image, euclidean_distance_image)
        if skeletonize
        else output_distance_image
    )[0]
