from dataclasses import dataclass
from enum import Enum
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import quad
import numpy as np


class InterpolationMethod(Enum):
    """Supported interpolation methods for arc length computation.

    Attributes:
        LINEAR: Straight-line (chordal) distances between successive points.
            Fastest and simplest, but less accurate for curved data.
        PCHIP: Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).
            Provides smoother parametric curves. Arc length is computed
            by numerically integrating the curve’s derivative.
        SPLINE: Cubic spline interpolation. Produces smooth curves and
            can give higher accuracy than PCHIP for smooth data, at
            higher computational cost.
    """

    LINEAR = "linear"
    PCHIP = "pchip"
    SPLINE = "spline"


def _assert_length(arr: NDArray):
    """
    Assert that all rows in a NumPy array have the same length.

    Handles both:
      - Regular 2D rectangular arrays (NumPy enforces equal lengths).
      - Ragged arrays with dtype=object (checks lengths manually).

    Args:
        arr (NDArray): A NumPy array containing coordinate sequences.

    Raises:
        AssertionError: If the array is not 2D (for rectangular case),
                        or if ragged rows have inconsistent lengths.

    Returns:
        None: If ok return None.
    """
    if arr.dtype == object:
        # ragged array: check lengths manually
        lengths = [len(row) for row in arr]
        assert len(set(lengths)) == 1, f"Not all rows have the same length: {lengths}"
    else:
        # already a rectangular NumPy array
        assert arr.ndim == 2, f"Expected 2D array, got shape {arr.shape}"


@dataclass
class Result:
    """
    Container for arc length results.

    Attributes:
        arclength (float): Total arc length of the curve.
        segment_lengths (NDArray): Array of arc lengths for each segment.
    """

    arclength: float
    segment_lengths: NDArray


def _speed(u: NDArray, splines: list[PchipInterpolator | CubicSpline]):
    """
    Evaluate curve speed (magnitude of derivative) at parameter u.

    Args:
        u (NDArray): Scalar or array of parameter values.
        splines (list): List of spline interpolators, one per coordinate.

    Returns:
        NDArray: Speed values at each u, i.e., sqrt(sum((dx/du)^2)).
    """
    return np.sqrt(sum(spline(u, 1) ** 2 for spline in splines))


def arclength(
    p: NDArray,
    method: InterpolationMethod = InterpolationMethod.LINEAR,
    assert_length: bool = True,
):
    """Compute the arc length of a curve represented by discrete points.

    This function supports chordal (linear) approximation or smooth
    interpolation via PCHIP or cubic splines, with numerical integration.

    Args:
        p (NDArray):
            A 2D array of shape (n_points, n_dims), representing a sequence
            of points along a curve. Each row is a point in n-dimensional
            space.
        method (InterpolationMethod, optional):
            Interpolation method for arc length computation. Options:
            - InterpolationMethod.LINEAR: straight-line distances between points.
            - InterpolationMethod.PCHIP: smooth PCHIP interpolation with integration.
            - InterpolationMethod.SPLINE: smooth cubic spline interpolation with integration.
            Defaults to InterpolationMethod.LINEAR.
        assert_length (bool, optional):
            If True, asserts that all rows of `p` have the same length
            (to catch ragged arrays). Defaults to True.

    Returns:
        Result:
            Dataclass with fields:
            - arclength (float): total arc length of the curve.
            - segment_lengths (NDArray): arc length of each segment.

    Raises:
        AssertionError: If input dimensions are invalid or rows of `p`
            have inconsistent lengths.
        ValueError: If no valid interpolation method is provided.
    """
    assert p.ndim >= 2, "At least two points needed"
    if assert_length:
        _assert_length(p)

    ndims = p.shape[1]
    diffs = np.diff(p, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    arclength_value = segment_lengths.sum()

    if method == InterpolationMethod.LINEAR:
        return Result(arclength=arclength_value, segment_lengths=segment_lengths)

    chord_lengths = segment_lengths
    segments_cumsum = np.hstack(([0], np.cumsum(chord_lengths)))

    splines = []
    interpolator = (
        PchipInterpolator if method == InterpolationMethod.PCHIP else CubicSpline
    )

    for i in range(ndims):
        splines.append(interpolator(segments_cumsum, p[:, i]))

    segment_lengths = np.zeros(len(chord_lengths))
    for i in range(len(chord_lengths)):
        val, _ = quad(
            lambda u: _speed(u, splines), segments_cumsum[i], segments_cumsum[i + 1]
        )
        segment_lengths[i] = val

    return Result(arclength=sum(segment_lengths), segment_lengths=segment_lengths)
