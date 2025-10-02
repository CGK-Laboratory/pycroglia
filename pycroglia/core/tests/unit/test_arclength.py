import numpy as np
from pycroglia.core.arclength import arclength, InterpolationMethod


def test_circle_linear():
    """Test arc length of a circle using linear chords.

    This test discretizes a unit circle into 20 points and computes
    its arc length using straight-line (linear chordal) distances.
    The result is expected to underestimate the true circumference
    slightly, but remain within 2% relative tolerance.

    Asserts:
        The computed arc length is close to ``2π`` within rtol=0.02.
    """
    theta = np.linspace(0, 2 * np.pi, 20)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    result = arclength(points, method=InterpolationMethod.LINEAR)
    # Linear underestimates but should be close
    assert np.isclose(result.arclength, 2 * np.pi, rtol=0.02)


def test_circle_pchip():
    """Test arc length of a circle using PCHIP interpolation.

    This test discretizes a unit circle into 20 points and computes
    its arc length using Piecewise Cubic Hermite Interpolating Polynomial
    (PCHIP) interpolation. The smoother parametric curve gives a more
    accurate estimate of the circumference.

    Asserts:
        The computed arc length is close to ``2π`` within rtol=0.005.
    """
    theta = np.linspace(0, 2 * np.pi, 20)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    result = arclength(points, method=InterpolationMethod.PCHIP)
    assert np.isclose(result.arclength, 2 * np.pi, rtol=0.005)


def test_circle_spline():
    """Test arc length of a circle using cubic spline interpolation.

    This test discretizes a unit circle into 20 points and computes
    its arc length using cubic spline interpolation. The smooth spline
    fit should provide a highly accurate estimate of the circumference.

    Asserts:
        The computed arc length is close to ``2π`` within rtol=0.005.
    """
    theta = np.linspace(0, 2 * np.pi, 20)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    result = arclength(points, method=InterpolationMethod.SPLINE)
    assert np.isclose(result.arclength, 2 * np.pi, rtol=0.005)
