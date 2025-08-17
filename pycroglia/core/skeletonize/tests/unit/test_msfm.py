import pytest
import numpy as np
from numpy import linalg as LA
from pycroglia.core.skeletonize.msfm import msfm2d


# TODO(jab227): change to pytest parametrize
def test_msfm2d():
    eps = 1e-2  # TODO(jab227): increment precision
    source_points = np.array([[51, 51]])
    speed_image = np.ones((101, 101))
    ys, xs = np.meshgrid(np.arange(0, 101), np.arange(0, 101))  # shape = (101, 101)
    euclidean_distance_from_source_point = np.sqrt(
        (xs - source_points[0][0]) ** 2 + (ys - source_points[0][1]) ** 2
    )
    cases = [
        {
            "case": "first order, no cross neighbours",
            "args": {"use_second": False, "use_cross": False},
            "expected": [0.74610, 0.69709, 1.31485],
        },
        {
            "case": "first order, cross neighbours",
            "args": {"use_second": False, "use_cross": True},
            "expected": [0.60693, 0.44561, 0.97252],
        },
        {
            "case": "second order, no cross neighbours",
            "args": {"use_second": True, "use_cross": False},
            "expected": [0.19740, 0.04234, 0.32895],
        },
        {
            "case": "second order, cross neighbours",
            "args": {"use_second": True, "use_cross": True},
            "expected": [0.03961, 0.00250, 0.18765],
        },
    ]

    for case in cases:
        distance_image = msfm2d(
            speed_image,
            source_points,
            use_second=case["args"]["use_second"],
            use_cross=case["args"]["use_cross"],
        )
        assert isinstance(distance_image, np.ndarray), f"case: {case['case']}"
        diff = euclidean_distance_from_source_point - distance_image
        errors = {
            "L1": LA.norm(diff.ravel(), ord=1) / diff.size,
            "L2": LA.norm(diff.ravel(), ord=2) ** 2 / diff.size,
            "Linf": LA.norm(diff.ravel(), ord=np.inf),
        }
        assert pytest.approx(errors["L1"], eps) == case["expected"][0], (
            f"case: {case['case']}, norm: L1"
        )
        assert pytest.approx(errors["L2"], eps) == case["expected"][1], (
            f"case: {case['case']}, norm: L2"
        )
        assert pytest.approx(errors["Linf"], eps) == case["expected"][2], (
            f"case: {case['case']}, norm: Linf"
        )


def test_msfm2d_multiple_starting_points():
    pass
