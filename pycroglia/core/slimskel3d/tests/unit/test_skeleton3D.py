import numpy as np
from pycroglia.core.slimskel3d import skeleton3D


def test_skeleton3D():
    # Generate a 5x5x5 voxel with a line in the middle
    vol = np.zeros((5, 5, 5), dtype=bool)
    vol[:, 2, 2] = True
    got = skeleton3D.skeleton3D(vol)
    expected = np.array(
        [
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
        ]
    )
    np.testing.assert_array_equal(got, expected)
