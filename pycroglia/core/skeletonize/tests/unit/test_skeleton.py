from pycroglia.core.skeletonize import skeleton
import pytest
import numpy as np
from pycroglia.core.skeletonize.raytracing import factory

def test_skeleton():
    line = np.zeros((10, 10), dtype=bool)
    line[4, 1:7] = True 
    result = skeleton.skeleton(line, stepper_type=factory.StepperType.Simple)
    print(result)
    expected = [np.array([[7,5],[6,5],[5,5], [4,5],[3,5],[3,5]])]
    assert len(result) == 1
    assert np.array_equal(expected[0],result[0])

