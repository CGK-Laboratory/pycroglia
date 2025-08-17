from pycroglia.core.skeletonize.raytracing import rk4, simple, euler
import numpy as np

def test_euler():
    X, Y = np.meshgrid(np.arange(100), np.arange(100))
    distance_map = np.sqrt((X-50)**2 + (Y-50)**2)
    grad_y, grad_x = np.gradient(distance_map)
    gradient_volume = np.stack([-grad_x, -grad_y], axis=-1)
    start_point = np.array([51.0, 51.0])
    step_size = 0.1
    stepper = euler.Euler(step_size, gradient_volume)
    result = stepper.step(start_point)
    expected = np.array([50.9293, 50.9293])  # small step toward center
    assert np.allclose(expected, result, atol=1e-2)

def test_simple():
    X, Y = np.meshgrid(np.arange(100), np.arange(100))
    distance_map = np.sqrt((X-50)**2 + (Y-50)**2)
    stepper = simple.Simple(distance_map)
    start_point = np.array([51.0, 51.0])
    result = stepper.step(start_point)
    expected = np.array([50, 50])  # simple step goes to lowest neighbor
    assert np.allclose(expected, result, atol=1)


def test_rk4():
    X, Y = np.meshgrid(np.arange(100), np.arange(100))
    distance_map = np.sqrt((X-50)**2 + (Y-50)**2)
    grad_y, grad_x = np.gradient(distance_map)
    gradient_volume = np.stack([-grad_x, -grad_y], axis=-1)
    start_point = np.array([51.0, 51.0])
    step_size = 0.1
    stepper = rk4.RK4(step_size, gradient_volume)
    result = stepper.step(start_point)
    expected = np.array([50.9293, 50.9293])  # small step toward center
    assert np.allclose(expected, result, atol=1e-2)    

