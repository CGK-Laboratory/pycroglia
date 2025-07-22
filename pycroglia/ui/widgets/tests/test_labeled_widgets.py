import pytest

from pycroglia.ui.widgets.labeled_widgets import (
    LabeledSpinBox,
    LabeledIntSlider,
    LabeledFloatSlider,
)


@pytest.fixture
def labeled_spin_box(qtbot):
    """Fixture for LabeledSpinBox widget.

    Args:
        qtbot: pytest-qt bot for widget handling.

    Returns:
        LabeledSpinBox: The widget instance.
    """
    widget = LabeledSpinBox(label_text="Example label")
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def labeled_int_slider(qtbot):
    """Fixture for LabeledIntSlider widget.

    Args:
        qtbot: pytest-qt bot for widget handling.

    Returns:
        LabeledIntSlider: The widget instance.
    """
    slider = LabeledIntSlider(min_value=0, max_value=10)
    qtbot.addWidget(slider)
    return slider


@pytest.fixture
def labeled_float_slider(qtbot):
    """Fixture for LabeledFloatSlider widget.

    Args:
        qtbot: pytest-qt bot for widget handling.

    Returns:
        LabeledFloatSlider: The widget instance.
    """
    slider = LabeledFloatSlider(min_value=0.0, max_value=1.0, step_size=0.1)
    qtbot.addWidget(slider)
    return slider


def test_labeled_spin_box_get_value(labeled_spin_box):
    """Test that get_value returns the correct value after setting it."""
    new_value = 10

    labeled_spin_box.spin_box.setValue(new_value)
    assert labeled_spin_box.get_value() == new_value


def test_labeled_spin_box_set_value_emits_signal(labeled_spin_box, qtbot):
    """Test that setting the spin box value emits the valueChanged signal."""
    set_value = 2

    def assert_correct_emit_value(x):
        return x == set_value

    with qtbot.waitSignal(
        labeled_spin_box.valueChanged,
        timeout=1000,
        check_params_cb=assert_correct_emit_value,
    ):
        labeled_spin_box.spin_box.setValue(2)


def test_labeled_spin_box_set_max(labeled_spin_box):
    """Test that set_max sets the maximum value for the spin box."""
    new_max_value = 10
    labeled_spin_box.set_max(10)

    assert labeled_spin_box.spin_box.maximum() == new_max_value


def test_labeled_spin_box_set_max_changes_value(labeled_spin_box):
    """Test that set_max changes the value if it exceeds the new maximum."""
    spin_box_value = 4
    new_max_value = 2

    labeled_spin_box.spin_box.setValue(spin_box_value)
    labeled_spin_box.set_max(new_max_value)

    assert new_max_value == labeled_spin_box.get_value()


def test_labeled_int_slider_initial_value(labeled_int_slider):
    """Test that the initial value of the int slider is correct."""
    assert labeled_int_slider.get_value() == 0
    assert labeled_int_slider.value_label.text() == "Value: 0"


def test_labeled_int_slider_get_value(labeled_int_slider):
    """Test that get_value returns the correct value after setting it."""
    test_value = 10

    labeled_int_slider.slider.setValue(test_value)
    assert test_value == labeled_int_slider.get_value()


def test_labeled_slider_value_change_emits_signal(labeled_int_slider, qtbot):
    """Test that changing the slider value emits the valueChanged signal."""
    with qtbot.waitSignal(
        labeled_int_slider.valueChanged, timeout=1000, raising=True
    ) as blocker:
        labeled_int_slider.slider.setValue(5)
    assert blocker.args == [5]
    assert labeled_int_slider.value_label.text() == "Value: 5"


def test_labeled_float_slider_initial_value(labeled_float_slider):
    """Test that the initial value of the float slider is correct."""
    assert labeled_float_slider.get_value() == pytest.approx(0.0, abs=1e-6)
    assert labeled_float_slider.value_label.text() == "Value: 0.00"


def test_labeled_float_slider_value_change_emits_signal(labeled_float_slider, qtbot):
    """Test that changing the float slider value emits the valueChanged signal."""
    expected_value = 0.3
    with qtbot.waitSignal(labeled_float_slider.valueChanged, timeout=1000) as blocker:
        labeled_float_slider.set_value(expected_value)

    assert blocker.args[0] == pytest.approx(expected_value, abs=1e-6)
    assert labeled_float_slider.value_label.text() == f"Value: {expected_value:.2f}"


def test_labeled_float_slider_get_value(labeled_float_slider):
    """Test that get_value returns the correct float value after setting it."""
    test_value = 10

    labeled_float_slider.slider.setValue(test_value)
    assert labeled_float_slider.get_value() == test_value * labeled_float_slider._step


def test_labeled_float_slider_set_value(labeled_float_slider):
    """Test that set_value sets the float slider to the correct value."""
    labeled_float_slider.set_value(0.5)
    assert labeled_float_slider.get_value() == pytest.approx(0.5, abs=1e-6)
