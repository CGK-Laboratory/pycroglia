import pytest
from unittest.mock import MagicMock
from pycroglia.ui.widgets.imagefilters.tasks import (
    TaskSignals,
    ImageReaderTask,
    GrayFilterTask,
    SmallObjectFilterTask,
)


@pytest.fixture
def mock_editor_state():
    state = MagicMock()
    return state


def test_task_signals_initialization():
    signals = TaskSignals()
    assert hasattr(signals, "finished")


def test_image_reader_task_initialization(mock_editor_state):
    task = ImageReaderTask(state=mock_editor_state, ch=3, chi=1)
    assert task.state == mock_editor_state
    assert task.ch == 3
    assert task.chi == 1


def test_gray_filter_task_initialization(mock_editor_state):
    task = GrayFilterTask(state=mock_editor_state, adjust_value=1.5)
    assert task.state == mock_editor_state
    assert task.adjust_value == 1.5


def test_small_object_filter_task_initialization(mock_editor_state):
    task = SmallObjectFilterTask(state=mock_editor_state, threshold=100)
    assert task.state == mock_editor_state
    assert task.threshold == 100
