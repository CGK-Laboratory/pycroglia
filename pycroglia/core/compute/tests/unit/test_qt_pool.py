from typing import Any
from pycroglia.core.compute.computable import Computable
from pycroglia.core.compute.qt_pool import QPool


class DummyComputable(Computable):
    """Simple computable used for testing the Pool.

    Args:
        cell_id (int): Identifier for the dummy computation.
        should_fail (bool, optional): If True, `compute` will raise
            a ValueError instead of returning a result.
    """

    def __init__(self, cell_id: int, should_fail: bool = False):
        self.cell_id = cell_id
        self.should_fail = should_fail

    def compute(self) -> dict[str, Any]:
        """Run a dummy computation.

        Returns:
            dict[str, Any]: Dictionary with `cell_id` and computed value.

        Raises:
            ValueError: If `should_fail` is True.
        """
        if self.should_fail:
            raise ValueError(f"Computation failed for {self.cell_id}")
        return {"cell_id": self.cell_id, "value": self.cell_id * 2}


def test_single_task_success(qtbot):
    """Test Pool correctly executes a single task.

    Asserts:
        The result callback receives the correct dictionary.
        The finish callback is invoked with the task ID.
        The `all_finished` signal is emitted after completion.
    """
    pool = QPool()

    results = []
    finished_called = []

    def on_result(res):
        results.append(res)

    def on_finished(task_id):
        finished_called.append(task_id)

    with qtbot.waitSignal(pool.all_finished, timeout=2000):
        pool.submit(DummyComputable(1), on_result=on_result, on_finish=on_finished)
        pool.run()

    assert len(results) == 1
    assert results[0]["cell_id"] == 1
    assert finished_called


def test_multiple_tasks_success(qtbot):
    """Test Pool correctly executes multiple tasks in parallel.

    Asserts:
        The result callback is called for each task.
        Each finish callback is invoked with the corresponding task ID.
        The `all_finished` signal is emitted once all tasks complete.
    """
    pool = QPool()

    results = []
    finished_called = []

    def on_result(res):
        results.append(res)

    def on_finished(task_id):
        finished_called.append(task_id)

    with qtbot.waitSignal(pool.all_finished, timeout=2000):
        for i in range(3):
            pool.submit(DummyComputable(i), on_result=on_result, on_finish=on_finished)
        pool.run()

    assert len(results) == 3
    assert set(r["cell_id"] for r in results) == {0, 1, 2}
    assert len(finished_called) == 3


def test_task_failure_triggers_error(qtbot):
    """Test Pool correctly propagates computation errors.

    Asserts:
        The error callback is called with the raised exception.
        The result callback is not invoked when computation fails.
        The finish callback is still invoked for the task.
        The `all_finished` signal is emitted after error handling.
    """
    pool = QPool()

    results = []
    errors = []
    finished_called = []

    def on_result(res):
        results.append(res)

    def on_error(task_id, exc):
        errors.append((task_id, exc))

    def on_finished(task_id):
        finished_called.append(task_id)

    with qtbot.waitSignal(pool.all_finished, timeout=2000):
        pool.submit(
            DummyComputable(1, should_fail=True),
            on_result=on_result,
            on_error=on_error,
            on_finish=on_finished,
        )
        pool.run()

    assert not results
    assert errors
    assert finished_called


def test_all_finished_emitted_once(qtbot):
    """Test Pool emits `all_finished` exactly once.

    Asserts:
        The `all_finished` signal is emitted only once, even when
        multiple tasks are submitted to the pool.
    """
    pool = QPool()

    finished_signal_count = 0

    def on_all_finished():
        nonlocal finished_signal_count
        finished_signal_count += 1

    pool.all_finished.connect(on_all_finished)

    with qtbot.waitSignal(pool.all_finished, timeout=2000):
        for i in range(5):
            pool.submit(DummyComputable(i), on_result=lambda _: None)
        pool.run()

    assert finished_signal_count == 1


def test_pool_cancellation(qtbot):
    """Test that QPool cooperatively cancels running tasks.

    Asserts:
        Tasks running before cancellation are skipped after flag is set.
        No results are emitted after cancellation.
        The finish callback is invoked for all submitted tasks.
        The `all_finished` signal is emitted once after cancellation.
        The internal `tasks` list is cleared after all finishes.
    """
    pool = QPool()
    results = []
    finished_called = []

    def on_result(res):
        results.append(res)

    def on_finished(task_id):
        finished_called.append(task_id)

    # Submit and cancel before running
    for i in range(3):
        pool.submit(DummyComputable(i), on_result=on_result, on_finish=on_finished)
    pool.cancel()

    with qtbot.waitSignal(pool.all_finished, timeout=2000):
        pool.run()

    assert pool.cancel_flag.is_set()
    assert not results
    assert len(finished_called) == 3
    assert not pool.tasks
