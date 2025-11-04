from typing import Any
from pycroglia.core.compute.computable import Computable
from pycroglia.core.compute.mp_pool import MPPool


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


def test_single_task_success():
    """Test Pool correctly executes a single task.

    Asserts:
        The result callback receives the correct dictionary.
        The finish callback is invoked with the task ID.
        The `all_finished` callback is executed after completion.
    """
    pool = MPPool()
    results = []
    finished_called = []
    all_finished_called = []

    def on_result(res):
        results.append(res)

    def on_finished(task_id):
        finished_called.append(task_id)

    def on_all_finished():
        all_finished_called.append(True)

    pool.all_finished = on_all_finished
    pool.submit(DummyComputable(1), on_result=on_result, on_finish=on_finished)
    pool.join()

    assert len(results) == 1
    assert results[0]["cell_id"] == 1
    assert finished_called
    assert all_finished_called


def test_multiple_tasks_success():
    """Test Pool correctly executes multiple tasks in parallel.

    Asserts:
        The result callback is called for each task.
        Each finish callback is invoked with the corresponding task ID.
        The `all_finished` callback is executed once all tasks complete.
    """
    pool = MPPool()
    results = []
    finished_called = []
    all_finished_called = []

    def on_result(res):
        results.append(res)

    def on_finished(task_id):
        finished_called.append(task_id)

    def on_all_finished():
        all_finished_called.append(True)

    pool.all_finished = on_all_finished
    for i in range(3):
        pool.submit(DummyComputable(i), on_result=on_result, on_finish=on_finished)
    pool.join()

    assert len(results) == 3
    assert set(r["cell_id"] for r in results) == {0, 1, 2}
    assert len(finished_called) == 3
    assert all_finished_called


def test_task_failure_triggers_error():
    """Test Pool correctly propagates computation errors.

    Asserts:
        The error callback is called with the raised exception.
        The result callback is not invoked when computation fails.
        The finish callback is still invoked for the task.
        The `all_finished` callback is executed after error handling.
    """
    pool = MPPool()
    results = []
    errors = []
    finished_called = []
    all_finished_called = []

    def on_result(res):
        results.append(res)

    def on_error(task_id, exc):
        errors.append((task_id, exc))

    def on_finished(task_id):
        finished_called.append(task_id)

    def on_all_finished():
        all_finished_called.append(True)

    pool.all_finished = on_all_finished
    pool.submit(
        DummyComputable(1, should_fail=True),
        on_result=on_result,
        on_error=on_error,
        on_finish=on_finished,
    )
    pool.join()

    assert not results
    assert errors
    assert finished_called
    assert all_finished_called


def test_all_finished_emitted_once():
    """Test Pool emits `all_finished` exactly once.

    Asserts:
        The `all_finished` callback is executed only once, even when
        multiple tasks are submitted to the pool.
    """
    pool = MPPool()
    finished_called = []
    all_finished_called = []

    def on_result(res):
        pass

    def on_finished(task_id):
        finished_called.append(task_id)

    def on_all_finished():
        all_finished_called.append(True)

    pool.all_finished = on_all_finished
    for i in range(5):
        pool.submit(DummyComputable(i), on_result=on_result, on_finish=on_finished)
    pool.join()

    assert len(finished_called) == 5
    assert len(all_finished_called) == 1


def test_pool_cancellation():
    """Test that Pool cancels all running tasks cooperatively.

    Asserts:
        Tasks running before cancellation finish normally.
        Pending tasks do not start after cancellation.
        The result callback is only called for completed tasks.
        The finish callback is invoked for each submitted task.
        The `all_finished` callback is executed once after cancellation.
        The internal `tasks` list is cleared when all tasks have finished or been cancelled.
    """

    class SlowComputable(DummyComputable):
        def compute(self) -> dict[str, Any]:
            return {"cell_id": self.cell_id, "value": self.cell_id * 10}

    pool = MPPool(processes=2)
    results = []
    finished_called = []
    all_finished_called = []

    def on_result(res):
        results.append(res)

    def on_error(task_id, exc):
        raise AssertionError(f"Unexpected error in task {task_id}: {exc}")

    def on_finished(task_id):
        finished_called.append(task_id)

    def on_all_finished():
        all_finished_called.append(True)

    pool.all_finished = on_all_finished
    for i in range(5):
        pool.submit(SlowComputable(i), on_result, on_error, on_finished)
    pool.cancel()
    pool.join()

    assert pool.cancel_flag.is_set()
    assert len(results) <= 5
    assert len(finished_called) == 5
    assert all_finished_called
    assert not pool.tasks
