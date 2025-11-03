import uuid
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from collections.abc import Callable
from typing import Any
from pycroglia.core.compute.computable import Computable


class CancelFlag:
    """Lightweight cooperative cancellation flag for multiprocessing tasks.

    This class provides a simple shared mechanism to signal cancellation
    across multiple worker processes using a `multiprocessing.Manager.Value`.
    Each `Computable` should periodically check the flag to terminate gracefully.

    Example:
        ```python
        flag = CancelFlag()
        if flag.is_set():
            return  # Stop early
        ```

    Attributes:
        _cancelled (mp.Value): Shared boolean indicating cancellation state.
    """

    def __init__(self) -> None:
        """Initialize the shared cancellation flag."""
        self._cancelled = mp.Value("b", False)

    def set(self) -> None:
        """Signal all workers to cancel cooperatively."""
        self._cancelled.value = True

    def is_set(self) -> bool:
        """Check whether cancellation has been requested.

        Returns:
            bool: True if cancellation has been requested, False otherwise.
        """
        return self._cancelled.value


class MPTask:
    """A multiprocessing-compatible task that executes a Computable.

    Attributes:
        task_id (str): Unique identifier for this task.
        computable (Computable): The computation object to run.
        cancel_flag (CancelFlag): Shared cancellation flag.
    """

    def __init__(self, computable: Computable, cancel_flag: CancelFlag) -> None:
        """Initialize the multiprocessing task.

        Args:
            computable (Computable): The computation object to execute.
            cancel_flag (CancelFlag): Shared cancellation flag.
        """
        self.task_id = uuid.uuid4().hex
        self.computable = computable
        self.cancel_flag = cancel_flag

    def run(self):
        """Execute the computable and return its result.

        The Computable is responsible for checking `cancel_flag.is_set()`
        periodically and exiting early if cancellation is requested.

        Returns:
            dict[str, Any]: Computation result dictionary.
        """
        if self.cancel_flag.is_set():
            return {"__cancelled__": True, "task_id": self.task_id}
        try:
            return self.computable.compute()
        except Exception as e:
            raise e


class MPPool:
    """Multiprocessing pool manager for executing Computable tasks.

    Provides submission, execution, and completion tracking
    for multiple concurrent tasks.

    Attributes:
        all_finished (Callable[[], None] | None): Optional callback
            invoked when all submitted tasks have completed.
    """

    def __init__(self, processes: int | None = None) -> None:
        """Initialize the pool.

        Args:
            processes (int | None): Number of worker processes.
                Defaults to os.cpu_count() if None.
        """
        self.pool = ThreadPool(processes=processes)
        self.tasks: list[MPTask] = []
        self.cancel_flag = CancelFlag()
        self.pending: int = 0
        self.all_finished: Callable[[], None] | None = None

    def submit(
        self,
        computable: Computable,
        on_result: Callable[[dict[str, Any]], None],
        on_error: Callable[[str, Exception], None] | None = None,
        on_finish: Callable[[str], None] | None = None,
    ) -> None:
        """Submit a Computable task to the pool.

        Args:
            computable (Computable): The computation object to execute.
            on_result (Callable[[dict[str, Any]], None]): Callback for result data.
            on_error (Callable[[str, Exception], None], optional): Callback for errors.
            on_finish (Callable[[str], None], optional): Callback when task finishes.
        """
        task = MPTask(computable, self.cancel_flag)

        def callback(result: dict[str, Any]) -> None:
            try:
                is_cancelled = result.get("__cancelled__")
                if (not self.cancel_flag.is_set()) and (not is_cancelled):
                    on_result(result)
            finally:
                if on_finish:
                    on_finish(task.task_id)
                self._decrement_pending()

        def error_callback(err: Exception) -> None:
            if on_error:
                on_error(task.task_id, err)
            if on_finish:
                on_finish(task.task_id)
            self._decrement_pending()

        self.tasks.append(task)
        self.pending += 1
        self.pool.apply_async(
            task.run, callback=callback, error_callback=error_callback
        )

    def run(self) -> None:
        """API symmetry: in multiprocessing tasks start immediately."""
        pass

    def join(self) -> None:
        """Block until all tasks are finished."""
        self.pool.close()
        self.pool.join()

    def cancel(self) -> None:
        """Request cooperative cancellation for all running tasks.

        Sets the shared cancellation flag. All tasks that periodically
        check it will terminate gracefully. Pending tasks are ignored.
        """
        self.cancel_flag.set()

    def _decrement_pending(self) -> None:
        """Track task completion and trigger all_finished if done."""
        self.pending -= 1
        if self.pending == 0 and self.all_finished:
            self.all_finished()
            self.tasks.clear()
