import uuid
import multiprocessing as mp
from collections.abc import Callable
from typing import Any
from pycroglia.core.compute.computable import Computable


class MPTask:
    """A multiprocessing-compatible task that executes a Computable.

    Attributes:
        task_id (str): Unique identifier for this task.
        computable (Computable): The computation object to run.
    """

    def __init__(self, computable: Computable) -> None:
        self.task_id = uuid.uuid4().hex
        self.computable = computable

    def run(self) -> dict[str, Any]:
        """Execute the computable and return its result."""
        return self.computable.compute()


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
        self.pool = mp.Pool(processes=processes)
        self.tasks: list[MPTask] = []
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
        task = MPTask(computable)

        def callback(result: dict[str, Any]) -> None:
            try:
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

    def _decrement_pending(self) -> None:
        """Track task completion and trigger all_finished if done."""
        self.pending -= 1
        if self.pending == 0 and self.all_finished:
            self.all_finished()
            self.tasks.clear()
