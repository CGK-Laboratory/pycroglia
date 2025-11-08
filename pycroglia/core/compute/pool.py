from collections.abc import Callable
from typing import Any
from pycroglia.core.compute.backend import (
    Backend,
    Task,
)  # assuming you defined the Protocols


class Pool:
    """Unified Pool that delegates to a specific backend.

    This class acts as a façade: it exposes a single API (`submit`, `run`,
    `join`) while delegating execution to the provided backend (e.g. QPool,
    MPPool).

    Attributes:
        backend (Backend): The backend used for task execution.
    """

    def __init__(self, backend: Backend) -> None:
        """Initialize the Pool.

        Args:
            backend (Backend): An instance of a backend that implements
                the required API (`submit`, `run`, `join`).
        """
        self.backend = backend

    def submit(
        self,
        task: Task,
        on_result: Callable[[dict[str, Any]], None],
        on_error: Callable[[str, Exception], None] | None = None,
        on_finish: Callable[[str], None] | None = None,
    ) -> None:
        """Submit a task to the backend.

        Args:
            task (Task): Task to execute.
            on_result (Callable[[dict[str, Any]], None]): Callback for results.
            on_error (Callable[[str, Exception], None], optional): Callback for errors.
            on_finish (Callable[[str], None], optional): Callback when the task finishes.
        """
        return self.backend.submit(task, on_result, on_error, on_finish)

    def run(self) -> None:
        """Start execution of all submitted tasks."""
        return self.backend.run()

    def cancel(self) -> None:
        """Cancels execution of all submitted tasks."""
        return self.backend.cancel()

    def join(self) -> None:
        """Block until all tasks are finished."""
        return self.backend.join()
