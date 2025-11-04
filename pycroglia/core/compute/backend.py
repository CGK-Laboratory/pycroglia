from typing import Protocol, Any
from collections.abc import Callable


class Task(Protocol):
    """Protocol for executable tasks.

    Any class implementing this protocol must define a `run`
    method that returns a dictionary with computation results.
    """

    def run(self) -> dict[str, Any]:
        """Execute the task and return its result.

        Returns:
            dict[str, Any]: Computation result as a dictionary.
        """
        ...


class Backend(Protocol):
    """Protocol for computation backends.

    Defines a common interface for running tasks, regardless of
    whether they are executed via PyQt, multiprocessing, or another
    concurrency model.
    """

    def submit(
        self,
        task: Task,
        on_result: Callable[[dict[str, Any]], None],
        on_error: Callable[[str, Exception], None] | None = None,
        on_finish: Callable[[str], None] | None = None,
    ) -> None:
        """Submit a task to the backend.

        Args:
            task (Task): Task to be executed.
            on_result (Callable): Callback for successful results.
            on_error (Callable | None): Optional callback for errors.
            on_finish (Callable | None): Optional callback when task finishes.
        """
        ...

    def run(self) -> None:
        """Start execution of all submitted tasks."""
        ...

    def join(self) -> None:
        """Block until all submitted tasks are finished."""
        ...

    def cancel(self) -> None:
        """Cancels execution of all submitted tasks."""
        ...
