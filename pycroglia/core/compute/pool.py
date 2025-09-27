from collections.abc import Callable
from typing import Any
import uuid
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot
from pycroglia.core.compute.computable import Computable

class TaskSignals(QObject):
    """Signals available from a running Task.

    Attributes:
        result (pyqtSignal): Emitted with the computation result as a dict.
        error (pyqtSignal): Emitted with (task_id, exception) if computation fails.
        finished (pyqtSignal): Emitted with task_id when the task completes.
    """
    result = pyqtSignal(object)
    error = pyqtSignal(str, Exception)
    finished = pyqtSignal(str)    
    
class Task(QRunnable):
    """A runnable task that executes a Computable object in a thread.

    Attributes:
        task_id (str): Unique identifier for this task.
        computable (Computable): The computation object to run.
        signals (TaskSignals): Signal manager for result, error, and finished.
    """
    def __init__(self, computable: Computable) -> None:
        """Initialize a new Task.

        Args:
            computable (Computable): The object implementing the compute() method.
        """
        super().__init__()
        self.task_id = uuid.uuid4().hex
        self.computable = computable
        self.signals = TaskSignals()

    @pyqtSlot()    
    def run(self):
        """Execute the task.

        Runs the `compute` method of the associated Computable.
        Emits:
            - result: when computation succeeds.
            - error: if an exception is raised.
            - finished: always, when the task ends.
        """
        try:
            result = self.computable.compute()
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(self.task_id, e)
        finally:
            self.signals.finished.emit(self.task_id)
        

class Pool(QObject):
    """Thread pool manager for executing Computable tasks.

    Provides submission, execution, and completion tracking
    for multiple concurrent tasks.

    Attributes:
        all_finished (pyqtSignal): Emitted when all submitted tasks have completed.
    """

    all_finished = pyqtSignal()
    
    def __init__(self) -> None:
        """Initialize the Pool."""
        super().__init__()
        self.threadpool = QThreadPool()
        self.tasks = []
        self.pending = 0

        
    def submit(self,
               computable: Computable,
               on_result: Callable[[dict[str, Any]], None],
               on_error: Callable[[str, Exception], None] | None = None,
               on_finish: Callable[[str], None] | None = None):
        """Submit a Computable task to the pool.

        Args:
            computable (Computable): The computation object to execute.
            on_result (Callable[[dict[str, Any]], None]): Callback for result data.
            on_error (Callable[[str, Exception], None], optional): Callback for errors.
            on_finish (Callable[[str], None], optional): Callback when task finishes.
        """
        task = Task(computable)
        task.signals.result.connect(on_result)
        task.signals.finished.connect(self._on_task_finished(on_finish))
        if on_error:
            task.signals.error.connect(on_error)
        self.tasks.append(task)
        self.pending += 1
        
    def run(self) -> None:
        """Start all submitted tasks asynchronously."""
        for task in self.tasks:
            self.threadpool.start(task)
        
    def join(self) -> None:
        """Block until all tasks are finished.

        Warning:
            This method blocks the main (GUI) thread if called inside
            a running Qt event loop. Prefer connecting to `all_finished`
            for a non-blocking alternative.
        """
        self.threadpool.waitForDone()

    def _on_task_finished(self, callback :Callable[[str], None] | None = None) -> Callable[[str], None]:
        """Create a slot to handle task completion.

        Args:
            callback (Callable[[str], None], optional): A user callback
                invoked with the task_id when the task completes.

        Returns:
            Callable[[str], None]: Slot function to connect to the finished signal.
        """        
        def on_finish(task_id: str):
            if callback:
                callback(task_id)
            self.pending -= 1
            if self.pending == 0:
                self.all_finished.emit()
                self.tasks.clear()
        return on_finish
