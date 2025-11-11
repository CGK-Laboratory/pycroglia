from PyQt6 import QtWidgets, QtCore
from typing import Optional, Iterable, Type, List

from pycroglia.core.io.output import OutputWriter


class OutputWriterSelector(QtWidgets.QWidget):
    """Widget for selecting one or more output writer classes.

    Attributes:
        _writers (List[Type[OutputWriter]]): List of available OutputWriter classes.
        title_text (str): Text displayed as the widget's title.
        title_label (QtWidgets.QLabel): Label widget for the title.
        list (QtWidgets.QListWidget): List widget displaying available writers.
    """

    itemChanged = QtCore.pyqtSignal()

    def __init__(
        self,
        writers: Iterable[Type[OutputWriter]],
        title_text: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the output writer selector widget.

        Args:
            writers (Iterable[Type[OutputWriter]]): Iterable of OutputWriter classes to display.
            title_text (str): Text for the title label.
            parent (Optional[QtWidgets.QWidget]): Optional parent widget.
        """
        super().__init__(parent)

        # Properties
        self._writers: List[Type[OutputWriter]] = list(writers)

        # Configuration
        self.title_text = title_text

        # Widgets
        self.title_label = QtWidgets.QLabel(self.title_text, parent=self)
        self.list = QtWidgets.QListWidget(self)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.list)
        self.setLayout(layout)

        # Setup
        self.set_writers(self._writers)
        self.list.itemChanged.connect(self._on_item_changed)

    def _on_item_changed(self):
        self.itemChanged.emit()

    def set_writers(self, writers: Iterable[Type[OutputWriter]]):
        """Set the list of available OutputWriter classes.

        Args:
            writers (Iterable[Type[OutputWriter]]): Iterable of OutputWriter classes to display.
        """
        self.list.clear()
        self._writers = list(writers)

        for w in self._writers:
            item = QtWidgets.QListWidgetItem(w.get_name(), self.list)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def get_selected_writers(self) -> List[Type[OutputWriter]]:
        """Return a list of selected OutputWriter classes.

        Returns:
            List[Type[OutputWriter]]: List of selected OutputWriter classes.
        """
        selected = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                selected.append(self._writers[i])

        return selected

    def has_selected_writers(self) -> bool:
        return len(self.get_selected_writers()) > 0
