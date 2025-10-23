from typing import List, Optional
from PyQt6 import QtWidgets, QtCore


class GraphSelectionWidget(QtWidgets.QWidget):
    """Widget to select which graphs to display and visualize each selected graph using checkboxes.

    Attributes:
        label (QLabel): Label describing the widget.
        checkboxes (List[QCheckBox]): List of checkboxes for available graphs.
        button (QPushButton): Button to visualize the selected graphs.
    """

    DEFAULT_BUTTON_TEXT = "Preview"
    DEFAULT_LABEL_TEXT = "Select graphs to display:"

    buttonClicked = QtCore.pyqtSignal(list)

    def __init__(
        self,
        graphs_list: List[str],
        label_txt: Optional[str] = None,
        button_txt: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the graph selection widget.

        Args:
            graphs_list (List[str]): List of available graph names.
            label_txt (Optional[str]): Custom label text.
            button_txt (Optional[str]): Custom button text.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        self.label_txt = label_txt or self.DEFAULT_LABEL_TEXT
        self.button_txt = button_txt or self.DEFAULT_BUTTON_TEXT

        self.label = QtWidgets.QLabel(self.label_txt)

        # Create checkboxes for each graph
        self.checkboxes = [
            QtWidgets.QCheckBox(name, parent=self) for name in graphs_list
        ]
        self.button = QtWidgets.QPushButton(self.button_txt, parent=self)

        # Connections
        self.button.clicked.connect(self._on_button_clicked)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        for cb in self.checkboxes:
            layout.addWidget(cb)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def get_selected_graphs(self) -> List[str]:
        """Get the list of currently selected graphs.

        Returns:
            List[str]: Names of selected graphs.
        """
        return [cb.text() for cb in self.checkboxes if cb.isChecked()]

    def _on_button_clicked(self):
        """Handle the preview button click.

        Collects the currently selected graph names and emits them via the
        `buttonClicked` signal so external code can react to a preview request.

        Returns:
            None
        """
        list_of_graphs = self.get_selected_graphs()
        self.buttonClicked.emit(list_of_graphs)
