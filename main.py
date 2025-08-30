from PyQt6 import QtWidgets

from pycroglia.ui.widgets.wizard.config import DEFAULT_CONFIG
from pycroglia.ui.widgets.wizard.wizard import ConfigurableMainStack

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    wizard = ConfigurableMainStack(config=DEFAULT_CONFIG)
    wizard.setWindowTitle("Main Window")
    screen = app.primaryScreen().availableGeometry()
    wizard.resize(int(screen.width() * 0.75), int(screen.height() * 0.75))
    wizard.show()
    app.exec()
