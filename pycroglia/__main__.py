import os
from PyQt6 import QtWidgets, QtGui

from pycroglia.ui.widgets.wizard.config import DEFAULT_CONFIG
from pycroglia.ui.widgets.wizard.wizard import ConfigurableMainStack

def main():
    app = QtWidgets.QApplication([])
    
    icon_path = os.path.join(os.path.dirname(__file__), "assets", "logo.svg")
    if os.path.exists(icon_path):
        app.setWindowIcon(QtGui.QIcon(icon_path))
        
    wizard = ConfigurableMainStack(config=DEFAULT_CONFIG)
    wizard.setWindowTitle("Pycroglia — 3D Cell Morphology Analyzer")
    screen = app.primaryScreen().availableGeometry()
    wizard.resize(int(screen.width() * 0.75), int(screen.height() * 0.75))
    wizard.show()
    app.exec()

if __name__ == "__main__":
    main()
