from PyQt6 import QtWidgets

from pycroglia.ui.widgets.main_stack import MainStack

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    wizard = MainStack()
    wizard.setWindowTitle("Main Window")
    screen = app.primaryScreen().availableGeometry()
    wizard.resize(int(screen.width() * 0.75), int(screen.height() * 0.75))
    wizard.show()
    app.exec()
