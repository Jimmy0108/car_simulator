from gui import PlannerGui
from PyQt6.QtWidgets import QApplication


if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName('Hybrid A* GUI')
    win = PlannerGui()
    win.show()
    app.exec()
