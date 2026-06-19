import sys
import multiprocessing as mp
from PySide6.QtWidgets import QApplication
from gui import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    mp.freeze_support()
    main()