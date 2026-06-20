import multiprocessing as mp
import sys

from PySide6.QtWidgets import QApplication

from osu_mania_replay_renderer.gui import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    mp.freeze_support()
    main()
