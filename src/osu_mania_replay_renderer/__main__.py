import multiprocessing as mp
import sys


def main():
    mp.freeze_support()

    if "--multiprocessing-smoke-test" in sys.argv:
        from osu_mania_replay_renderer.renderer import run_process_pool_smoke_test

        if not run_process_pool_smoke_test():
            raise SystemExit(1)

        print("multiprocessing smoke test passed")
        return

    from PySide6.QtWidgets import QApplication
    from osu_mania_replay_renderer.gui import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
