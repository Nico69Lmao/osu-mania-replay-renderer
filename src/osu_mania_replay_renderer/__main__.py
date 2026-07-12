import multiprocessing as mp
import os
import platform
import sys


def main():
    mp.freeze_support()

    if "--multiprocessing-smoke-test" in sys.argv:
        from osu_mania_replay_renderer.renderer import run_process_pool_smoke_test

        if not run_process_pool_smoke_test():
            raise SystemExit(1)

        print("multiprocessing smoke test passed")
        return

    if "--gpu-smoke-test" in sys.argv:
        from osu_mania_replay_renderer.gpu_compositor import gpu_smoke_test

        print(f"gpu smoke test passed: {gpu_smoke_test()}")
        return

    if platform.system().lower() == "linux":
        os.environ.setdefault("QT_QPA_PLATFORMTHEME", "xdgdesktopportal")

    from PySide6.QtWidgets import QApplication
    from osu_mania_replay_renderer.gui import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
