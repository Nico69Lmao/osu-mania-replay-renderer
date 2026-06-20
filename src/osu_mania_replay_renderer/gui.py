from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QComboBox, QProgressBar, QMessageBox, QDoubleSpinBox,
    QSpinBox
)
from PySide6.QtCore import QThread, Signal

from osu_mania_replay_renderer.osu_finder import list_skins, find_beatmap_from_replay, get_replay_info
from osu_mania_replay_renderer.renderer import render_video
from osu_mania_replay_renderer.settings import load_settings, update_setting


class RenderThread(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, osu_file, skin_folder, output_file, replay_file, scroll_speed, resolution, motion_blur):
        super().__init__()
        self.osu_file = osu_file
        self.skin_folder = skin_folder
        self.output_file = output_file
        self.replay_file = replay_file
        self.scroll_speed = scroll_speed
        self.resolution = resolution
        self.motion_blur = motion_blur

    def run(self):
        try:
            render_video(
                osu_file=self.osu_file,
                skin_folder=self.skin_folder,
                output_file=self.output_file,
                replay_file=self.replay_file,
                scroll_speed_value=self.scroll_speed,
                resolution=self.resolution,
                motion_blur=self.motion_blur,
                progress_callback=self.progress.emit,
            )
            self.finished_ok.emit(self.output_file)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("osu!mania Local Renderer")
        self.setMinimumWidth(680)

        self.settings = load_settings()

        self.osu_folder = self.settings["osu_folder"] or None
        self.replay_file = self.settings["last_replay"] or None
        self.beatmap_file = self.settings["last_beatmap"] or None
        self.skin_folder = None

        root = QWidget()
        layout = QVBoxLayout(root)

        self.osu_label = QLabel(f"osu! folder: {self.osu_folder}" if self.osu_folder else "osu! folder: not selected")
        self.replay_label = QLabel(f"Replay: {self.replay_file}" if self.replay_file else "Replay: not selected")
        self.beatmap_label = QLabel(f"Beatmap: {self.beatmap_file}" if self.beatmap_file else "Beatmap: not found")
        self.info_label = QLabel("Replay info: -")

        self.skin_combo = QComboBox()
        self.skin_combo.addItem("No skin found")

        self.scroll_speed_spin = QDoubleSpinBox()
        self.scroll_speed_spin.setRange(1.0, 40.0)
        self.scroll_speed_spin.setSingleStep(0.5)
        self.scroll_speed_spin.setDecimals(1)
        self.scroll_speed_spin.setValue(float(self.settings.get("scroll_speed", "30.0")))

        self.motion_blur_spin = QSpinBox()
        self.motion_blur_spin.setRange(0, 8)
        self.motion_blur_spin.setSingleStep(1)
        self.motion_blur_spin.setValue(int(self.settings.get("motion_blur", "0")))

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "640x360",
            "854x480",
            "960x540",
            "1024x576",
            "1280x720",
            "1920x1080",
            "2560x1440",
            "3840x2160",
        ])
        
        index = self.resolution_combo.findText(self.settings.get("resolution", "1280x720"))

        if index >= 0:
            self.resolution_combo.setCurrentIndex(index)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress_label = QLabel("Frame: - | ETA: -")

        btn_osu = QPushButton("Select osu! folder")
        btn_replay = QPushButton("Select .osr replay")
        btn_manual_beatmap = QPushButton("Select beatmap manually")
        btn_render = QPushButton("Render video")

        btn_osu.clicked.connect(self.select_osu_folder)
        btn_replay.clicked.connect(self.select_replay)
        btn_manual_beatmap.clicked.connect(self.select_manual_beatmap)
        btn_render.clicked.connect(self.start_render)

        layout.addWidget(self.osu_label)
        layout.addWidget(btn_osu)
        layout.addWidget(self.replay_label)
        layout.addWidget(btn_replay)
        layout.addWidget(self.info_label)
        layout.addWidget(self.beatmap_label)
        layout.addWidget(btn_manual_beatmap)
        layout.addWidget(QLabel("Skin:"))
        layout.addWidget(self.skin_combo)
        layout.addWidget(QLabel("osu!mania scroll speed:"))
        layout.addWidget(self.scroll_speed_spin)
        layout.addWidget(QLabel("Motion blur:"))
        layout.addWidget(self.motion_blur_spin)
        layout.addWidget(QLabel("Resolution:"))
        layout.addWidget(self.resolution_combo)
        layout.addWidget(self.progress)
        layout.addWidget(self.progress_label)
        layout.addWidget(btn_render)

        self.load_saved_skins()

        self.skin_combo.currentTextChanged.connect(lambda v: update_setting("last_skin", v))
        self.scroll_speed_spin.valueChanged.connect(lambda v: update_setting("scroll_speed", str(v)))
        self.motion_blur_spin.valueChanged.connect(lambda v: update_setting("motion_blur", str(v)))
        self.resolution_combo.currentTextChanged.connect(lambda v: update_setting("resolution", v))

        self.setCentralWidget(root)

        if self.replay_file:
            self.load_replay_info()

    def load_saved_skins(self):
        if not self.osu_folder:
            return

        skins = list_skins(self.osu_folder)
        self.skin_combo.clear()

        if skins:
            self.skin_combo.addItems(skins)
            last_skin = self.settings["last_skin"]
            index = self.skin_combo.findText(last_skin)

            if index >= 0:
                self.skin_combo.setCurrentIndex(index)
        else:
            self.skin_combo.addItem("No skin found")

    def select_osu_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select osu! folder", self.osu_folder or str(Path.home()))

        if not folder:
            return

        self.osu_folder = folder
        update_setting("osu_folder", folder)
        self.osu_label.setText(f"osu! folder: {folder}")
        self.load_saved_skins()
        self.try_auto_find_beatmap()

    def select_replay(self):
        start_dir = str(Path(self.replay_file).parent) if self.replay_file else str(Path.home())
        file, _ = QFileDialog.getOpenFileName(self, "Select osu! replay", start_dir, "osu! replay (*.osr)")

        if not file:
            return

        self.replay_file = file
        update_setting("last_replay", file)
        self.replay_label.setText(f"Replay: {file}")
        self.load_replay_info()
        self.try_auto_find_beatmap()

    def load_replay_info(self):
        try:
            info = get_replay_info(self.replay_file)
            self.info_label.setText(
                f"Player: {info['username']} | Score: {info['score']} | Combo: {info['max_combo']} | Mods: {info['mods']}"
            )
        except Exception as e:
            self.info_label.setText("Replay info: failed to read replay")
            QMessageBox.warning(self, "Replay error", str(e))

    def try_auto_find_beatmap(self):
        if not self.osu_folder or not self.replay_file:
            return

        self.beatmap_label.setText("Searching beatmap...")
        found = find_beatmap_from_replay(self.osu_folder, self.replay_file)

        if found:
            self.beatmap_file = found
            update_setting("last_beatmap", found)
            self.beatmap_label.setText(f"Beatmap found: {found}")
        else:
            self.beatmap_file = None
            update_setting("last_beatmap", "")
            self.beatmap_label.setText("Beatmap not found. Select it manually.")

    def select_manual_beatmap(self):
        start_dir = str(Path(self.beatmap_file).parent) if self.beatmap_file else str(Path.home())
        file, _ = QFileDialog.getOpenFileName(self, "Select .osu beatmap", start_dir, "osu! beatmap (*.osu)")

        if not file:
            return

        self.beatmap_file = file
        update_setting("last_beatmap", file)
        self.beatmap_label.setText(f"Manual beatmap: {file}")

    def start_render(self):
        if not self.beatmap_file:
            QMessageBox.warning(self, "Error", "Select a beatmap first.")
            return

        output_start = self.settings["last_output_folder"] or str(Path.home())
        output_file, _ = QFileDialog.getSaveFileName(self, "Save video", str(Path(output_start) / "render.mp4"), "MP4 video (*.mp4)")

        if not output_file:
            return

        if not output_file.endswith(".mp4"):
            output_file += ".mp4"

        update_setting("last_output_folder", str(Path(output_file).parent))

        selected_skin = self.skin_combo.currentText()

        if self.osu_folder and selected_skin != "No skin found":
            self.skin_folder = str(Path(self.osu_folder) / "Skins" / selected_skin)
            update_setting("last_skin", selected_skin)
        else:
            self.skin_folder = None

        scroll_speed = self.scroll_speed_spin.value()
        resolution = self.resolution_combo.currentText()
        motion_blur = self.motion_blur_spin.value()

        update_setting("scroll_speed", str(scroll_speed))
        update_setting("motion_blur", str(motion_blur))
        update_setting("resolution", resolution)

        self.progress.setValue(0)
        self.progress_label.setText("Preparing render...")

        self.thread = RenderThread(
            self.beatmap_file,
            self.skin_folder,
            output_file,
            self.replay_file,
            scroll_speed,
            resolution,
            motion_blur,
        )

        self.thread.progress.connect(self.update_progress)
        self.thread.finished_ok.connect(self.render_done)
        self.thread.failed.connect(self.render_failed)
        self.thread.start()

    def update_progress(self, percent, text):
        self.progress.setValue(percent)
        self.progress_label.setText(text)

    def render_done(self, output_file):
        QMessageBox.information(self, "Done", f"Video created:\n{output_file}")

    def render_failed(self, error):
        QMessageBox.critical(self, "Render error", error)
