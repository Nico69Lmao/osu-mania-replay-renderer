from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from osu_mania_replay_renderer.osu_finder import list_skins, find_beatmap_from_replay, get_replay_info
from osu_mania_replay_renderer.renderer import render_video
from osu_mania_replay_renderer.settings import load_settings, update_setting


def setting_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in ("1", "true", "yes", "on")


class RenderThread(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, osu_file, skin_folder, output_file, replay_file, options):
        super().__init__()
        self.osu_file = osu_file
        self.skin_folder = skin_folder
        self.output_file = output_file
        self.replay_file = replay_file
        self.options = options

    def run(self):
        try:
            render_video(
                osu_file=self.osu_file,
                skin_folder=self.skin_folder,
                output_file=self.output_file,
                replay_file=self.replay_file,
                progress_callback=self.progress.emit,
                **self.options,
            )
            self.finished_ok.emit(self.output_file)
        except Exception as error:
            self.failed.emit(str(error))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("osu!mania Replay Renderer")
        self.setMinimumSize(880, 650)

        self.settings = load_settings()
        self.osu_folder = self.settings["osu_folder"] or None
        self.replay_file = self.settings["last_replay"] or None
        self.beatmap_file = self.settings["last_beatmap"] or None
        self.skin_folder = None
        self.thread = None

        self._build_ui()
        self._apply_style()
        self.load_saved_skins()

        if self.replay_file:
            self.load_replay_info()

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(24, 20, 24, 22)
        root_layout.setSpacing(16)

        title = QLabel("osu!mania Replay Renderer")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Render local osu!mania replays with legacy skin support")
        subtitle.setObjectName("pageSubtitle")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(16)
        content.addWidget(self._build_source_group(), 3)
        content.addWidget(self._build_options_group(), 2)
        root_layout.addLayout(content, 1)
        root_layout.addWidget(self._build_render_group())

        self.setCentralWidget(root)

    def _path_row(self, label, button_text, icon, callback):
        row = QVBoxLayout()
        row.setSpacing(6)
        path_label = QLabel(label)
        path_label.setObjectName("pathLabel")
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button = QPushButton(button_text)
        button.setIcon(self.style().standardIcon(icon))
        button.clicked.connect(callback)
        row.addWidget(path_label)
        row.addWidget(button)
        return row, path_label

    def _build_source_group(self):
        group = QGroupBox("Source")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(16, 22, 16, 16)
        layout.setSpacing(14)

        osu_row, self.osu_label = self._path_row(
            self.osu_folder or "No osu! folder selected",
            "Select osu! folder",
            QStyle.SP_DirOpenIcon,
            self.select_osu_folder,
        )
        replay_row, self.replay_label = self._path_row(
            self.replay_file or "No replay selected",
            "Select replay",
            QStyle.SP_FileIcon,
            self.select_replay,
        )
        beatmap_row, self.beatmap_label = self._path_row(
            self.beatmap_file or "No beatmap selected",
            "Select beatmap manually",
            QStyle.SP_FileDialogContentsView,
            self.select_manual_beatmap,
        )

        layout.addLayout(osu_row)
        layout.addLayout(replay_row)

        self.info_label = QLabel("Replay details unavailable")
        self.info_label.setObjectName("infoLabel")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        layout.addLayout(beatmap_row)

        skin_label = QLabel("Skin")
        skin_label.setObjectName("fieldLabel")
        self.skin_combo = QComboBox()
        self.skin_combo.addItem("No skin found")
        self.skin_combo.currentTextChanged.connect(lambda value: update_setting("last_skin", value))
        layout.addWidget(skin_label)
        layout.addWidget(self.skin_combo)
        layout.addStretch()
        return group

    def _build_options_group(self):
        group = QGroupBox("Render Options")
        layout = QFormLayout(group)
        layout.setContentsMargins(16, 22, 16, 16)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(12)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "640x360", "854x480", "960x540", "1024x576",
            "1280x720", "1920x1080", "2560x1440", "3840x2160",
        ])
        resolution_index = self.resolution_combo.findText(self.settings.get("resolution", "1280x720"))
        self.resolution_combo.setCurrentIndex(max(0, resolution_index))

        self.scroll_speed_spin = QDoubleSpinBox()
        self.scroll_speed_spin.setRange(1.0, 40.0)
        self.scroll_speed_spin.setSingleStep(0.5)
        self.scroll_speed_spin.setDecimals(1)
        self.scroll_speed_spin.setValue(float(self.settings.get("scroll_speed", "30.0")))

        self.motion_blur_spin = QSpinBox()
        self.motion_blur_spin.setRange(0, 8)
        self.motion_blur_spin.setValue(int(self.settings.get("motion_blur", "0")))

        self.vignette_spin = QSpinBox()
        self.vignette_spin.setRange(0, 60)
        self.vignette_spin.setSuffix(" %")
        self.vignette_spin.setValue(int(self.settings.get("vignette_strength", "12")))

        self.results_opacity_spin = QSpinBox()
        self.results_opacity_spin.setRange(20, 100)
        self.results_opacity_spin.setSuffix(" %")
        self.results_opacity_spin.setValue(int(self.settings.get("results_background_opacity", "62")))

        self.results_duration_spin = QDoubleSpinBox()
        self.results_duration_spin.setRange(1.0, 15.0)
        self.results_duration_spin.setSingleStep(0.5)
        self.results_duration_spin.setSuffix(" s")
        self.results_duration_spin.setValue(float(self.settings.get("results_duration", "4.5")))

        self.side_overlay_check = QCheckBox("Side statistics")
        self.side_overlay_check.setChecked(setting_bool(self.settings.get("show_side_overlay", True)))
        self.strain_graph_check = QCheckBox("Strain graph")
        self.strain_graph_check.setChecked(setting_bool(self.settings.get("show_strain_graph", True)))
        self.results_screen_check = QCheckBox("Results screen")
        self.results_screen_check.setChecked(setting_bool(self.settings.get("show_results_screen", True)))
        self.hold_combo_colour_check = QCheckBox("Colour combo during holds")
        self.hold_combo_colour_check.setChecked(setting_bool(self.settings.get("colour_combo_during_holds", True)))

        effects = QVBoxLayout()
        effects.setSpacing(8)
        effects.addWidget(self.side_overlay_check)
        effects.addWidget(self.strain_graph_check)
        effects.addWidget(self.results_screen_check)
        effects.addWidget(self.hold_combo_colour_check)

        layout.addRow("Resolution", self.resolution_combo)
        layout.addRow("Scroll speed", self.scroll_speed_spin)
        layout.addRow("Motion blur", self.motion_blur_spin)
        layout.addRow("Vignette", self.vignette_spin)
        layout.addRow("Results background", self.results_opacity_spin)
        layout.addRow("Results duration", self.results_duration_spin)
        layout.addRow("Overlays", effects)
        return group

    def _build_render_group(self):
        group = QGroupBox("Output")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(16, 22, 16, 16)
        layout.setSpacing(10)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_label = QLabel("Ready")
        self.progress_label.setObjectName("progressLabel")

        self.render_button = QPushButton("Render video")
        self.render_button.setObjectName("primaryButton")
        self.render_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.render_button.clicked.connect(self.start_render)

        layout.addWidget(self.progress)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.render_button)
        return group

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #171819;
                color: #e7e7e7;
                font-family: Inter, Noto Sans, sans-serif;
                font-size: 13px;
            }
            QLabel, QCheckBox { background: transparent; }
            QLabel#pageTitle { font-size: 22px; font-weight: 700; color: #ffffff; }
            QLabel#pageSubtitle { color: #9fa3a7; margin-bottom: 4px; }
            QLabel#pathLabel { color: #b8bcc0; padding: 2px 0; }
            QLabel#fieldLabel, QLabel#progressLabel { color: #b8bcc0; }
            QLabel#infoLabel {
                color: #80d8e8;
                background: #202326;
                border: 1px solid #34383c;
                border-radius: 6px;
                padding: 10px;
            }
            QGroupBox {
                background: #1d1f21;
                border: 1px solid #34373a;
                border-radius: 7px;
                margin-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
                color: #f1f1f1;
            }
            QPushButton, QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 34px;
                background: #292c2f;
                border: 1px solid #41454a;
                border-radius: 6px;
                padding: 0 10px;
            }
            QPushButton:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #64c7d5;
                background: #303438;
            }
            QPushButton:disabled { color: #74787c; background: #242628; }
            QPushButton#primaryButton {
                background: #2e9b68;
                border-color: #38b879;
                color: #ffffff;
                font-weight: 700;
                min-height: 40px;
            }
            QPushButton#primaryButton:hover { background: #35aa73; }
            QComboBox::drop-down { border: 0; width: 28px; }
            QCheckBox { spacing: 8px; min-height: 22px; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid #555a60;
                border-radius: 4px;
                background: #242628;
            }
            QCheckBox::indicator:checked { background: #59bdcc; border-color: #76d7e4; }
            QProgressBar {
                min-height: 10px;
                max-height: 10px;
                border: 0;
                border-radius: 5px;
                background: #303337;
                text-align: center;
            }
            QProgressBar::chunk { border-radius: 5px; background: #59bdcc; }
        """)

    def load_saved_skins(self):
        if not self.osu_folder:
            return

        skins = list_skins(self.osu_folder)
        self.skin_combo.blockSignals(True)
        self.skin_combo.clear()

        if skins:
            self.skin_combo.addItems(skins)
            index = self.skin_combo.findText(self.settings.get("last_skin", ""))

            if index >= 0:
                self.skin_combo.setCurrentIndex(index)
        else:
            self.skin_combo.addItem("No skin found")

        self.skin_combo.blockSignals(False)

    def select_osu_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select osu! folder", self.osu_folder or str(Path.home()))

        if not folder:
            return

        self.osu_folder = folder
        update_setting("osu_folder", folder)
        self.osu_label.setText(folder)
        self.load_saved_skins()
        self.try_auto_find_beatmap()

    def select_replay(self):
        start_dir = str(Path(self.replay_file).parent) if self.replay_file else str(Path.home())
        file, _ = QFileDialog.getOpenFileName(self, "Select replay", start_dir, "osu! replay (*.osr)")

        if not file:
            return

        self.replay_file = file
        update_setting("last_replay", file)
        self.replay_label.setText(file)
        self.load_replay_info()
        self.try_auto_find_beatmap()

    def load_replay_info(self):
        try:
            info = get_replay_info(self.replay_file)
            self.info_label.setText(
                f"{info['username']}  |  {info['score']:,} score  |  {info['max_combo']}x  |  {info['mods']}"
            )
        except Exception as error:
            self.info_label.setText("Replay could not be read")
            QMessageBox.warning(self, "Replay error", str(error))

    def try_auto_find_beatmap(self):
        if not self.osu_folder or not self.replay_file:
            return

        self.beatmap_label.setText("Searching for matching beatmap...")
        found = find_beatmap_from_replay(self.osu_folder, self.replay_file)

        if found:
            self.beatmap_file = found
            update_setting("last_beatmap", found)
            self.beatmap_label.setText(found)
        else:
            self.beatmap_file = None
            update_setting("last_beatmap", "")
            self.beatmap_label.setText("Beatmap not found")

    def select_manual_beatmap(self):
        start_dir = str(Path(self.beatmap_file).parent) if self.beatmap_file else str(Path.home())
        file, _ = QFileDialog.getOpenFileName(self, "Select beatmap", start_dir, "osu! beatmap (*.osu)")

        if not file:
            return

        self.beatmap_file = file
        update_setting("last_beatmap", file)
        self.beatmap_label.setText(file)

    def _render_options(self):
        return {
            "scroll_speed_value": self.scroll_speed_spin.value(),
            "resolution": self.resolution_combo.currentText(),
            "motion_blur": self.motion_blur_spin.value(),
            "show_side_overlay": self.side_overlay_check.isChecked(),
            "show_strain_graph": self.strain_graph_check.isChecked(),
            "vignette_strength": self.vignette_spin.value(),
            "results_background_opacity": self.results_opacity_spin.value() / 100.0,
            "results_duration": self.results_duration_spin.value(),
            "show_results_screen": self.results_screen_check.isChecked(),
            "colour_combo_during_holds": self.hold_combo_colour_check.isChecked(),
        }

    def _save_render_settings(self, options):
        update_setting("scroll_speed", str(options["scroll_speed_value"]))
        update_setting("resolution", options["resolution"])
        update_setting("motion_blur", str(options["motion_blur"]))
        update_setting("show_side_overlay", options["show_side_overlay"])
        update_setting("show_strain_graph", options["show_strain_graph"])
        update_setting("vignette_strength", str(options["vignette_strength"]))
        update_setting("results_background_opacity", str(int(options["results_background_opacity"] * 100)))
        update_setting("results_duration", str(options["results_duration"]))
        update_setting("show_results_screen", options["show_results_screen"])
        update_setting("colour_combo_during_holds", options["colour_combo_during_holds"])

    def start_render(self):
        if not self.beatmap_file or not self.replay_file:
            QMessageBox.warning(self, "Missing source", "Select both a replay and a beatmap first.")
            return

        output_start = self.settings.get("last_output_folder") or str(Path.home())
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save rendered video",
            str(Path(output_start) / "render.mp4"),
            "MP4 video (*.mp4)",
        )

        if not output_file:
            return

        if not output_file.lower().endswith(".mp4"):
            output_file += ".mp4"

        update_setting("last_output_folder", str(Path(output_file).parent))
        selected_skin = self.skin_combo.currentText()
        self.skin_folder = (
            str(Path(self.osu_folder) / "Skins" / selected_skin)
            if self.osu_folder and selected_skin != "No skin found"
            else None
        )

        if selected_skin != "No skin found":
            update_setting("last_skin", selected_skin)

        options = self._render_options()
        self._save_render_settings(options)
        self.progress.setValue(0)
        self.progress_label.setText("Preparing render...")
        self.render_button.setEnabled(False)

        self.thread = RenderThread(
            self.beatmap_file,
            self.skin_folder,
            output_file,
            self.replay_file,
            options,
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.finished_ok.connect(self.render_done)
        self.thread.failed.connect(self.render_failed)
        self.thread.start()

    def update_progress(self, percent, text):
        self.progress.setValue(percent)
        self.progress_label.setText(text)

    def render_done(self, output_file):
        self.render_button.setEnabled(True)
        self.progress.setValue(100)
        self.progress_label.setText("Render complete")
        QMessageBox.information(self, "Render complete", f"Video created:\n{output_file}")

    def render_failed(self, error):
        self.render_button.setEnabled(True)
        self.progress_label.setText("Render failed")
        QMessageBox.critical(self, "Render error", error)
