from pathlib import Path
import os
import platform
import shutil
import subprocess
import threading

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QThread, QTimer, Signal, Qt, QUrl
from PySide6.QtGui import QDesktopServices
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from osu_mania_replay_renderer.beatmap_parser import parse_osu
from osu_mania_replay_renderer.bundled_skins import (
    bundled_skin_entries,
    bundled_skin_labels,
    bundled_skin_path,
    matching_bundled_skin,
)
from osu_mania_replay_renderer.osu_finder import (
    find_beatmap_by_hash,
    find_osu_folder,
    get_replay_info,
    is_osu_folder,
    list_recent_replays,
    list_skins,
)
from osu_mania_replay_renderer.renderer import RenderCancelled, render_video
from osu_mania_replay_renderer.settings import load_settings, update_setting
from osu_mania_replay_renderer.updater import RELEASES_URL, check_for_update
from osu_mania_replay_renderer.version import __version__


SUPPORT_URL = "https://ko-fi.com/nico69yaza"
SUPPORT_MESSAGE = (
    "This tool is made by one person after many hours of testing and coding.\n\n"
    "A small donation really helps the project grow and helps me keep going.\n\n"
    "Thank you to everyone who decides to donate."
)


def setting_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in ("1", "true", "yes", "on")


class RenderThread(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, osu_file, skin_folder, output_file, replay_file, options):
        super().__init__()
        self.osu_file = osu_file
        self.skin_folder = skin_folder
        self.output_file = output_file
        self.replay_file = replay_file
        self.options = options
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        previous_require_fast = os.environ.get("MANIA_RENDERER_REQUIRE_FAST_GPU")
        previous_require_gpu = os.environ.get("MANIA_RENDERER_REQUIRE_GPU")

        try:
            if self.options.pop("require_fast_gpu", False):
                os.environ["MANIA_RENDERER_REQUIRE_FAST_GPU"] = "1"
                os.environ["MANIA_RENDERER_REQUIRE_GPU"] = "1"

            render_video(
                osu_file=self.osu_file,
                skin_folder=self.skin_folder,
                output_file=self.output_file,
                replay_file=self.replay_file,
                progress_callback=self.progress.emit,
                cancel_callback=self.cancel_event.is_set,
                **self.options,
            )
            self.finished_ok.emit(self.output_file)
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            if previous_require_fast is None:
                os.environ.pop("MANIA_RENDERER_REQUIRE_FAST_GPU", None)
            else:
                os.environ["MANIA_RENDERER_REQUIRE_FAST_GPU"] = previous_require_fast

            if previous_require_gpu is None:
                os.environ.pop("MANIA_RENDERER_REQUIRE_GPU", None)
            else:
                os.environ["MANIA_RENDERER_REQUIRE_GPU"] = previous_require_gpu


class UpdateCheckThread(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            self.completed.emit(check_for_update())
        except Exception as error:
            self.failed.emit(str(error))


class OsuFolderDiscoveryThread(QThread):
    completed = Signal(object)

    def run(self):
        self.completed.emit(find_osu_folder())


class BeatmapLookupThread(QThread):
    replay_loaded = Signal(object)
    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, osu_folder, replay_file, preferred_beatmap=None, parent=None):
        super().__init__(parent)
        self.osu_folder = osu_folder
        self.replay_file = replay_file
        self.preferred_beatmap = preferred_beatmap

    @staticmethod
    def _eta_text(seconds):
        if seconds is None:
            return "calculating ETA..."

        seconds = max(0, int(round(seconds)))

        if seconds >= 60:
            minutes, seconds = divmod(seconds, 60)
            return f"ETA {minutes}m {seconds:02d}s"

        return f"ETA {seconds}s"

    def run(self):
        try:
            info = get_replay_info(self.replay_file)

            if self.isInterruptionRequested():
                return

            self.replay_loaded.emit(info)

            if not self.osu_folder:
                self.completed.emit(None)
                return

            def report(checked, total, eta):
                if self.isInterruptionRequested():
                    return

                if total <= 0:
                    self.progress.emit(-1, "Indexing installed beatmaps...")
                    return

                percent = int(checked / max(1, total) * 100)
                self.progress.emit(
                    percent,
                    f"Scanning beatmaps: {checked:,}/{total:,} | {self._eta_text(eta)}",
                )

            found = find_beatmap_by_hash(
                self.osu_folder,
                info["beatmap_hash"],
                progress_callback=report,
                cancel_callback=self.isInterruptionRequested,
                preferred_path=self.preferred_beatmap,
            )

            if not self.isInterruptionRequested():
                self.completed.emit(found)
        except Exception as error:
            if not self.isInterruptionRequested():
                self.failed.emit(str(error))


class SearchableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.setCursor(Qt.PointingHandCursor)

        self.arrow_button = QToolButton(self)
        self.arrow_button.setObjectName("comboArrowButton")
        self.arrow_button.setText("▾")
        self.arrow_button.setCursor(Qt.PointingHandCursor)
        self.arrow_button.setFocusPolicy(Qt.NoFocus)
        self.arrow_button.clicked.connect(self.showPopup)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        button_width = 32
        self.arrow_button.setGeometry(
            self.width() - button_width - 2,
            2,
            button_width,
            max(0, self.height() - 4),
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("osu!mania Replay Renderer")
        self.setMinimumSize(980, 680)

        self.settings = load_settings()
        self.osu_folder = self.settings["osu_folder"] or None
        self.replay_file = self.settings["last_replay"] or None
        self.beatmap_file = self.settings["last_beatmap"] or None
        self.skin_folder = None
        self.thread = None
        self.update_thread = None
        self.discovery_thread = None
        self.beatmap_thread = None
        self.replay_combo_paths = []
        self.all_skin_names = []
        self.replay_search_dirty = False

        self._build_ui()
        self._apply_style()
        if self.osu_folder and is_osu_folder(self.osu_folder):
            self._activate_osu_folder(self.osu_folder, persist=False)
        else:
            self.osu_folder = None
            self._start_osu_discovery()

        if self.replay_file:
            self._start_beatmap_lookup()

        QTimer.singleShot(1200, self.check_updates_on_startup)

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("appRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(28, 24, 28, 24)
        root_layout.setSpacing(18)

        header = QHBoxLayout()
        header.setSpacing(14)
        title_column = QVBoxLayout()
        title_column.setSpacing(2)
        title = QLabel("osu!mania Fast Renderer")
        title.setObjectName("pageTitle")
        subtitle = QLabel(f"GPU-first replay rendering with dynamic skin support  •  v{__version__}")
        subtitle.setObjectName("pageSubtitle")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)

        self.support_button = QPushButton("♥")
        self.support_button.setObjectName("supportButton")
        self.support_button.setToolTip("Support the project")
        self.support_button.clicked.connect(lambda: self.show_support_popup(manual=True))

        header.addLayout(title_column, 1)
        header.addWidget(self.support_button, 0, Qt.AlignRight | Qt.AlignTop)
        root_layout.addLayout(header)

        render_tab = QWidget()
        render_layout = QVBoxLayout(render_tab)
        render_layout.setContentsMargins(0, 8, 0, 0)
        render_layout.setSpacing(16)
        content = QHBoxLayout()
        content.setSpacing(16)
        content.addWidget(self._build_source_group(), 3)
        content.addWidget(self._build_options_group(), 2)
        render_layout.addLayout(content, 1)
        render_layout.addWidget(self._build_render_group())

        self.skin_combo.currentTextChanged.connect(self._skin_selection_changed)
        root_layout.addWidget(render_tab, 1)

        self.setCentralWidget(root)

    def _skin_selection_changed(self, value):
        if value and value in self.all_skin_names:
            update_setting("last_skin", value)

    def _path_row(self, label, button_text, callback):
        row = QVBoxLayout()
        row.setSpacing(6)
        path_label = QLabel(label)
        path_label.setObjectName("pathLabel")
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button = QPushButton(button_text)
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
            self.select_osu_folder,
        )
        replay_row, self.replay_label = self._path_row(
            self.replay_file or "No replay selected",
            "Select replay",
            self.select_replay,
        )
        self.replay_combo = SearchableComboBox()
        self.replay_combo.setPlaceholderText("Search recent replays...")
        self.replay_combo.activated.connect(self._recent_replay_selected)
        self.replay_combo.lineEdit().textEdited.connect(self._replay_search_edited)
        self.replay_combo.lineEdit().editingFinished.connect(self._refresh_replays_from_search)
        beatmap_row, self.beatmap_label = self._path_row(
            self.beatmap_file or "No beatmap selected",
            "Select beatmap manually",
            self.select_manual_beatmap,
        )

        layout.addLayout(osu_row)
        layout.addLayout(replay_row)
        layout.addWidget(self.replay_combo)

        self.info_label = QLabel("Replay details unavailable")
        self.info_label.setObjectName("infoLabel")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.lookup_progress = QProgressBar()
        self.lookup_progress.setRange(0, 100)
        self.lookup_progress.hide()
        self.lookup_status = QLabel("")
        self.lookup_status.setObjectName("lookupStatus")
        self.lookup_status.setWordWrap(True)
        self.lookup_status.hide()
        layout.addWidget(self.lookup_progress)
        layout.addWidget(self.lookup_status)
        layout.addLayout(beatmap_row)

        skin_label = QLabel("Skin")
        skin_label.setObjectName("fieldLabel")
        self.skin_combo = SearchableComboBox()
        self.skin_combo.setPlaceholderText("Search skins...")
        self.skin_combo.lineEdit().editingFinished.connect(self._filter_skins_from_search)
        self.skin_combo.addItem("No skin found")
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

        self.locked_resolution = "1920x1080"
        self.resolution_label = QLabel(self.locked_resolution)
        self.resolution_label.setObjectName("fixedTextLabel")

        self.scroll_speed_spin = QDoubleSpinBox()
        self.scroll_speed_spin.setRange(1.0, 40.0)
        self.scroll_speed_spin.setSingleStep(0.5)
        self.scroll_speed_spin.setDecimals(1)
        self.scroll_speed_spin.setValue(float(self.settings.get("scroll_speed", "30.0")))

        self.vignette_spin = QDoubleSpinBox()
        self.vignette_spin.setRange(0, 60)
        self.vignette_spin.setDecimals(0)
        self.vignette_spin.setSuffix(" %")
        self.vignette_spin.setValue(float(self.settings.get("vignette_strength", "12")))

        self.results_opacity_spin = QDoubleSpinBox()
        self.results_opacity_spin.setRange(20, 100)
        self.results_opacity_spin.setDecimals(0)
        self.results_opacity_spin.setSuffix(" %")
        self.results_opacity_spin.setValue(float(self.settings.get("results_background_opacity", "62")))

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

        layout.addRow("Resolution", self.resolution_label)
        layout.addRow("Scroll speed", self.scroll_speed_spin)
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
        self.progress_animation = QPropertyAnimation(self.progress, b"value", self)
        self.progress_animation.setDuration(180)
        self.progress_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.progress_label = QLabel("Ready")
        self.progress_label.setObjectName("progressLabel")

        self.render_button = QPushButton("Render video")
        self.render_button.setObjectName("primaryButton")
        self.render_button.clicked.connect(self.start_render)

        self.stop_button = QPushButton("Stop render")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.cancel_render)

        self.update_button = QPushButton("Check for updates")
        self.update_button.setObjectName("secondaryButton")
        self.update_button.clicked.connect(self.check_updates)

        self.renders_folder_button = QPushButton("Take me to renders folder")
        self.renders_folder_button.setObjectName("secondaryButton")
        self.renders_folder_button.clicked.connect(self.open_renders_folder)

        layout.addWidget(self.progress)
        layout.addWidget(self.progress_label)
        commands = QHBoxLayout()
        commands.setSpacing(10)
        commands.addWidget(self.render_button, 3)
        commands.addWidget(self.stop_button, 2)
        layout.addLayout(commands)
        layout.addWidget(self.renders_folder_button)
        layout.addWidget(self.update_button)
        return group

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #0c1017;
            }
            QMessageBox {
                background-color: #f3f5f8;
                color: #111827;
            }
            QMessageBox QLabel {
                color: #111827;
                background: transparent;
                font-size: 13px;
            }
            QMessageBox QCheckBox {
                color: #111827;
                background: transparent;
            }
            QMessageBox QPushButton {
                min-width: 88px;
                min-height: 32px;
                background: #263244;
                border: 1px solid #3a4658;
                border-radius: 9px;
                color: #ffffff;
                padding: 0 14px;
                font-weight: 700;
            }
            QMessageBox QPushButton:hover {
                background: #344155;
                border-color: #597089;
            }
            QWidget#appRoot {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #101722, stop:0.48 #0e121a, stop:1 #15111d);
                color: #edf3f8;
            }
            QWidget {
                color: #edf3f8;
                font-family: Inter, Noto Sans, sans-serif;
                font-size: 13px;
            }
            QLabel, QCheckBox { background: transparent; }
            QLabel#pageTitle { font-size: 28px; font-weight: 900; color: #ffffff; letter-spacing: 0.3px; }
            QLabel#pageSubtitle { color: #9fb0bf; margin-bottom: 6px; }
            QLabel#pathLabel { color: #a9b4bf; padding: 2px 0; }
            QLabel#fieldLabel, QLabel#progressLabel { color: #b8c4cf; }
            QLabel#lookupStatus { color: #aeb8c2; font-size: 12px; }
            QLabel#layoutStatus { color: #aeb4ba; font-weight: 600; }
            QLabel#fixedTextLabel {
                color: #d8e5ee;
                min-height: 34px;
                padding: 0;
            }
            QLabel#infoLabel {
                color: #92f0ff;
                background: rgba(25, 34, 45, 0.82);
                border: 1px solid rgba(96, 199, 213, 0.25);
                border-radius: 11px;
                padding: 10px;
            }
            QGroupBox {
                background: rgba(18, 24, 34, 0.88);
                border: 1px solid rgba(120, 151, 184, 0.18);
                border-radius: 16px;
                margin-top: 10px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 7px;
                color: #f1f1f1;
            }
            QPushButton, QComboBox, QDoubleSpinBox {
                min-height: 36px;
                background: rgba(35, 43, 56, 0.96);
                border: 1px solid #35445a;
                border-radius: 11px;
                padding: 0 12px;
            }
            QComboBox {
                padding-right: 34px;
            }
            QPushButton:hover, QComboBox:hover, QDoubleSpinBox:hover {
                border-color: #70d8e7;
                background: rgba(47, 59, 74, 0.98);
            }
            QPushButton:disabled { color: #747f89; background: rgba(32, 36, 43, 0.85); }
            QPushButton#primaryButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2aa66f, stop:1 #42c592);
                border-color: #5bdeb0;
                color: #ffffff;
                font-weight: 800;
                min-height: 42px;
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #31b97d, stop:1 #4cd4a1);
            }
            QPushButton#renderButtonBusy {
                background: rgba(62, 68, 78, 0.96);
                border-color: #7a8390;
                color: #d5dce4;
                font-weight: 800;
                min-height: 42px;
            }
            QPushButton#renderButtonBusy:disabled {
                background: rgba(62, 68, 78, 0.96);
                border-color: #7a8390;
                color: #d5dce4;
            }
            QPushButton#stopButton {
                background: rgba(68, 35, 42, 0.92);
                border-color: #8c4b56;
                color: #f4dfe2;
                font-weight: 800;
                min-height: 42px;
            }
            QPushButton#stopButton:hover { background: #57313a; border-color: #b76470; }
            QPushButton#stopButton:disabled { background: #242628; border-color: #34373a; color: #6f7478; }
            QPushButton#stopButtonActive {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #d93d4b, stop:1 #ff6a4a);
                border-color: #ff8d7b;
                color: #ffffff;
                font-weight: 900;
                min-height: 42px;
            }
            QPushButton#stopButtonActive:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ee4c5a, stop:1 #ff7658);
                border-color: #ffb0a3;
            }
            QPushButton#stopButtonActive:disabled {
                background: #463037;
                border-color: #65424a;
                color: #9f7d83;
            }
            QPushButton#secondaryButton {
                background: rgba(37, 44, 57, 0.92);
                border-color: #38475c;
                color: #d7e1ea;
                font-weight: 700;
            }
            QPushButton#secondaryButton:hover {
                background: rgba(49, 59, 75, 0.98);
                border-color: #70d8e7;
                color: #ffffff;
            }
            QPushButton#supportButton {
                min-width: 42px;
                max-width: 42px;
                min-height: 42px;
                max-height: 42px;
                border-radius: 10px;
                background: rgba(58, 66, 78, 0.96);
                border: 1px solid #6d7684;
                color: #ffffff;
                font-size: 20px;
                font-weight: 900;
                padding: 0;
            }
            QPushButton#supportButton:hover {
                background: rgba(76, 86, 101, 0.98);
                border-color: #9aa6b5;
            }
            QComboBox::drop-down { border: 0; width: 32px; }
            QComboBox::down-arrow { image: none; width: 0; height: 0; }
            QToolButton#comboArrowButton {
                border: 0;
                background: transparent;
                color: #9eeaf3;
                font-size: 15px;
                font-weight: 800;
                padding: 0;
            }
            QToolButton#comboArrowButton:hover {
                color: #ffffff;
                background: rgba(89, 189, 204, 0.15);
                border-radius: 7px;
            }
            QCheckBox { spacing: 8px; min-height: 22px; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid #555a60;
                border-radius: 4px;
                background: #242628;
            }
            QCheckBox::indicator:checked { background: #59bdcc; border-color: #76d7e4; }
            QProgressBar {
                min-height: 12px;
                max-height: 12px;
                border: 0;
                border-radius: 6px;
                background: rgba(55, 65, 78, 0.9);
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #61d6e8, stop:1 #6ef0b2);
            }
        """)

    def load_saved_skins(self):
        bundled_entries = bundled_skin_entries()
        verified_labels = [entry["label"] for entry in bundled_entries]
        bundled_source_names = {entry["source_name"] for entry in bundled_entries}
        bundled_folders = {entry["folder"] for entry in bundled_entries}
        local_skins = list_skins(self.osu_folder) if self.osu_folder else []
        local_skins = [
            skin
            for skin in local_skins
            if skin not in bundled_source_names and skin not in bundled_folders
        ]
        skins = [*verified_labels, *local_skins]
        self.all_skin_names = skins
        self.skin_combo.blockSignals(True)
        self.skin_combo.clear()

        if skins:
            self.skin_combo.addItems(skins)
            preferred = self._preferred_skin_name(skins)
            index = self.skin_combo.findText(preferred)

            if index >= 0:
                self.skin_combo.setCurrentIndex(index)
        else:
            self.skin_combo.addItem("No skin found")

        self.skin_combo.blockSignals(False)
        self._skin_selection_changed(self.skin_combo.currentText())

    def _preferred_skin_name(self, skins):
        key_count = None
        if self.beatmap_file:
            try:
                key_count = parse_osu(self.beatmap_file).keys
            except Exception:
                key_count = None

        preferred = None
        if key_count == 7:
            preferred = self.settings.get("default_skin_7k", "")
        elif key_count == 4:
            preferred = self.settings.get("default_skin_4k", "")

        for candidate in (preferred, self.settings.get("last_skin", "")):
            if candidate and candidate in skins:
                return candidate

        if key_count == 4:
            preferred_verified = matching_bundled_skin("nico69 v4")
            if preferred_verified in skins:
                return preferred_verified
            for skin in skins:
                if "nico69" in skin.lower() and "v4" in skin.lower():
                    return skin
        if key_count == 7:
            preferred_verified = matching_bundled_skin("cawolo new max")
            if preferred_verified in skins:
                return preferred_verified
            for skin in skins:
                lower = skin.lower()
                if "cawolo" in lower and "new" in lower and "max" in lower:
                    return skin

        return skins[0] if skins else ""

    def _filter_skins_from_search(self):
        query_text = self.skin_combo.currentText().strip()
        best = self._best_skin_name(query_text)

        if not best:
            return

        index = self.skin_combo.findText(best)

        if index >= 0:
            self.skin_combo.blockSignals(True)
            self.skin_combo.setCurrentIndex(index)
            if self.skin_combo.isEditable():
                self.skin_combo.lineEdit().setText(best)
            self.skin_combo.blockSignals(False)
            self._skin_selection_changed(best)
            self.skin_combo.showPopup()

    def _best_skin_name(self, query_text):
        query = str(query_text or "").strip().lower()
        if not self.all_skin_names or query in {"", "no skin found", "no matching skin"}:
            return None

        bundled_match = matching_bundled_skin(query)
        if bundled_match in self.all_skin_names:
            return bundled_match

        def score_skin(name):
            lower = name.lower()
            if lower == query:
                return (0, len(name), name)
            if lower.startswith(query):
                return (1, len(name), name)
            if query in lower:
                return (2, len(name), name)

            tokens = [token for token in query.replace("_", " ").replace("-", " ").split() if token]
            if tokens and all(token in lower for token in tokens):
                return (3, len(name), name)

            return None

        ranked = [(score_skin(skin), skin) for skin in self.all_skin_names]
        ranked = [(score, skin) for score, skin in ranked if score is not None]

        if not ranked:
            return None

        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]

    def _start_osu_discovery(self):
        if self.discovery_thread and self.discovery_thread.isRunning():
            return

        self.osu_label.setText("Detecting osu! installation...")
        self.discovery_thread = OsuFolderDiscoveryThread(self)
        self.discovery_thread.completed.connect(self._osu_discovery_done)
        self.discovery_thread.start()

    def _osu_discovery_done(self, folder):
        if folder:
            self._activate_osu_folder(folder, persist=True)
            self._start_beatmap_lookup()
        else:
            self.osu_label.setText("osu! installation not detected")

    def _activate_osu_folder(self, folder, persist=True):
        self.osu_folder = str(Path(folder))
        self.settings["osu_folder"] = self.osu_folder
        self.osu_label.setText(self.osu_folder)

        if persist:
            update_setting("osu_folder", self.osu_folder)

        self.load_saved_skins()
        self.refresh_recent_replays()

    def refresh_recent_replays(self, query=""):
        if not getattr(self, "replay_combo", None):
            return

        query = query.strip()
        self.replay_combo.blockSignals(True)
        self.replay_combo.clear()
        self.replay_combo_paths = []
        selected_index = -1

        if not self.osu_folder:
            self.replay_combo.addItem("No osu! folder selected")
        else:
            replays = list_recent_replays(self.osu_folder, limit=250, query=query)

            if replays:
                self.replay_combo.addItem("Recent replays...")
                self.replay_combo_paths.append("")

                for replay in replays:
                    self.replay_combo.addItem(replay["name"])
                    self.replay_combo_paths.append(replay["path"])
                    if self.replay_file and Path(replay["path"]) == Path(self.replay_file):
                        selected_index = self.replay_combo.count() - 1
            else:
                self.replay_combo.addItem("No replay found")

        if selected_index >= 0:
            self.replay_combo.setCurrentIndex(selected_index)
        elif query and self.replay_combo.isEditable():
            self.replay_combo.setCurrentIndex(-1)
            self.replay_combo.lineEdit().setText(query)

        self.replay_combo.blockSignals(False)

    def _replay_search_edited(self):
        self.replay_search_dirty = True

    def _refresh_replays_from_search(self):
        if not self.replay_search_dirty:
            return

        text = self.replay_combo.currentText()

        if text in {"Recent replays...", "No replay found", "No osu! folder selected"}:
            self.replay_search_dirty = False
            return

        self.refresh_recent_replays(text)
        self.replay_search_dirty = False
        self.replay_combo.showPopup()

    def _recent_replay_selected(self, index):
        if index < 0 or index >= len(self.replay_combo_paths):
            return

        replay = self.replay_combo_paths[index]

        if not replay:
            return

        self.replay_search_dirty = False
        self.replay_file = replay
        self.settings["last_replay"] = replay
        update_setting("last_replay", replay)
        self.replay_label.setText(replay)
        self._start_beatmap_lookup()

    def _native_dialog_directory(self, path):
        directory = Path(path).expanduser()
        return str(directory if directory.exists() else Path.home())

    @staticmethod
    def _host_process_environment():
        environment = os.environ.copy()

        if "LD_LIBRARY_PATH_ORIG" in environment:
            environment["LD_LIBRARY_PATH"] = environment["LD_LIBRARY_PATH_ORIG"]
        else:
            environment.pop("LD_LIBRARY_PATH", None)

        return environment

    @staticmethod
    def _filter_parts(file_filter):
        name, separator, pattern = file_filter.partition("(")
        return name.strip(), pattern.rstrip(")").strip() if separator else "*"

    def _linux_system_picker(self, mode, title, start_path, file_filter=""):
        if platform.system().lower() != "linux":
            return None

        start_path = str(Path(start_path).expanduser())
        environment = self._host_process_environment()
        name, pattern = self._filter_parts(file_filter)
        zenity = shutil.which("zenity")

        if zenity:
            command = [zenity, "--file-selection", f"--title={title}"]

            if mode == "directory":
                command.append("--directory")
                command.append(f"--filename={start_path.rstrip('/')}/")
            elif mode == "save":
                command.extend(("--save", "--confirm-overwrite", f"--filename={start_path}"))
            else:
                command.append(f"--filename={start_path.rstrip('/')}/")

            if file_filter:
                command.append(f"--file-filter={name} | {pattern}")

            result = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)

            if result.returncode in (0, 1):
                return result.stdout.strip() if result.returncode == 0 else ""

        kdialog = shutil.which("kdialog")

        if kdialog:
            if mode == "directory":
                command = [kdialog, "--getexistingdirectory", start_path, "--title", title]
            elif mode == "save":
                command = [kdialog, "--getsavefilename", start_path, f"{pattern}|{name}", "--title", title]
            else:
                command = [kdialog, "--getopenfilename", start_path, f"{pattern}|{name}", "--title", title]

            result = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)

            if result.returncode in (0, 1):
                return result.stdout.strip() if result.returncode == 0 else ""

        return None

    def _get_existing_directory(self, title, start_dir):
        system_result = self._linux_system_picker("directory", title, start_dir)

        if system_result is not None:
            return system_result

        dialog = QFileDialog(self, title, self._native_dialog_directory(start_dir))
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, False)

        if not dialog.exec():
            return ""

        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def _get_open_file_name(self, title, start_dir, file_filter):
        system_result = self._linux_system_picker("open", title, start_dir, file_filter)

        if system_result is not None:
            return system_result

        dialog = QFileDialog(self, title, self._native_dialog_directory(start_dir), file_filter)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setOption(QFileDialog.DontUseNativeDialog, False)

        if not dialog.exec():
            return ""

        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def _get_save_file_name(self, title, start_path, file_filter):
        system_result = self._linux_system_picker("save", title, start_path, file_filter)

        if system_result is not None:
            return system_result

        dialog = QFileDialog(self, title, self._native_dialog_directory(Path(start_path).parent), file_filter)
        dialog.selectFile(str(start_path))
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setOption(QFileDialog.DontUseNativeDialog, False)

        if not dialog.exec():
            return ""

        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def select_osu_folder(self):
        folder = self._get_existing_directory("Select osu! folder", self.osu_folder or Path.home())

        if not folder:
            return

        self._activate_osu_folder(folder)
        self._start_beatmap_lookup()

    def select_replay(self):
        start_dir = (
            Path(self.osu_folder) / "Replays"
            if self.osu_folder
            else Path(self.replay_file).parent if self.replay_file else Path.home()
        )
        file = self._get_open_file_name("Select replay", start_dir, "osu! replay (*.osr)")

        if not file:
            return

        self.replay_file = file
        self.settings["last_replay"] = file
        update_setting("last_replay", file)
        self.replay_label.setText(file)
        self._start_beatmap_lookup()

    def load_replay_info(self):
        self._start_beatmap_lookup()

    def try_auto_find_beatmap(self):
        self._start_beatmap_lookup()

    def _start_beatmap_lookup(self):
        if not self.replay_file:
            return

        if self.beatmap_thread and self.beatmap_thread.isRunning():
            self.beatmap_thread.requestInterruption()

        preferred_beatmap = self.settings.get("last_beatmap") or self.beatmap_file
        self.beatmap_file = None
        self.settings["last_beatmap"] = ""
        self.render_button.setEnabled(False)
        self.info_label.setText("Reading replay...")
        self.beatmap_label.setText("Searching for matching beatmap...")
        self.lookup_progress.setRange(0, 0)
        self.lookup_progress.show()
        self.lookup_status.setText("Reading replay metadata...")
        self.lookup_status.show()

        thread = BeatmapLookupThread(self.osu_folder, self.replay_file, preferred_beatmap, self)
        self.beatmap_thread = thread
        thread.replay_loaded.connect(lambda info, owner=thread: self._replay_loaded(owner, info))
        thread.progress.connect(lambda value, text, owner=thread: self._beatmap_lookup_progress(owner, value, text))
        thread.completed.connect(lambda found, owner=thread: self._beatmap_lookup_done(owner, found))
        thread.failed.connect(lambda error, owner=thread: self._beatmap_lookup_failed(owner, error))
        thread.finished.connect(lambda owner=thread: self._beatmap_lookup_finished(owner))
        thread.start()

    def _beatmap_lookup_finished(self, owner):
        if owner is self.beatmap_thread:
            self.beatmap_thread = None

        owner.deleteLater()

    def _replay_loaded(self, owner, info):
        if owner is not self.beatmap_thread:
            return

        self.info_label.setText(
            f"{info['username']}  |  {info['score']:,} score  |  {info['max_combo']}x  |  {info['mods']}"
        )

    def _beatmap_lookup_progress(self, owner, value, text):
        if owner is not self.beatmap_thread:
            return

        if value < 0:
            self.lookup_progress.setRange(0, 0)
        else:
            self.lookup_progress.setRange(0, 100)
            self.lookup_progress.setValue(value)

        self.lookup_status.setText(text)

    def _beatmap_lookup_done(self, owner, found):
        if owner is not self.beatmap_thread:
            return

        self.lookup_progress.hide()
        self.lookup_status.hide()

        if found:
            self.beatmap_file = found
            self.settings["last_beatmap"] = found
            update_setting("last_beatmap", found)
            self.beatmap_label.setText(found)
            self.render_button.setEnabled(True)
            self.load_saved_skins()
            self._skin_selection_changed(self.skin_combo.currentText())
        else:
            self.beatmap_file = None
            self.settings["last_beatmap"] = ""
            update_setting("last_beatmap", "")
            self.beatmap_label.setText(
                "Beatmap not found" if self.osu_folder else "Select the osu! folder to search for the beatmap"
            )
            self.render_button.setEnabled(False)

    def _beatmap_lookup_failed(self, owner, error):
        if owner is not self.beatmap_thread:
            return

        self.lookup_progress.hide()
        self.lookup_status.hide()
        self.info_label.setText("Replay could not be read")
        self.beatmap_label.setText("Beatmap search failed")
        self.render_button.setEnabled(False)
        QMessageBox.warning(self, "Replay error", error)

    def select_manual_beatmap(self):
        start_dir = (
            Path(self.beatmap_file).parent
            if self.beatmap_file
            else Path(self.osu_folder) / "Songs" if self.osu_folder else Path.home()
        )
        file = self._get_open_file_name("Select beatmap", start_dir, "osu! beatmap (*.osu)")

        if not file:
            return

        self.beatmap_file = file
        self.settings["last_beatmap"] = file
        update_setting("last_beatmap", file)
        self.beatmap_label.setText(file)
        self.render_button.setEnabled(bool(self.replay_file))
        self.load_saved_skins()
        self._skin_selection_changed(self.skin_combo.currentText())

        if self.beatmap_thread and self.beatmap_thread.isRunning():
            self.beatmap_thread.requestInterruption()

        self.beatmap_thread = None

        self.lookup_progress.hide()
        self.lookup_status.hide()

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.cancel()

        background_threads = (self.beatmap_thread, self.discovery_thread, self.update_thread, self.thread)

        for thread in background_threads:
            if thread and thread.isRunning():
                thread.requestInterruption()

        deadline = 3000

        for thread in background_threads:
            if thread and thread.isRunning():
                thread.wait(deadline)

        super().closeEvent(event)

    def _render_options(self):
        return {
            "scroll_speed_value": self.scroll_speed_spin.value(),
            "resolution": self.locked_resolution,
            "motion_blur": 0,
            "show_side_overlay": self.side_overlay_check.isChecked(),
            "show_strain_graph": self.strain_graph_check.isChecked(),
            "vignette_strength": self.vignette_spin.value(),
            "results_background_opacity": self.results_opacity_spin.value() / 100.0,
            "results_duration": self.results_duration_spin.value(),
            "show_results_screen": self.results_screen_check.isChecked(),
            "colour_combo_during_holds": self.hold_combo_colour_check.isChecked(),
            "gpu_compositing": True,
            "require_fast_gpu": True,
            "layout_positions": {},
        }

    def _save_render_settings(self, options):
        update_setting("scroll_speed", str(options["scroll_speed_value"]))
        update_setting("resolution", options["resolution"])
        update_setting("show_side_overlay", options["show_side_overlay"])
        update_setting("show_strain_graph", options["show_strain_graph"])
        update_setting("vignette_strength", str(int(options["vignette_strength"])))
        update_setting("results_background_opacity", str(int(options["results_background_opacity"] * 100)))
        update_setting("results_duration", str(options["results_duration"]))
        update_setting("show_results_screen", options["show_results_screen"])
        update_setting("colour_combo_during_holds", options["colour_combo_during_holds"])
        update_setting("gpu_compositing", True)

    def _refresh_button_style(self, *buttons):
        for button in buttons:
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    def _set_rendering_controls(self, rendering):
        if rendering:
            self.render_button.setObjectName("renderButtonBusy")
            self.render_button.setText("Render video")
            self.render_button.setEnabled(False)
            self.stop_button.setObjectName("stopButtonActive")
            self.stop_button.setEnabled(True)
        else:
            self.render_button.setObjectName("primaryButton")
            self.render_button.setText("Render video")
            self.render_button.setEnabled(bool(self.replay_file and self.beatmap_file))
            self.stop_button.setObjectName("stopButton")
            self.stop_button.setEnabled(False)

        self._refresh_button_style(self.render_button, self.stop_button)

    def _renders_folder(self):
        folder = self.settings.get("last_output_folder") or str(Path.home() / "Videos")
        path = Path(folder).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def open_renders_folder(self):
        path = self._renders_folder()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def show_support_popup(self, manual=False):
        if not manual and setting_bool(self.settings.get("hide_support_popup", False)):
            return

        box = QMessageBox(self)
        box.setWindowTitle("Support the project")
        box.setText("Support osu!mania Fast Renderer")
        box.setInformativeText(SUPPORT_MESSAGE)
        support_button = box.addButton("Support on Ko-fi", QMessageBox.AcceptRole)
        box.addButton("Not now", QMessageBox.RejectRole)

        checkbox = QCheckBox("Do not show again")
        checkbox.setChecked(False)
        box.setCheckBox(checkbox)
        box.exec()

        if checkbox.isChecked():
            self.settings["hide_support_popup"] = True
            update_setting("hide_support_popup", True)

        if box.clickedButton() is support_button:
            QDesktopServices.openUrl(QUrl(SUPPORT_URL))

    def start_render(self):
        if not self.beatmap_file or not self.replay_file:
            QMessageBox.warning(self, "Missing source", "Select both a replay and a beatmap first.")
            return

        output_start = self.settings.get("last_output_folder") or str(Path.home())
        output_file = self._get_save_file_name(
            "Save rendered video",
            Path(output_start) / "render.mp4",
            "MP4 video (*.mp4)",
        )

        if not output_file:
            return

        if not output_file.lower().endswith(".mp4"):
            output_file += ".mp4"

        output_folder = str(Path(output_file).parent)
        self.settings["last_output_folder"] = output_folder
        update_setting("last_output_folder", output_folder)
        selected_skin = self._best_skin_name(self.skin_combo.currentText()) or self.skin_combo.currentText().strip()
        if selected_skin and selected_skin != self.skin_combo.currentText():
            index = self.skin_combo.findText(selected_skin)
            if index >= 0:
                self.skin_combo.setCurrentIndex(index)

        bundled_path = bundled_skin_path(selected_skin)
        self.skin_folder = bundled_path or (
            str(Path(self.osu_folder) / "Skins" / selected_skin)
            if self.osu_folder and selected_skin and selected_skin != "No skin found"
            else None
        )

        if self.skin_folder and not Path(self.skin_folder).is_dir():
            QMessageBox.warning(
                self,
                "Skin not found",
                f"The selected skin folder does not exist:\n{self.skin_folder}\n\n"
                "Select a skin from the dropdown before rendering.",
            )
            return

        if selected_skin != "No skin found":
            update_setting("last_skin", selected_skin)

        options = self._render_options()
        self._save_render_settings(options)
        self.progress.setValue(0)
        self.progress_label.setText("Preparing render...")
        self._set_rendering_controls(True)

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
        self.thread.cancelled.connect(self.render_cancelled)
        self.thread.start()

    def update_progress(self, percent, text):
        self.progress_animation.stop()
        self.progress_animation.setStartValue(self.progress.value())
        self.progress_animation.setEndValue(percent)
        self.progress_animation.start()
        self.progress_label.setText(text)

    def cancel_render(self):
        if not self.thread or not self.thread.isRunning():
            return

        self.thread.cancel()
        self.stop_button.setEnabled(False)
        self.progress_label.setText("Stopping render...")

    def render_done(self, output_file):
        self._set_rendering_controls(False)
        self.progress.setValue(100)
        self.progress_label.setText("Render complete")
        QMessageBox.information(self, "Render complete", f"Video created:\n{output_file}")
        self.show_support_popup()

    def render_failed(self, error):
        self._set_rendering_controls(False)
        self.progress_label.setText("Render failed")
        QMessageBox.critical(self, "Render error", error)

    def render_cancelled(self):
        self._set_rendering_controls(False)
        self.progress_animation.stop()
        self.progress.setValue(0)
        self.progress_label.setText("Render cancelled")

    def check_updates_on_startup(self):
        self.check_updates(automatic=True)

    def check_updates(self, automatic=False):
        if self.update_thread and self.update_thread.isRunning():
            return

        if not automatic:
            self.update_button.setEnabled(False)
            self.update_button.setText("Checking...")

        self.update_thread = UpdateCheckThread(self)
        self.update_thread.completed.connect(
            lambda update: self.update_check_done(update, automatic=automatic)
        )
        self.update_thread.failed.connect(
            lambda error: self.update_check_failed(error, automatic=automatic)
        )
        self.update_thread.start()

    def update_check_done(self, update, automatic=False):
        self.update_button.setEnabled(True)
        self.update_button.setText("Check for updates")

        if update is None:
            if not automatic:
                QMessageBox.information(self, "Updates", "You already have the latest version.")
            return

        answer = QMessageBox.question(
            self,
            "Update available",
            f"Version {update.version} is available.\n\nOpen the GitHub release to download {update.asset_name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if answer == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl(update.release_url))

    def update_check_failed(self, error, automatic=False):
        self.update_button.setEnabled(True)
        self.update_button.setText("Check for updates")

        if automatic:
            return

        answer = QMessageBox.question(
            self,
            "Update check unavailable",
            f"{error}\n\nOpen GitHub Releases in your browser instead?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if answer == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl(RELEASES_URL))
