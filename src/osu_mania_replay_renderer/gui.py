from pathlib import Path
import os
import platform
import shutil
import subprocess
import threading

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QThread, Signal, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsOpacityEffect,
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from osu_mania_replay_renderer.osu_finder import (
    find_beatmap_by_hash,
    find_osu_folder,
    get_replay_info,
    is_osu_folder,
    list_skins,
)
from osu_mania_replay_renderer.layout_editor import LayoutEditor
from osu_mania_replay_renderer.renderer import RenderCancelled, render_video
from osu_mania_replay_renderer.settings import load_settings, update_setting
from osu_mania_replay_renderer.updater import RELEASES_URL, check_for_update
from osu_mania_replay_renderer.version import __version__


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
        try:
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("osu!mania Replay Renderer")
        self.setMinimumSize(880, 650)

        self.settings = load_settings()
        self.osu_folder = self.settings["osu_folder"] or None
        self.replay_file = self.settings["last_replay"] or None
        self.beatmap_file = self.settings["last_beatmap"] or None
        self.layout_positions = dict(self.settings.get("layout_positions") or {})
        self.skin_folder = None
        self.thread = None
        self.update_thread = None
        self.discovery_thread = None
        self.beatmap_thread = None

        self._build_ui()
        self._apply_style()
        if self.osu_folder and is_osu_folder(self.osu_folder):
            self._activate_osu_folder(self.osu_folder, persist=False)
        else:
            self.osu_folder = None
            self._start_osu_discovery()

        if self.replay_file:
            self._start_beatmap_lookup()

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(24, 20, 24, 22)
        root_layout.setSpacing(16)

        title = QLabel("osu!mania Replay Renderer")
        title.setObjectName("pageTitle")
        subtitle = QLabel(f"Render local osu!mania replays with legacy skin support  •  v{__version__}")
        subtitle.setObjectName("pageSubtitle")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

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

        self.layout_editor = LayoutEditor(self.layout_positions)
        self.layout_editor.positions_changed.connect(self._layout_positions_changed)
        self.tabs.addTab(render_tab, "Render")
        self.tabs.addTab(self.layout_editor, "Layout")
        self.tabs.currentChanged.connect(self._animate_current_tab)
        root_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(root)

    def _layout_positions_changed(self, positions):
        self.layout_positions = dict(positions)
        self.settings["layout_positions"] = dict(positions)
        update_setting("layout_positions", positions)

    def _animate_current_tab(self, index):
        page = self.tabs.widget(index)
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", page)
        animation.setDuration(170)
        animation.setStartValue(0.62)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.finished.connect(lambda: page.setGraphicsEffect(None))
        page._fade_animation = animation
        animation.start()

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
        self.progress_animation = QPropertyAnimation(self.progress, b"value", self)
        self.progress_animation.setDuration(180)
        self.progress_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.progress_label = QLabel("Ready")
        self.progress_label.setObjectName("progressLabel")

        self.render_button = QPushButton("Render video")
        self.render_button.setObjectName("primaryButton")
        self.render_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.render_button.clicked.connect(self.start_render)

        self.stop_button = QPushButton("Stop render")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.cancel_render)

        self.update_button = QPushButton("Check for updates")
        self.update_button.clicked.connect(self.check_updates)

        layout.addWidget(self.progress)
        layout.addWidget(self.progress_label)
        commands = QHBoxLayout()
        commands.setSpacing(10)
        commands.addWidget(self.render_button, 3)
        commands.addWidget(self.stop_button, 2)
        layout.addLayout(commands)
        layout.addWidget(self.update_button)
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
            QLabel#lookupStatus { color: #aeb4ba; font-size: 12px; }
            QLabel#layoutStatus { color: #aeb4ba; font-weight: 600; }
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
            QPushButton#stopButton {
                background: #3a2528;
                border-color: #704047;
                color: #f4dfe2;
                font-weight: 700;
                min-height: 40px;
            }
            QPushButton#stopButton:hover { background: #4a2c31; border-color: #a35560; }
            QPushButton#stopButton:disabled { background: #242628; border-color: #34373a; color: #6f7478; }
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
            QTabWidget::pane {
                border: 0;
                background: transparent;
                top: -1px;
            }
            QTabBar { qproperty-drawBase: 0; }
            QTabBar::tab {
                min-width: 112px;
                min-height: 34px;
                padding: 0 14px;
                color: #9fa5aa;
                background: transparent;
                border-bottom: 2px solid transparent;
                font-weight: 600;
            }
            QTabBar::tab:hover { color: #e8ecef; background: #202326; }
            QTabBar::tab:selected { color: #ffffff; border-bottom-color: #59bdcc; }
            QGraphicsView#layoutPreview {
                border: 1px solid #34383c;
                border-radius: 7px;
                background: #050607;
            }
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
            "resolution": self.resolution_combo.currentText(),
            "motion_blur": self.motion_blur_spin.value(),
            "show_side_overlay": self.side_overlay_check.isChecked(),
            "show_strain_graph": self.strain_graph_check.isChecked(),
            "vignette_strength": self.vignette_spin.value(),
            "results_background_opacity": self.results_opacity_spin.value() / 100.0,
            "results_duration": self.results_duration_spin.value(),
            "show_results_screen": self.results_screen_check.isChecked(),
            "colour_combo_during_holds": self.hold_combo_colour_check.isChecked(),
            "layout_positions": self.layout_editor.positions(),
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
        update_setting("layout_positions", options["layout_positions"])

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
        self.stop_button.setEnabled(True)

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

        if percent >= 85:
            self.stop_button.setEnabled(False)

    def cancel_render(self):
        if not self.thread or not self.thread.isRunning():
            return

        self.thread.cancel()
        self.stop_button.setEnabled(False)
        self.progress_label.setText("Stopping render...")

    def render_done(self, output_file):
        self.render_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setValue(100)
        self.progress_label.setText("Render complete")
        QMessageBox.information(self, "Render complete", f"Video created:\n{output_file}")

    def render_failed(self, error):
        self.render_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_label.setText("Render failed")
        QMessageBox.critical(self, "Render error", error)

    def render_cancelled(self):
        self.render_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_animation.stop()
        self.progress.setValue(0)
        self.progress_label.setText("Render cancelled")

    def check_updates(self):
        if self.update_thread and self.update_thread.isRunning():
            return

        self.update_button.setEnabled(False)
        self.update_button.setText("Checking...")
        self.update_thread = UpdateCheckThread(self)
        self.update_thread.completed.connect(self.update_check_done)
        self.update_thread.failed.connect(self.update_check_failed)
        self.update_thread.start()

    def update_check_done(self, update):
        self.update_button.setEnabled(True)
        self.update_button.setText("Check for updates")

        if update is None:
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

    def update_check_failed(self, error):
        self.update_button.setEnabled(True)
        self.update_button.setText("Check for updates")
        answer = QMessageBox.question(
            self,
            "Update check unavailable",
            f"{error}\n\nOpen GitHub Releases in your browser instead?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if answer == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl(RELEASES_URL))
