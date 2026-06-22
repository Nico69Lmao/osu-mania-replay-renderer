import json
import os
import platform
from pathlib import Path


def config_directory():
    if platform.system().lower() == "windows" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "mania-renderer"

    if os.environ.get("XDG_CONFIG_HOME"):
        return Path(os.environ["XDG_CONFIG_HOME"]) / "mania-renderer"

    return Path.home() / ".config" / "mania-renderer"


APP_DIR = config_directory()
SETTINGS_FILE = APP_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "osu_folder": "",
    "last_replay": "",
    "last_beatmap": "",
    "last_output_folder": "",
    "last_skin": "",
    "scroll_speed": "30.0",
    "motion_blur": "0",
    "resolution": "1280x720",
    "show_side_overlay": True,
    "show_strain_graph": True,
    "vignette_strength": "12",
    "results_background_opacity": "62",
    "results_duration": "4.5",
    "show_results_screen": True,
    "colour_combo_during_holds": True,
    "layout_positions": {},
}


def load_settings():
    APP_DIR.mkdir(parents=True, exist_ok=True)

    if not SETTINGS_FILE.exists():
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        settings = DEFAULT_SETTINGS.copy()
        settings.update(data)
        return settings
    except Exception:
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    APP_DIR.mkdir(parents=True, exist_ok=True)

    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


def update_setting(key, value):
    settings = load_settings()
    settings[key] = value
    save_settings(settings)
