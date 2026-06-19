import json
from pathlib import Path

APP_DIR = Path.home() / ".config" / "mania-renderer"
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
