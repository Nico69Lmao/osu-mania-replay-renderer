from pathlib import Path
from hashlib import md5
import os
import platform
import re
import threading
import time

from osrparse import Replay

DT = 64
HT = 256
NC = 512
MR = 1073741824
_HASH_CACHE = {}
_HASH_CACHE_LOCK = threading.Lock()


def osu_folder_score(path):
    folder = Path(path).expanduser()

    if not folder.is_dir():
        return 0

    score = 0
    score += 5 if (folder / "Songs").is_dir() else 0
    score += 2 if (folder / "Skins").is_dir() else 0
    score += 3 if (folder / "osu!.db").is_file() else 0
    score += 1 if (folder / "osu!.exe").is_file() else 0
    return score


def is_osu_folder(path):
    return osu_folder_score(path) >= 5


def _windows_registry_candidates():
    if platform.system().lower() != "windows":
        return []

    try:
        import winreg
    except ImportError:
        return []

    candidates = []
    registry_paths = (
        (winreg.HKEY_CURRENT_USER, r"Software\Classes\osu!\shell\open\command"),
        (winreg.HKEY_CLASSES_ROOT, r"osu!\shell\open\command"),
    )

    for root, key_path in registry_paths:
        try:
            with winreg.OpenKey(root, key_path) as key:
                command = str(winreg.QueryValue(key, None))
        except OSError:
            continue

        match = re.match(r'^"([^"]+\.exe)"|^([^\s]+\.exe)', command, re.IGNORECASE)

        if match:
            candidates.append(Path(match.group(1) or match.group(2)).parent)

    return candidates


def osu_folder_candidates(system=None, home=None, environ=None):
    system = (system or platform.system()).lower()
    home = Path(home or Path.home()).expanduser()
    environ = environ or os.environ
    candidates = []

    for variable in ("OSU_FOLDER", "OSU_PATH"):
        if environ.get(variable):
            candidates.append(Path(environ[variable]).expanduser())

    if system == "windows":
        for variable in ("LOCALAPPDATA", "APPDATA", "USERPROFILE", "PROGRAMFILES", "PROGRAMFILES(X86)"):
            value = environ.get(variable)

            if not value:
                continue

            base = Path(value)
            candidates.extend((base / "osu!", base / "AppData" / "Local" / "osu!"))

        candidates.extend(_windows_registry_candidates())
    else:
        candidates.extend((
            home / ".local" / "share" / "osu-wine" / "osu!",
            home / "osu!",
            home / ".osu" / "osu!",
            home / ".wine" / "drive_c" / "osu!",
        ))

        patterns = (
            ".local/share/osu-wine/*/drive_c/users/*/AppData/Local/osu!",
            ".local/share/osu-wine/prefix/drive_c/users/*/AppData/Local/osu!",
            ".wine/drive_c/users/*/AppData/Local/osu!",
            "Games/*/drive_c/users/*/AppData/Local/osu!",
            ".local/share/lutris/prefixes/*/drive_c/users/*/AppData/Local/osu!",
        )

        for pattern in patterns:
            candidates.extend(home.glob(pattern))

    unique = []
    seen = set()

    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate)))

        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    return unique


def find_osu_folder(system=None, home=None, environ=None):
    scored = [
        (osu_folder_score(candidate), index, candidate)
        for index, candidate in enumerate(osu_folder_candidates(system, home, environ))
    ]
    valid = [item for item in scored if item[0] >= 5]

    if not valid:
        return None

    _, _, best = max(valid, key=lambda item: (item[0], -item[1]))
    return str(best.resolve())


def list_skins(osu_folder: str):
    skins_dir = Path(osu_folder) / "Skins"

    if not skins_dir.exists():
        return []

    return sorted([p.name for p in skins_dir.iterdir() if p.is_dir()])


def get_replay(replay_path: str):
    return Replay.from_path(replay_path)


def mods_to_int(mods):
    try:
        return int(mods)
    except Exception:
        try:
            return int(mods.value)
        except Exception:
            return 0


def mod_settings_from_replay(replay):
    mods = mods_to_int(replay.mods)

    names = []
    speed_multiplier = 1.0
    nightcore_pitch = False

    if mods & NC:
        names.append("NC")
        speed_multiplier = 1.5
        nightcore_pitch = True
    elif mods & DT:
        names.append("DT")
        speed_multiplier = 1.5
    elif mods & HT:
        names.append("HT")
        speed_multiplier = 0.75

    mirror = bool(mods & MR)

    if mirror:
        names.append("MR")

    if not names:
        names.append("NM")

    return {
        "mods_int": mods,
        "mods": " ".join(names),
        "speed_multiplier": speed_multiplier,
        "mirror": mirror,
        "nightcore_pitch": nightcore_pitch,
    }


def get_mod_settings(replay_path: str):
    return mod_settings_from_replay(get_replay(replay_path))


def get_replay_info(replay_path: str):
    replay = get_replay(replay_path)
    mod_info = mod_settings_from_replay(replay)

    return {
        "beatmap_hash": replay.beatmap_hash,
        "username": replay.username,
        "score": replay.score,
        "max_combo": getattr(replay, "max_combo", 0),
        "mods": mod_info["mods"],
    }


def get_stable_mania_accuracy(replay_path: str):
    replay = get_replay(replay_path)

    c300 = getattr(replay, "count_300", 0)
    c100 = getattr(replay, "count_100", 0)
    c50 = getattr(replay, "count_50", 0)
    cmiss = getattr(replay, "count_miss", 0)
    cgeki = getattr(replay, "count_geki", 0)
    ckatu = getattr(replay, "count_katu", 0)

    total = c300 + c100 + c50 + cmiss + cgeki + ckatu

    if total <= 0:
        return 100.0

    return ((50 * c50) + (100 * c100) + (200 * ckatu) + (300 * (c300 + cgeki))) / (300 * total) * 100


def _cached_file_md5(path):
    stat = path.stat()
    cache_key = str(path)
    signature = (stat.st_mtime_ns, stat.st_size)

    with _HASH_CACHE_LOCK:
        cached = _HASH_CACHE.get(cache_key)

    if cached and cached[:2] == signature:
        return cached[2]

    digest = md5()

    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    value = digest.hexdigest()

    with _HASH_CACHE_LOCK:
        _HASH_CACHE[cache_key] = (*signature, value)

    return value


def _collect_osu_files(songs_dir, cancel_callback=None):
    files = []

    for root, _, names in os.walk(songs_dir):
        if cancel_callback and cancel_callback():
            return None

        folder = Path(root)
        files.extend(folder / name for name in names if name.lower().endswith(".osu"))

    return files


def find_beatmap_by_hash(
    osu_folder,
    target_hash,
    progress_callback=None,
    cancel_callback=None,
    preferred_path=None,
):
    target_hash = str(target_hash).lower()
    songs_dir = Path(osu_folder) / "Songs"

    if not songs_dir.exists():
        return None

    preferred = Path(preferred_path) if preferred_path else None

    if preferred and preferred.is_file():
        try:
            if _cached_file_md5(preferred) == target_hash:
                if progress_callback:
                    progress_callback(1, 1, 0.0)

                return str(preferred)
        except OSError:
            pass

    if progress_callback:
        progress_callback(0, 0, None)

    osu_files = _collect_osu_files(songs_dir, cancel_callback)

    if osu_files is None:
        return None

    total = len(osu_files)
    started = time.monotonic()
    last_update = 0.0

    for checked, osu_file in enumerate(osu_files, 1):
        if cancel_callback and cancel_callback():
            return None

        try:
            if _cached_file_md5(osu_file) == target_hash.lower():
                if progress_callback:
                    progress_callback(checked, total, 0.0)

                return str(osu_file)
        except Exception:
            pass

        now = time.monotonic()

        if progress_callback and (checked == total or now - last_update >= 0.15):
            elapsed = max(0.001, now - started)
            rate = checked / elapsed
            eta = (total - checked) / rate if rate > 0 else None
            progress_callback(checked, total, eta)
            last_update = now

    return None


def find_beatmap_from_replay(
    osu_folder,
    replay_path,
    progress_callback=None,
    cancel_callback=None,
    preferred_path=None,
):
    target_hash = get_replay_info(replay_path)["beatmap_hash"].lower()
    return find_beatmap_by_hash(
        osu_folder,
        target_hash,
        progress_callback,
        cancel_callback,
        preferred_path,
    )
