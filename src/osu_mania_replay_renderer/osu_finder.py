from pathlib import Path
from hashlib import md5
import json
import os
import platform
import re
import threading
import time

from osrparse import Replay

from osu_mania_replay_renderer.osu_db_reader import beatmap_path_by_hash
from osu_mania_replay_renderer.settings import APP_DIR

DT = 64
HT = 256
NC = 512
SV2 = 536870912
MR = 1073741824
_HASH_CACHE = {}
_HASH_CACHE_LOCK = threading.Lock()
_BEATMAP_LOOKUP_FILE = APP_DIR / "beatmap_hash_cache.json"
_BEATMAP_LOOKUP_CACHE = None
_BEATMAP_LOOKUP_LOCK = threading.Lock()
_REPLAY_LIST_CACHE = {}
_REPLAY_LIST_LOCK = threading.Lock()


def _cache_osu_folder_key(osu_folder):
    try:
        return str(Path(osu_folder).expanduser().resolve())
    except Exception:
        return str(osu_folder)


def _load_beatmap_lookup_cache():
    global _BEATMAP_LOOKUP_CACHE
    with _BEATMAP_LOOKUP_LOCK:
        if _BEATMAP_LOOKUP_CACHE is not None:
            return _BEATMAP_LOOKUP_CACHE

        try:
            with open(_BEATMAP_LOOKUP_FILE, "r", encoding="utf-8") as stream:
                data = json.load(stream)
        except Exception:
            data = {}

        if not isinstance(data, dict):
            data = {}

        _BEATMAP_LOOKUP_CACHE = data
        return _BEATMAP_LOOKUP_CACHE


def _save_beatmap_lookup_cache(cache):
    APP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = _BEATMAP_LOOKUP_FILE.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as stream:
        json.dump(cache, stream, ensure_ascii=False, indent=2)
    tmp_file.replace(_BEATMAP_LOOKUP_FILE)


def _cached_beatmap_path(osu_folder, target_hash):
    cache = _load_beatmap_lookup_cache()
    folder_cache = cache.get(_cache_osu_folder_key(osu_folder), {})

    if not isinstance(folder_cache, dict):
        return None

    cached = folder_cache.get(str(target_hash).lower())

    if not cached:
        return None

    path = Path(cached)

    if not path.is_file():
        return None

    try:
        if _cached_file_md5(path) == str(target_hash).lower():
            return str(path)
    except OSError:
        return None

    return None


def _remember_beatmap_path(osu_folder, target_hash, osu_file):
    if not osu_file:
        return

    cache = _load_beatmap_lookup_cache()
    folder_key = _cache_osu_folder_key(osu_folder)

    with _BEATMAP_LOOKUP_LOCK:
        folder_cache = cache.setdefault(folder_key, {})
        folder_cache[str(target_hash).lower()] = str(osu_file)
        try:
            _save_beatmap_lookup_cache(cache)
        except OSError:
            pass


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


def list_recent_replays(osu_folder: str, limit=200, query=""):
    replays_dir = Path(osu_folder) / "Replays"

    if not replays_dir.is_dir():
        return []

    needle = str(query or "").strip().lower()
    try:
        signature = (str(replays_dir.resolve()), replays_dir.stat().st_mtime_ns)
    except OSError:
        return []

    with _REPLAY_LIST_LOCK:
        cached = _REPLAY_LIST_CACHE.get(signature)

    if cached is None:
        entries = []

        try:
            with os.scandir(replays_dir) as iterator:
                for entry in iterator:
                    if not entry.is_file() or not entry.name.lower().endswith(".osr"):
                        continue

                    try:
                        stat = entry.stat()
                    except OSError:
                        continue

                    entries.append((stat.st_mtime_ns, entry.name, entry.path, entry.name.lower()))
        except OSError:
            return []

        entries.sort(reverse=True)

        with _REPLAY_LIST_LOCK:
            _REPLAY_LIST_CACHE.clear()
            _REPLAY_LIST_CACHE[signature] = entries
    else:
        entries = cached

    if needle:
        entries = [entry for entry in entries if needle in entry[3]]

    return [
        {
            "name": name,
            "path": path,
            "mtime_ns": mtime_ns,
        }
        for mtime_ns, name, path, _ in entries[:max(1, int(limit))]
    ]


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
    score_v2 = bool(mods & SV2)

    if mirror:
        names.append("MR")
    if score_v2:
        names.append("V2")

    if not names:
        names.append("NM")

    return {
        "mods_int": mods,
        "mods": " ".join(names),
        "speed_multiplier": speed_multiplier,
        "mirror": mirror,
        "score_v2": score_v2,
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
    cached = _cached_beatmap_path(osu_folder, target_hash)

    if cached:
        if progress_callback:
            progress_callback(1, 1, 0.0)

        return cached

    if preferred and preferred.is_file():
        try:
            if _cached_file_md5(preferred) == target_hash:
                if progress_callback:
                    progress_callback(1, 1, 0.0)

                _remember_beatmap_path(osu_folder, target_hash, preferred)
                return str(preferred)
        except OSError:
            pass

    if progress_callback:
        progress_callback(0, 100, None)

    try:
        found = beatmap_path_by_hash(osu_folder, target_hash)

        if found:
            if progress_callback:
                progress_callback(100, 100, 0.0)

            _remember_beatmap_path(osu_folder, target_hash, found)
            return found
    except Exception:
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

                _remember_beatmap_path(osu_folder, target_hash, osu_file)
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
