from pathlib import Path
from hashlib import md5
from osrparse import Replay

DT = 64
HT = 256
NC = 512
MR = 1073741824


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


def get_mod_settings(replay_path: str):
    replay = get_replay(replay_path)
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


def get_replay_info(replay_path: str):
    replay = get_replay(replay_path)
    mod_info = get_mod_settings(replay_path)

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


def find_beatmap_from_replay(osu_folder: str, replay_path: str):
    target_hash = get_replay_info(replay_path)["beatmap_hash"].lower()
    songs_dir = Path(osu_folder) / "Songs"

    if not songs_dir.exists():
        return None

    for osu_file in songs_dir.rglob("*.osu"):
        try:
            with open(osu_file, "rb") as f:
                if md5(f.read()).hexdigest() == target_hash:
                    return str(osu_file)
        except Exception:
            pass

    return None
