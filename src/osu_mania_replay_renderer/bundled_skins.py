from pathlib import Path
import sys


BUNDLED_SKINS = (
    {
        "label": "★ Nico69_ v4 — Verified [4k]",
        "folder": "Nico69_ v4",
        "source_name": "-       《69》 Nico69_ v4",
        "key_count": 4,
        "aliases": ("nico69", "nico69 v4", "nico69_ v4", "4k"),
    },
    {
        "label": "★ Cawolo skin new Max — Verified [7k]",
        "folder": "Cawolo skin new Max",
        "source_name": "Cawolo skin new Max",
        "key_count": 7,
        "aliases": ("cawolo", "cawolo new max", "new max", "7k"),
    },
)


def bundled_skin_root():
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "osu_mania_replay_renderer" / "bundled_skins"

    return Path(__file__).resolve().parent / "bundled_skins"


def bundled_skin_entries():
    root = bundled_skin_root()
    entries = []

    for skin in BUNDLED_SKINS:
        path = root / skin["folder"]

        if path.is_dir():
            entry = dict(skin)
            entry["path"] = str(path)
            entries.append(entry)

    return entries


def bundled_skin_labels():
    return [entry["label"] for entry in bundled_skin_entries()]


def is_bundled_skin_label(label):
    text = str(label or "")
    return any(text == entry["label"] for entry in bundled_skin_entries())


def bundled_skin_path(label):
    text = str(label or "")

    for entry in bundled_skin_entries():
        if text in {entry["label"], entry["source_name"], entry["folder"]}:
            return entry["path"]

    return None


def bundled_skin_key_count(label):
    text = str(label or "")

    for entry in bundled_skin_entries():
        if text in {entry["label"], entry["source_name"], entry["folder"]}:
            return entry["key_count"]

    return None


def matching_bundled_skin(query):
    needle = str(query or "").strip().lower()

    if not needle:
        return None

    for entry in bundled_skin_entries():
        haystack = " ".join(
            [
                entry["label"],
                entry["folder"],
                entry["source_name"],
                *entry.get("aliases", ()),
            ]
        ).lower()

        if needle in haystack or all(token in haystack for token in needle.replace("_", " ").replace("-", " ").split()):
            return entry["label"]

    return None
