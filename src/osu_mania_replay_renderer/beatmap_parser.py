from dataclasses import dataclass
from hashlib import md5
import csv


@dataclass
class Note:
    lane: int
    time: int
    end_time: int | None = None


@dataclass
class Beatmap:
    title: str
    artist: str
    version: str
    creator: str
    audio_file: str
    background_file: str
    md5_hash: str
    keys: int
    overall_difficulty: float
    mode: int
    notes: list[Note]


def parse_osu(path: str):
    title = "Unknown"
    artist = "Unknown"
    version = "Unknown"
    creator = "Unknown"
    audio_file = ""
    background_file = ""
    keys = 4
    overall_difficulty = 5.0
    mode = 3
    notes = []
    section = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    with open(path, "rb") as f:
        md5_hash = md5(f.read()).hexdigest()

    for line in lines:
        line = line.strip()

        if not line or line.startswith("//"):
            continue

        if line.startswith("[") and line.endswith("]"):
            section = line
            continue

        if section == "[General]":
            if line.startswith("AudioFilename:"):
                audio_file = line.split(":", 1)[1].strip()
            elif line.startswith("Mode:"):
                mode = int(float(line.split(":", 1)[1].strip()))

        elif section == "[Metadata]":
            if line.startswith("Title:"):
                title = line.split(":", 1)[1].strip()
            elif line.startswith("Artist:"):
                artist = line.split(":", 1)[1].strip()
            elif line.startswith("Version:"):
                version = line.split(":", 1)[1].strip()
            elif line.startswith("Creator:"):
                creator = line.split(":", 1)[1].strip()

        elif section == "[Difficulty]":
            if line.startswith("CircleSize:"):
                keys = int(float(line.split(":", 1)[1].strip()))
            elif line.startswith("OverallDifficulty:"):
                overall_difficulty = float(line.split(":", 1)[1].strip())

        elif section == "[Events]":
            try:
                parts = next(csv.reader([line]))

                if len(parts) >= 3 and parts[0].strip() == "0" and parts[1].strip() == "0":
                    background_file = parts[2].strip()
            except (csv.Error, StopIteration):
                pass

        elif section == "[HitObjects]":
            parts = line.split(",")

            if len(parts) < 5:
                continue

            x = int(parts[0])
            time = int(parts[2])
            obj_type = int(parts[3])
            lane = min(keys - 1, max(0, int(x * keys / 512)))

            if obj_type & 128 and len(parts) > 5:
                end_time = int(parts[5].split(":")[0])
                notes.append(Note(lane=lane, time=time, end_time=end_time))
            else:
                notes.append(Note(lane=lane, time=time))

    notes.sort(key=lambda n: n.time)

    return Beatmap(
        title=title,
        artist=artist,
        version=version,
        creator=creator,
        audio_file=audio_file,
        background_file=background_file,
        md5_hash=md5_hash,
        keys=keys,
        overall_difficulty=overall_difficulty,
        mode=mode,
        notes=notes,
    )
