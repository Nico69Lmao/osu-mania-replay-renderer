from pathlib import Path
import cv2


def read_image(path):
    if not path:
        return None

    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def restore_alpha_from_rgb(img):
    if img is None or len(img.shape) < 3 or img.shape[2] < 4:
        return img

    if img[:, :, 3].max() > 0:
        return img

    rgb_visible = img[:, :, :3].max(axis=2)

    if rgb_visible.max() == 0:
        return img

    img = img.copy()
    img[:, :, 3] = rgb_visible
    return img


def clean_key(key: str):
    return key.strip()


def clean_value(value: str):
    return value.split("//", 1)[0].strip()


def parse_csv_ints(value, count=None):
    values = []

    for item in value.replace("|", ",").split(","):
        item = item.strip()

        if not item:
            continue

        try:
            values.append(int(float(item)))
        except ValueError:
            pass

    if count and values:
        while len(values) < count:
            values.append(values[-1])

        values = values[:count]

    return values


def resolve_case_insensitive(path: Path):
    if path.exists():
        return path

    parts = path.parts

    if not parts:
        return path

    current = Path(parts[0])

    for part in parts[1:]:
        candidate = current / part

        if candidate.exists():
            current = candidate
            continue

        if not current.exists() or not current.is_dir():
            return path

        match = None
        part_lower = part.lower()

        for child in current.iterdir():
            if child.name.lower() == part_lower:
                match = child
                break

        if match is None:
            return path

        current = match

    return current


def parse_skin_ini(skin_folder: Path, keys: int):
    ini = skin_folder / "skin.ini"

    cfg = {
        "keys": keys,
        "column_start": None,
        "hit_position": None,
        "column_widths": None,
        "column_spacing": [0] * (keys - 1),
        "column_line_widths": [1] * (keys + 1),
        "barline_height": 0,
        "score_position": None,
        "combo_position": None,
        "light_position": None,
        "keys_under_notes": False,
        "upside_down": False,
        "judgement_line": True,
        "colours": {},
        "images": {},
    }

    if not ini.exists():
        return cfg

    current_section = None
    mania_blocks = []
    current_mania = None

    with open(ini, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            if not line or line.startswith("//") or line.startswith(";"):
                continue

            if line.startswith("[") and line.endswith("]"):
                if current_section == "Mania" and current_mania:
                    mania_blocks.append(current_mania)

                current_section = line[1:-1]
                current_mania = {} if current_section == "Mania" else None
                continue

            if current_section != "Mania":
                continue

            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = clean_key(key)
            value = clean_value(value)

            if current_mania is not None:
                current_mania[key] = value

        if current_section == "Mania" and current_mania:
            mania_blocks.append(current_mania)

    block = None

    for b in mania_blocks:
        try:
            if int(float(b.get("Keys", -1))) == keys:
                block = b
                break
        except Exception:
            pass

    if not block:
        return cfg

    def get_int(name):
        try:
            return int(float(block[name]))
        except Exception:
            return None

    cfg["column_start"] = get_int("ColumnStart")
    cfg["hit_position"] = get_int("HitPosition")
    cfg["column_widths"] = parse_csv_ints(block.get("ColumnWidth", ""), keys) or None
    cfg["column_spacing"] = parse_csv_ints(block.get("ColumnSpacing", "")) or [0] * (keys - 1)
    cfg["column_line_widths"] = parse_csv_ints(block.get("ColumnLineWidth", ""), keys + 1) or [1] * (keys + 1)
    cfg["barline_height"] = get_int("BarlineHeight") or 0
    cfg["score_position"] = get_int("ScorePosition")
    cfg["combo_position"] = get_int("ComboPosition")
    cfg["light_position"] = get_int("LightPosition")
    cfg["keys_under_notes"] = block.get("KeysUnderNotes", "0") == "1"
    cfg["upside_down"] = block.get("UpsideDown", "0") == "1"
    cfg["judgement_line"] = block.get("JudgementLine", "1") != "0"

    for k, v in block.items():
        if k.startswith("Colour"):
            cfg["colours"][k] = parse_csv_ints(v)

        if (
            k.startswith("KeyImage")
            or k.startswith("NoteImage")
            or k.startswith("Stage")
            or k.startswith("Hit")
        ):
            cfg["images"][k] = v

    return cfg


def find_image(folder: Path, name: str | None):
    if not name:
        return None

    name = name.strip().replace("\\", "/")
    name_path = Path(name)

    candidates = []

    for ext in ["", ".png", ".jpg", ".jpeg"]:
        candidates.append(folder / f"{name}{ext}")

        if ext:
            candidates.append(folder / f"{name}-0{ext}")
            candidates.append(folder / f"{name}@2x{ext}")
            candidates.append(folder / f"{name_path.parent}/{name_path.stem}-0{ext}")
            candidates.append(folder / f"{name_path.parent}/{name_path.stem}@2x{ext}")

    for path in candidates:
        path = resolve_case_insensitive(path)

        if path.exists():
            return path

        parent = path.parent

        if parent.exists():
            target = path.name.lower()

            for candidate in parent.iterdir():
                if candidate.name.lower() == target:
                    return candidate

    parent = folder / name_path.parent

    if parent.exists():
        stem = name_path.stem.lower()

        for candidate in parent.iterdir():
            candidate_stem = candidate.stem.lower()

            if candidate_stem == stem or candidate_stem.replace("-", "") == stem.replace("-", ""):
                return candidate

    return None


def load_mania_skin(skin_folder: str | None, keys: int):
    empty = {
        "cfg": {
            "column_start": None,
            "hit_position": None,
            "column_widths": None,
            "column_spacing": [0] * (keys - 1),
            "column_line_widths": [1] * (keys + 1),
            "keys_under_notes": False,
            "upside_down": False,
            "judgement_line": True,
            "colours": {},
            "images": {},
        },
        "notes": [None] * keys,
        "ln_heads": [None] * keys,
        "ln_bodies": [None] * keys,
        "ln_tails": [None] * keys,
        "keys": [None] * keys,
        "keys_down": [None] * keys,
        "hit_images": {},
        "stage_left": None,
        "stage_right": None,
        "stage_bottom": None,
        "stage_light": None,
        "stage_hint": None,
        "ranking_panel": None,
        "ranking_ranks": {},
    }

    if not skin_folder:
        return empty

    folder = Path(skin_folder)

    if not folder.exists():
        return empty

    cfg = parse_skin_ini(folder, keys)
    images = cfg["images"]

    skin = empty.copy()
    skin["cfg"] = cfg
    skin["notes"] = []
    skin["ln_heads"] = []
    skin["ln_bodies"] = []
    skin["ln_tails"] = []
    skin["keys"] = []
    skin["keys_down"] = []
    skin["hit_images"] = {}

    for lane in range(keys):
        note = images.get(f"NoteImage{lane}")
        ln_head = images.get(f"NoteImage{lane}H") or note
        ln_body = images.get(f"NoteImage{lane}L") or ln_head
        ln_tail = images.get(f"NoteImage{lane}T") or ln_head

        key = images.get(f"KeyImage{lane}")
        key_down = images.get(f"KeyImage{lane}D") or key

        note_img = read_image(find_image(folder, note))
        ln_head_img = read_image(find_image(folder, ln_head))

        if ln_head_img is None:
            ln_head_img = note_img

        skin["notes"].append(note_img)
        skin["ln_heads"].append(ln_head_img)
        skin["ln_bodies"].append(read_image(find_image(folder, ln_body)))
        skin["ln_tails"].append(read_image(find_image(folder, ln_tail)))
        skin["keys"].append(read_image(find_image(folder, key)))
        skin["keys_down"].append(read_image(find_image(folder, key_down)))

    skin["stage_left"] = read_image(find_image(folder, images.get("StageLeft")))
    skin["stage_right"] = read_image(find_image(folder, images.get("StageRight")))
    skin["stage_bottom"] = read_image(find_image(folder, images.get("StageBottom")) or find_image(folder, "mania-stage-bottom"))
    skin["stage_light"] = read_image(find_image(folder, images.get("StageLight")) or find_image(folder, "mania-stage-light"))
    skin["stage_hint"] = read_image(find_image(folder, images.get("StageHint")) or find_image(folder, "mania-stage-hint"))
    skin["ranking_panel"] = read_image(find_image(folder, "ranking-panel"))
    skin["ranking_ranks"] = {
        rank: read_image(find_image(folder, f"ranking-{rank.lower()}"))
        for rank in ("X", "XH", "S", "SH", "A", "B", "C", "D")
    }

    for value in ("0", "50", "100", "200", "300", "300g"):
        image_name = images.get(f"Hit{value}")
        skin["hit_images"][value] = read_image(find_image(folder, image_name))

    if skin["hit_images"].get("300g") is None:
        skin["hit_images"]["300g"] = skin["hit_images"].get("300")

    for lane in range(keys):
        if skin["ln_bodies"][lane] is None:
            note_name = images.get(f"NoteImage{lane}L", "")
            fallback = note_name.replace("noteL", "NoteL").replace("notel", "NoteL")
            skin["ln_bodies"][lane] = read_image(find_image(folder, fallback))

        if skin["ln_tails"][lane] is None:
            note_name = images.get(f"NoteImage{lane}T", "")
            fallback = note_name.replace("noteT", "NoteT").replace("notet", "NoteT")
            skin["ln_tails"][lane] = read_image(find_image(folder, fallback))

    return skin
