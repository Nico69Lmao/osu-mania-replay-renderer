from pathlib import Path
import cv2
import numpy as np


def read_image(path):
    if not path:
        return None

    path = Path(path)

    try:
        # cv2.imread can fail on Windows when the osu!/skin path contains
        # unicode or decorative characters (for example "《69》 Nico69_ v4").
        # Reading bytes through Python first keeps path handling native and
        # lets OpenCV only decode the image payload.
        image_bytes = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED) if image_bytes.size else None
    except Exception:
        image = None

    if image is not None:
        # Skin textures are immutable after loading. Marking them read-only lets
        # renderer workers safely cache resized copies without caching temporary
        # tinted/animated images.
        image.flags.writeable = False

    return image


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
        "skin_version": 1.0,
        "combo_prefix": "score",
        "combo_overlap": 0,
        "score_prefix": "score",
        "score_overlap": 0,
        "combo_colours": [],
        "column_start": None,
        "hit_position": None,
        "column_widths": None,
        "column_spacing": [0] * (keys - 1),
        "column_line_widths": [1] * (keys + 1),
        "barline_height": 0,
        "score_position": None,
        "combo_position": None,
        "light_position": None,
        "light_frame_per_second": 60,
        "lighting_n_widths": None,
        "lighting_l_widths": None,
        "width_for_note_height_scale": None,
        "note_body_style": 1,
        "note_body_styles": [1] * keys,
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

            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = clean_key(key)
            value = clean_value(value)

            if current_section == "General" and key == "Version":
                try:
                    cfg["skin_version"] = float(value)
                except ValueError:
                    pass

            if current_section == "Fonts":
                if key == "ComboPrefix":
                    cfg["combo_prefix"] = value
                elif key == "ComboOverlap":
                    try:
                        cfg["combo_overlap"] = int(float(value))
                    except ValueError:
                        pass
                elif key == "ScorePrefix":
                    cfg["score_prefix"] = value
                elif key == "ScoreOverlap":
                    try:
                        cfg["score_overlap"] = int(float(value))
                    except ValueError:
                        pass

            if current_section == "Colours" and key.lower().startswith("combo"):
                suffix = key[5:]

                if suffix.isdigit():
                    colour = parse_csv_ints(value)

                    if len(colour) >= 3:
                        cfg["combo_colours"].append((int(suffix), colour[:3]))

            if current_section != "Mania":
                continue

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
        cfg["combo_colours"] = [colour for _, colour in sorted(cfg["combo_colours"])]
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
    cfg["light_frame_per_second"] = get_int("LightFramePerSecond") or 60
    cfg["width_for_note_height_scale"] = get_int("WidthForNoteHeightScale")
    cfg["note_body_style"] = get_int("NoteBodyStyle")
    if cfg["note_body_style"] is None:
        cfg["note_body_style"] = 1
    cfg["note_body_styles"] = [cfg["note_body_style"]] * keys
    for lane in range(keys):
        lane_style = get_int(f"NoteBodyStyle{lane}")
        if lane_style is not None:
            cfg["note_body_styles"][lane] = lane_style
    cfg["lighting_n_widths"] = parse_csv_ints(block.get("LightingNWidth", ""), keys) or None
    cfg["lighting_l_widths"] = parse_csv_ints(block.get("LightingLWidth", ""), keys) or None
    cfg["keys_under_notes"] = block.get("KeysUnderNotes", "0") == "1"
    cfg["upside_down"] = block.get("UpsideDown", "0") == "1"
    cfg["judgement_line"] = block.get("JudgementLine", "1") != "0"
    cfg["combo_colours"] = [colour for _, colour in sorted(cfg["combo_colours"])]

    for k, v in block.items():
        if k.startswith("Colour"):
            cfg["colours"][k] = parse_csv_ints(v)

        if (
            k.startswith("KeyImage")
            or k.startswith("NoteImage")
            or k.startswith("Stage")
            or k.startswith("Hit")
            or k.startswith("Lighting")
        ):
            cfg["images"][k] = v

    return cfg


def is_disabled_image_name(name: str | None) -> bool:
    if name is None:
        return False
    normalized = name.strip().replace("\\", "/").lower()
    stem = Path(normalized).stem
    return normalized in {"null", "none", "blank", "_blank"} or stem in {"null", "none", "blank", "_blank"}


def find_image(folder: Path, name: str | None, prefer_highres: bool = False):
    if not name:
        return None

    name = name.strip().replace("\\", "/")
    if is_disabled_image_name(name):
        return None
    name_path = Path(name)

    candidates = []

    extensions = ["", ".png", ".jpg", ".jpeg"]
    if prefer_highres:
        for ext in [".png", ".jpg", ".jpeg"]:
            candidates.append(folder / f"{name}@2x{ext}")
            candidates.append(folder / f"{name_path.parent}/{name_path.stem}@2x{ext}")

    for ext in extensions:
        candidates.append(folder / f"{name}{ext}")
        if ext:
            candidates.append(folder / f"{name}-0{ext}")
            if not prefer_highres:
                candidates.append(folder / f"{name}@2x{ext}")
            candidates.append(folder / f"{name_path.parent}/{name_path.stem}-0{ext}")
            if not prefer_highres:
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


def find_image_with_default(folder: Path, explicit_name: str | None, default_name: str, prefer_highres: bool = False):
    if is_disabled_image_name(explicit_name):
        return None
    return find_image(folder, explicit_name, prefer_highres=prefer_highres) or find_image(folder, default_name, prefer_highres=prefer_highres)


def find_animation_paths(folder: Path, name: str | None, prefer_highres: bool = False):
    if not name or is_disabled_image_name(name):
        return []

    name = name.strip().replace("\\", "/")
    name_path = Path(name)
    frames = []
    index = 0

    while True:
        candidates = []

        for ext in (".png", ".jpg", ".jpeg"):
            if prefer_highres:
                candidates.append(folder / f"{name_path.parent}/{name_path.stem}-{index}@2x{ext}")
                candidates.append(folder / f"{name}-{index}@2x{ext}")
            candidates.append(folder / f"{name_path.parent}/{name_path.stem}-{index}{ext}")
            candidates.append(folder / f"{name}-{index}{ext}")

        found = None

        for candidate in candidates:
            candidate = resolve_case_insensitive(candidate)
            if candidate.exists():
                found = candidate
                break

        if found is None:
            break

        frames.append(found)
        index += 1

    return frames


def find_animation_paths_with_default(folder: Path, explicit_name: str | None, default_name: str, prefer_highres: bool = False):
    frames = find_animation_paths(folder, explicit_name, prefer_highres=prefer_highres)
    if frames:
        return frames
    return find_animation_paths(folder, default_name, prefer_highres=prefer_highres)


def load_mania_skin(skin_folder: str | None, keys: int):
    empty = {
        "cfg": {
            "column_start": None,
            "skin_version": 1.0,
            "combo_prefix": "score",
            "combo_overlap": 0,
            "score_prefix": "score",
            "score_overlap": 0,
            "combo_colours": [],
            "hit_position": None,
            "column_widths": None,
            "column_spacing": [0] * (keys - 1),
            "column_line_widths": [1] * (keys + 1),
            "light_frame_per_second": 60,
            "lighting_n_widths": None,
            "lighting_l_widths": None,
            "width_for_note_height_scale": None,
            "note_body_style": 1,
            "keys_under_notes": False,
            "upside_down": False,
            "judgement_line": True,
            "colours": {},
            "images": {},
        },
        "notes": [None] * keys,
        "note_frames": [[] for _ in range(keys)],
        "ln_heads": [None] * keys,
        "ln_head_frames": [[] for _ in range(keys)],
        "ln_bodies": [None] * keys,
        "ln_body_frames": [[] for _ in range(keys)],
        "ln_tails": [None] * keys,
        "ln_tail_frames": [[] for _ in range(keys)],
        "ln_tail_explicit": [False] * keys,
        "keys": [None] * keys,
        "keys_down": [None] * keys,
        "hit_images": {},
        "hit_image_frames": {},
        "hit_image_densities": {},
        "stage_left": None,
        "stage_right": None,
        "stage_bottom": None,
        "stage_bottom_frames": [],
        "stage_light": None,
        "stage_light_frames": [],
        "stage_hint": None,
        "hit_lighting_normal": None,
        "hit_lighting_long": None,
        "hit_lighting_normal_density": 1.0,
        "hit_lighting_long_density": 1.0,
        "ranking_panel": None,
        "ranking_panel_density": 1.0,
        "ranking_ranks": {},
        "ranking_rank_densities": {},
        "ranking_elements": {},
        "ranking_element_densities": {},
        "ranking_hit_images": {},
        "mod_icons": {},
        "mod_icon_densities": {},
        "combo_glyphs": {},
        "score_glyphs": {},
        "missing_elements": [],
    }

    if not skin_folder:
        empty["missing_elements"] = [{"element": "skin_folder", "requested": "", "fallback": "", "status": "not configured"}]
        return empty

    folder = resolve_case_insensitive(Path(str(skin_folder).strip().strip('"')))

    if not folder.exists():
        empty["missing_elements"] = [{"element": "skin_folder", "requested": str(folder), "fallback": "", "status": "folder not found"}]
        return empty

    cfg = parse_skin_ini(folder, keys)
    images = cfg["images"]

    skin = empty.copy()
    skin["cfg"] = cfg
    skin["notes"] = []
    skin["note_frames"] = []
    skin["ln_heads"] = []
    skin["ln_head_frames"] = []
    skin["ln_bodies"] = []
    skin["ln_body_frames"] = []
    skin["ln_tails"] = []
    skin["ln_tail_frames"] = []
    skin["ln_tail_explicit"] = []
    skin["keys"] = []
    skin["keys_down"] = []
    skin["hit_images"] = {}
    skin["hit_image_frames"] = {}
    skin["hit_image_densities"] = {}
    skin["missing_elements"] = []

    def record_missing(element, requested=None, fallback=None, status="missing"):
        skin["missing_elements"].append({
            "element": element,
            "requested": requested or "",
            "fallback": fallback or "",
            "status": status,
        })

    def find_skin_image(element, requested, fallback=None, prefer_highres=False):
        if is_disabled_image_name(requested):
            record_missing(element, requested, fallback, "disabled in skin.ini")
            return None

        path = find_image(folder, requested, prefer_highres=prefer_highres)
        if path is not None:
            return path

        fallback_path = find_image(folder, fallback, prefer_highres=prefer_highres) if fallback else None
        if fallback_path is not None:
            record_missing(element, requested, fallback, "using fallback")
            return fallback_path

        record_missing(element, requested, fallback, "missing")
        return None

    def fallback_note_name(lane, suffix=""):
        if keys % 2 == 1 and lane == keys // 2:
            return f"mania-noteS{suffix}"
        return f"mania-note{1 if lane % 2 == 0 else 2}{suffix}"

    def fallback_key_name(lane, down=False):
        if keys % 2 == 1 and lane == keys // 2:
            base = "mania-keyS"
        else:
            base = f"mania-key{1 if lane % 2 == 0 else 2}"
        return f"{base}D" if down else base

    for lane in range(keys):
        note = images.get(f"NoteImage{lane}") or fallback_note_name(lane)
        tail_explicit = bool(images.get(f"NoteImage{lane}T"))
        if note.startswith("mania-note"):
            ln_head = images.get(f"NoteImage{lane}H") or f"{note}H"
            ln_body = images.get(f"NoteImage{lane}L") or f"{note}L"
            ln_tail = images.get(f"NoteImage{lane}T") or f"{note}T"
            tail_explicit = True
        else:
            ln_head = images.get(f"NoteImage{lane}H") or note
            ln_body = images.get(f"NoteImage{lane}L") or ln_head
            ln_tail = images.get(f"NoteImage{lane}T") or ln_head

        key = images.get(f"KeyImage{lane}") or fallback_key_name(lane)
        key_down = images.get(f"KeyImage{lane}D") or fallback_key_name(lane, down=True) or key

        note_img = read_image(find_skin_image(f"NoteImage{lane}", note))
        ln_head_img = read_image(find_skin_image(f"NoteImage{lane}H", ln_head))
        note_frames = [read_image(path) for path in find_animation_paths(folder, note)]
        ln_head_frames = [read_image(path) for path in find_animation_paths(folder, ln_head)]
        ln_body_frames = [read_image(path) for path in find_animation_paths(folder, ln_body)]
        ln_tail_frames = [read_image(path) for path in find_animation_paths(folder, ln_tail)]

        if ln_head_img is None:
            ln_head_img = note_img
        if note_img is None:
            note_img = ln_head_img

        skin["notes"].append(note_img)
        skin["note_frames"].append([frame for frame in note_frames if frame is not None])
        skin["ln_heads"].append(ln_head_img)
        skin["ln_head_frames"].append([frame for frame in ln_head_frames if frame is not None])
        skin["ln_bodies"].append(read_image(find_skin_image(f"NoteImage{lane}L", ln_body)))
        skin["ln_body_frames"].append([frame for frame in ln_body_frames if frame is not None])
        skin["ln_tails"].append(read_image(find_skin_image(f"NoteImage{lane}T", ln_tail)))
        skin["ln_tail_frames"].append([frame for frame in ln_tail_frames if frame is not None])
        skin["ln_tail_explicit"].append(tail_explicit)
        skin["keys"].append(read_image(find_skin_image(f"KeyImage{lane}", key)))
        skin["keys_down"].append(read_image(find_skin_image(f"KeyImage{lane}D", key_down)))

    skin["stage_left"] = read_image(find_skin_image("StageLeft", images.get("StageLeft"), "mania-stage-left", prefer_highres=True))
    skin["stage_right"] = read_image(find_skin_image("StageRight", images.get("StageRight"), "mania-stage-right", prefer_highres=True))
    skin["stage_bottom"] = read_image(find_skin_image("StageBottom", images.get("StageBottom"), "mania-stage-bottom", prefer_highres=True))
    stage_bottom_frames = find_animation_paths_with_default(folder, images.get("StageBottom"), "mania-stage-bottom", prefer_highres=True)
    skin["stage_bottom_frames"] = [read_image(path) for path in stage_bottom_frames]
    skin["stage_bottom_frames"] = [frame for frame in skin["stage_bottom_frames"] if frame is not None]
    skin["stage_light"] = read_image(find_skin_image("StageLight", images.get("StageLight"), "mania-stage-light"))
    stage_light_frames = find_animation_paths_with_default(folder, images.get("StageLight"), "mania-stage-light")
    skin["stage_light_frames"] = [read_image(path) for path in stage_light_frames]
    skin["stage_light_frames"] = [frame for frame in skin["stage_light_frames"] if frame is not None]
    skin["stage_hint"] = read_image(find_skin_image("StageHint", images.get("StageHint"), "mania-stage-hint"))
    lighting_n_frames = find_animation_paths_with_default(folder, images.get("LightingN"), "lightingN")
    lighting_l_frames = find_animation_paths_with_default(folder, images.get("LightingL"), "lightingL")
    lighting_n_path = lighting_n_frames[0] if lighting_n_frames else find_skin_image("LightingN", images.get("LightingN"), "lightingN")
    lighting_l_path = lighting_l_frames[0] if lighting_l_frames else find_skin_image("LightingL", images.get("LightingL"), "lightingL")
    skin["hit_lighting_normal"] = read_image(lighting_n_path)
    skin["hit_lighting_long"] = read_image(lighting_l_path)
    skin["hit_lighting_normal_frames"] = [read_image(path) for path in lighting_n_frames] if lighting_n_frames else []
    skin["hit_lighting_long_frames"] = [read_image(path) for path in lighting_l_frames] if lighting_l_frames else []
    skin["hit_lighting_normal_density"] = 2.0 if lighting_n_path and "@2x" in lighting_n_path.stem.lower() else 1.0
    skin["hit_lighting_long_density"] = 2.0 if lighting_l_path and "@2x" in lighting_l_path.stem.lower() else 1.0
    ranking_panel_path = find_image(folder, "ranking-panel")
    skin["ranking_panel"] = read_image(ranking_panel_path)
    skin["ranking_panel_density"] = 2.0 if ranking_panel_path and "@2x" in ranking_panel_path.stem.lower() else 1.0

    for rank in ("X", "XH", "S", "SH", "A", "B", "C", "D"):
        path = find_image(folder, f"ranking-{rank.lower()}")
        skin["ranking_ranks"][rank] = read_image(path)
        skin["ranking_rank_densities"][rank] = 2.0 if path and "@2x" in path.stem.lower() else 1.0

    for element in ("accuracy", "maxcombo", "graph", "perfect", "title"):
        path = find_image(folder, f"ranking-{element}")
        skin["ranking_elements"][element] = read_image(path)
        skin["ranking_element_densities"][element] = 2.0 if path and "@2x" in path.stem.lower() else 1.0

    ranking_hit_names = {
        "300g": "mania-hit300g",
        "300": "mania-hit300",
        "200": "mania-hit200",
        "100": "mania-hit100",
        "50": "mania-hit50",
        "0": "mania-hit0",
    }

    for key, name in ranking_hit_names.items():
        variants = {}

        for density, suffix in ((1.0, ".png"), (2.0, "@2x.png")):
            animation_path = resolve_case_insensitive(folder / f"{name}-0{suffix}")
            static_path = resolve_case_insensitive(folder / f"{name}{suffix}")
            path = animation_path if animation_path.exists() else static_path

            if path.exists():
                variants[density] = read_image(path)

        skin["ranking_hit_images"][key] = variants

    mod_icon_names = {
        "DT": "doubletime",
        "NC": "nightcore",
        "HT": "halftime",
        "EZ": "easy",
        "HR": "hardrock",
        "HD": "hidden",
        "FI": "fadein",
        "FL": "flashlight",
        "NF": "nofail",
    }

    for acronym, name in mod_icon_names.items():
        path = find_image(folder, f"selection-mod-{name}")
        skin["mod_icons"][acronym] = read_image(path)
        skin["mod_icon_densities"][acronym] = 2.0 if path and "@2x" in path.stem.lower() else 1.0

    def load_font_glyphs(prefix, characters, fallback_prefixes=()):
        glyphs = {}
        suffix_names = {".": "dot", ",": "comma", "%": "percent"}
        prefix = (prefix or "score").strip().replace("\\", "/")
        prefixes = []
        for item in (prefix, *fallback_prefixes):
            item = (item or "").strip().replace("\\", "/")
            if item and item not in prefixes:
                prefixes.append(item)

        for character in characters:
            variants = {}
            suffix_name = suffix_names.get(character, character)

            for candidate_prefix in prefixes:
                if variants:
                    break

                for density, suffix in ((1.0, ".png"), (2.0, "@2x.png")):
                    path = resolve_case_insensitive(folder / f"{candidate_prefix}-{suffix_name}{suffix}")

                    # Punctuation is commonly kept in the skin root even when the
                    # digit prefix lives in a subdirectory.
                    if not path.exists() and (candidate_prefix.lower() == "score" or character in "x,.%"):
                        path = resolve_case_insensitive(folder / f"score-{suffix_name}{suffix}")

                    if path.exists():
                        variants[density] = read_image(path)

            glyphs[character] = variants
            if not variants:
                record_missing(f"FontGlyph:{prefix}:{character}", f"{prefix}-{suffix_name}", ",".join(fallback_prefixes))

        return glyphs

    skin["combo_glyphs"] = load_font_glyphs(cfg.get("combo_prefix", "score"), "0123456789x", ("combo", "score"))
    skin["score_glyphs"] = load_font_glyphs(cfg.get("score_prefix", "score"), "0123456789x,.%", ("score",))

    default_hit_names = {
        "0": "mania-hit0",
        "50": "mania-hit50",
        "100": "mania-hit100",
        "200": "mania-hit200",
        "300": "mania-hit300",
        "300g": "mania-hit300g",
    }

    for value in ("0", "50", "100", "200", "300", "300g"):
        image_name = images.get(f"Hit{value}") or default_hit_names[value]
        frame_paths = find_animation_paths(folder, image_name)
        if not frame_paths:
            frame_paths = find_animation_paths(folder, default_hit_names[value])
        image_path = find_image(folder, image_name)

        if image_path is None:
            image_path = frame_paths[0] if frame_paths else None

        if image_path is None:
            judge_name = "330" if value == "300g" else value
            image_path = find_image(folder, f"judge/{judge_name}")

        skin["hit_images"][value] = read_image(image_path)
        skin["hit_image_frames"][value] = [read_image(path) for path in frame_paths]
        skin["hit_image_densities"][value] = 2.0 if image_path and "@2x" in image_path.stem.lower() else 1.0

    for lane in range(keys):
        if skin["ln_bodies"][lane] is None:
            note_name = images.get(f"NoteImage{lane}L", "")
            fallback = note_name.replace("noteL", "NoteL").replace("notel", "NoteL")
            skin["ln_bodies"][lane] = read_image(find_image(folder, fallback))

        if skin["ln_tails"][lane] is None:
            note_name = images.get(f"NoteImage{lane}T", "")
            fallback = note_name.replace("noteT", "NoteT").replace("notet", "NoteT")
            fallback_img = read_image(find_image(folder, fallback))
            if fallback_img is not None:
                skin["ln_tails"][lane] = fallback_img
                skin["ln_tail_explicit"][lane] = True

    return skin
