import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from bisect import bisect_left, bisect_right
import os
import time
import json

from beatmap_parser import parse_osu
from osu_finder import get_mod_settings, get_stable_mania_accuracy, get_replay
from skin_loader import load_mania_skin
from replay_parser import get_replay_events
from osu_db_reader import read_mania_star_rating

CTX = {}
MANIA_MAX_TIME_RANGE_MS = 11485.0
MANIA_MIN_TIME_RANGE_MS = 290.0


def paste_rgba(frame, img, x, y, w=None, h=None):
    if img is None:
        return

    if w and h:
        img = cv2.resize(img, (max(1, w), max(1, h)), interpolation=cv2.INTER_AREA)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

    if img.shape[2] == 3:
        alpha_full = np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) * 255
        img = np.concatenate([img, alpha_full], axis=2)

    if x >= frame.shape[1] or y >= frame.shape[0] or x + img.shape[1] <= 0 or y + img.shape[0] <= 0:
        return

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + img.shape[1], frame.shape[1])
    y2 = min(y + img.shape[0], frame.shape[0])

    part = img[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha_channel = part[:, :, 3]

    if alpha_channel.min() == 255:
        frame[y1:y2, x1:x2] = part[:, :, :3]
        return

    alpha = part[:, :, 3:4].astype(np.float32) / 255.0
    dst = frame[y1:y2, x1:x2].astype(np.float32)
    src = part[:, :, :3].astype(np.float32)
    frame[y1:y2, x1:x2] = (alpha * src + (1 - alpha) * dst).astype(np.uint8)


def paste_rgba_centered(frame, img, cx, cy, scale=1.0, max_width=None):
    if img is None:
        return

    h, w = img.shape[:2]
    w = int(w * scale)
    h = int(h * scale)

    if max_width and w > max_width:
        ratio = max_width / w
        w = int(w * ratio)
        h = int(h * ratio)

    img = cv2.resize(img, (max(1, w), max(1, h)), interpolation=cv2.INTER_AREA)

    x = int(cx - w / 2)
    y = int(cy - h / 2)

    paste_rgba(frame, img, x, y)


def paste_rgba_bottom_centered(frame, img, cx, bottom_y, scale=1.0, max_width=None):
    if img is None:
        return

    h, w = img.shape[:2]
    w = int(w * scale)
    h = int(h * scale)

    if max_width and w > max_width:
        ratio = max_width / w
        w = int(w * ratio)
        h = int(h * ratio)

    img = cv2.resize(img, (max(1, w), max(1, h)), interpolation=cv2.INTER_AREA)

    x = int(cx - w / 2)
    y = int(bottom_y - h)

    paste_rgba(frame, img, x, y)


def alpha_bbox(img, threshold=8):
    if img is None or len(img.shape) < 3 or img.shape[2] < 4:
        return None

    ys, xs = np.where(img[:, :, 3] > threshold)

    if len(xs) == 0:
        return None

    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def has_visible_alpha(img):
    if img is None:
        return False

    if len(img.shape) < 3 or img.shape[2] < 4:
        return True

    return bool(np.any(img[:, :, 3] > 8))


def paste_rgba_centered_sized(frame, img, cx, cy, width, height, crop_alpha=False):
    if img is None:
        return

    if crop_alpha:
        bbox = alpha_bbox(img)

        if bbox:
            x1, y1, x2, y2 = bbox
            img = img[y1:y2, x1:x2]

    img = cv2.resize(img, (max(1, width), max(1, height)), interpolation=cv2.INTER_AREA)
    x = int(cx - width / 2)
    y = int(cy - height / 2)

    paste_rgba(frame, img, x, y)


def paste_ln_body(frame, img, cx, top_y, bottom_y, target_width):
    if img is None:
        return

    body_h = int(bottom_y - top_y)

    if body_h <= 0:
        return

    source = img
    bbox = alpha_bbox(source)

    if bbox:
        _, y1, _, _ = bbox
        source = source[y1:]

    source_h, source_w = source.shape[:2]
    scale = target_width / source_w
    source_needed = max(1, min(source_h, int(body_h / scale) + 2))
    source = source[:source_needed]

    body = cv2.resize(source, (max(1, target_width), max(1, body_h)), interpolation=cv2.INTER_AREA)
    paste_rgba(frame, body, int(cx - target_width / 2), int(top_y))


def scaled_size_for_width(img, target_width, scale=1.0):
    h, w = img.shape[:2]
    w = int(w * scale)
    h = int(h * scale)

    if target_width and w != target_width:
        ratio = target_width / w
        w = int(w * ratio)
        h = int(h * ratio)

    return max(1, w), max(1, h)


def skin_colour_to_bgra(values, default_alpha=255):
    if not values or len(values) < 3:
        return None

    alpha = values[3] if len(values) > 3 else default_alpha
    return values[2], values[1], values[0], alpha


def fill_rgba_rect(frame, x1, y1, x2, y2, colour):
    if colour is None:
        return

    b, g, r, a = colour

    if a <= 0:
        return

    x1 = max(0, min(frame.shape[1], int(x1)))
    x2 = max(0, min(frame.shape[1], int(x2)))
    y1 = max(0, min(frame.shape[0], int(y1)))
    y2 = max(0, min(frame.shape[0], int(y2)))

    if x2 <= x1 or y2 <= y1:
        return

    alpha = a / 255.0
    colour_arr = np.array([b, g, r], dtype=np.float32)
    frame[y1:y2, x1:x2] = alpha * colour_arr + (1 - alpha) * frame[y1:y2, x1:x2]


def mania_scroll_time_ms(scroll_speed_value):
    scroll_speed = max(1.0, min(40.0, float(scroll_speed_value)))
    return max(MANIA_MIN_TIME_RANGE_MS, MANIA_MAX_TIME_RANGE_MS / scroll_speed)


def apply_motion_blur(frame, x, y, w, h, strength):
    strength = int(strength)

    if strength <= 0:
        return

    kernel_size = max(3, strength * 2 + 1)
    kernel = np.zeros((kernel_size, 1), dtype=np.float32)
    kernel[:, 0] = 1.0 / kernel_size

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(frame.shape[1], int(x + w))
    y2 = min(frame.shape[0], int(y + h))

    if x2 <= x1 or y2 <= y1:
        return

    frame[y1:y2, x1:x2] = cv2.filter2D(frame[y1:y2, x1:x2], -1, kernel)


def mania_hit_windows(overall_difficulty, is_convert=False):
    if is_convert:
        return {
            "perfect": 16,
            "great": 34 if overall_difficulty > 4 else 47,
            "good": 67 if overall_difficulty > 4 else 77,
            "ok": 97,
            "meh": 121,
            "miss": 158,
        }

    return {
        "perfect": 16,
        "great": int(64 - 3 * overall_difficulty),
        "good": int(97 - 3 * overall_difficulty),
        "ok": int(127 - 3 * overall_difficulty),
        "meh": int(151 - 3 * overall_difficulty),
        "miss": int(188 - 3 * overall_difficulty),
    }


def mania_score_value(diff, windows):
    if diff <= windows["great"]:
        return 300
    if diff <= windows["good"]:
        return 200
    if diff <= windows["ok"]:
        return 100
    if diff <= windows["meh"]:
        return 50
    return 0


def hit_image_key(value, diff=None):
    if value == 300 and diff is not None and diff <= 16:
        return "300g"
    if value >= 300:
        return "300"
    return str(value)


def build_judgements(notes, events, keys, mirror, overall_difficulty=5.0, is_convert=False):
    windows = mania_hit_windows(overall_difficulty, is_convert)
    press_events = [e for e in events if e["pressed"]]
    release_events = [e for e in events if not e["pressed"]]

    used_press = set()
    used_release = set()

    objects = []

    for note_id, note in enumerate(notes):
        lane = keys - 1 - note["lane"] if mirror else note["lane"]
        is_ln = note["end_time"] is not None

        objects.append({
            "time": note["time"],
            "lane": lane,
            "kind": "tap" if not is_ln else "ln_head",
            "note_id": note_id,
            "end_time": note["end_time"],
        })

        if is_ln:
            objects.append({
                "time": note["end_time"],
                "lane": lane,
                "kind": "ln_tail",
                "note_id": note_id,
                "start_time": note["time"],
            })

    objects.sort(key=lambda x: x["time"])

    results = []
    debug_bad = []

    combo = 0
    score_sum = 0
    judged = 0
    lane_holding = [False] * keys
    ln_heads = {}

    for obj in objects:
        value = 0
        diff_used = None

        counts_accuracy = obj["kind"] != "ln_head"

        if obj["kind"] in ("tap", "ln_head"):
            best = None
            best_i = None

            for i, event in enumerate(press_events):
                if i in used_press or event["lane"] != obj["lane"]:
                    continue

                diff = abs(event["time"] - obj["time"])

                if diff <= windows["miss"] and (best is None or diff < best):
                    best = diff
                    best_i = i

            if best_i is not None:
                used_press.add(best_i)
                lane_holding[obj["lane"]] = True
                diff_used = best

                value = mania_score_value(best, windows)

                if obj["kind"] == "ln_head":
                    ln_heads[obj["note_id"]] = {
                        "diff": best,
                        "value": value,
                    }

        elif obj["kind"] == "ln_tail":
            best = None
            best_i = None

            for i, event in enumerate(release_events):
                if i in used_release or event["lane"] != obj["lane"]:
                    continue

                diff = abs(event["time"] - obj["time"])

                if diff <= windows["miss"] and (best is None or diff < best):
                    best = diff
                    best_i = i

            head = ln_heads.get(obj["note_id"])

            if best_i is not None and head is not None and head["value"] > 0:
                used_release.add(best_i)
                lane_holding[obj["lane"]] = False
                diff_used = best

                combined = head["diff"] + best

                if head["diff"] <= windows["perfect"] * 1.2 and combined <= windows["perfect"] * 2.4:
                    value = 300
                    diff_used = max(head["diff"], best)
                elif head["diff"] <= windows["great"] * 1.1 and combined <= windows["great"] * 2.2:
                    value = 300
                    diff_used = max(head["diff"], best, windows["perfect"] + 1)
                elif head["diff"] <= windows["good"] and combined <= windows["good"] * 2:
                    value = 200
                elif head["diff"] <= windows["ok"] and combined <= windows["ok"] * 2:
                    value = 100
                else:
                    value = 50

            lane_holding[obj["lane"]] = False

        if counts_accuracy:
            combo = combo + 1 if value > 0 else 0
            judged += 1
            score_sum += value

        accuracy = (score_sum / (judged * 300)) * 100 if judged else 100.0

        result = {
            "time": obj["time"],
            "lane": obj["lane"],
            "kind": obj["kind"],
            "combo": combo,
            "accuracy": accuracy,
            "value": value,
            "diff": diff_used,
            "image_key": hit_image_key(value, diff_used),
            "display_time": obj["time"] if diff_used is None else obj["time"] + (diff_used if value > 0 else 0),
            "counts_accuracy": counts_accuracy,
        }

        results.append(result)

        if counts_accuracy and value < 300 and len(debug_bad) < 100:
            debug_bad.append(result)

    return results, debug_bad


def find_best_offset(notes, events, keys, mirror, overall_difficulty=5.0, is_convert=False):
    best_offset = 0
    best_score = None
    best_judgements = []
    best_bad = []

    for offset in range(-100, 101):
        shifted_events = [
            {
                "time": e["time"] + offset,
                "lane": e["lane"],
                "pressed": e["pressed"],
            }
            for e in events
        ]

        test_judgements, test_bad = build_judgements(notes, shifted_events, keys, mirror, overall_difficulty, is_convert)
        test_accuracy = test_judgements[-1]["accuracy"] if test_judgements else 100.0
        misses = sum(1 for j in test_judgements if j["value"] == 0)
        timing_error = sum((j["diff"] or 180) for j in test_judgements)
        score = (-test_accuracy, misses, timing_error)

        if best_score is None or score < best_score:
            best_score = score
            best_offset = offset
            best_judgements = test_judgements
            best_bad = test_bad

    shifted_events = [
        {
            "time": e["time"] + best_offset,
            "lane": e["lane"],
            "pressed": e["pressed"],
        }
        for e in events
    ]

    return best_offset, shifted_events, best_judgements, best_bad


def init_worker(ctx):
    global CTX
    CTX = ctx


def draw_receptors(frame, skin, pressed, keys, lane_xs, column_widths, judge_y, note_widths):
    for lane in range(keys):
        lane_x = lane_xs[lane]
        lane_width = column_widths[lane]
        center_x = lane_x + lane_width // 2
        receptor_size = note_widths[lane]
        receptor_y = judge_y - receptor_size // 2

        img_list = skin["keys_down"] if pressed[lane] else skin["keys"]
        receptor_img = img_list[lane] if lane < len(img_list) else None

        if receptor_img is not None:
            paste_rgba_centered_sized(
                frame,
                receptor_img,
                center_x,
                receptor_y,
                receptor_size,
                receptor_size,
                crop_alpha=True,
            )
        else:
            receptor_w = receptor_size
            receptor_h = receptor_size
            receptor_x = center_x - receptor_w // 2
            color = (180, 180, 210) if pressed[lane] else (90, 90, 100)
            cv2.ellipse(
                frame,
                (center_x, receptor_y),
                (receptor_w // 2, receptor_h // 2),
                0,
                0,
                360,
                color,
                3,
            )


def draw_stage_lights(frame, skin, pressed, lane_xs, column_widths, height, cfg, skin_scale):
    stage_light = skin.get("stage_light")

    if stage_light is None or not has_visible_alpha(stage_light):
        return

    light_y = int((cfg.get("light_position") or cfg.get("hit_position") or 480) * skin_scale)

    for lane, is_pressed in enumerate(pressed):
        if not is_pressed or lane >= len(lane_xs):
            continue

        center_x = lane_xs[lane] + column_widths[lane] // 2
        light_w = column_widths[lane]
        light_h = min(height, int(stage_light.shape[0] * skin_scale))
        top = max(0, light_y - light_h)
        paste_rgba(frame, stage_light, int(center_x - light_w / 2), top, light_w, light_h)


def draw_stage_bottom(frame, skin, play_x, play_width, height, skin_scale):
    stage_bottom = skin.get("stage_bottom")

    if stage_bottom is None or not has_visible_alpha(stage_bottom):
        return

    cover_h = min(height, int(stage_bottom.shape[0] * skin_scale))
    bbox = alpha_bbox(stage_bottom)
    y = 0 if bbox and bbox[3] <= stage_bottom.shape[0] / 2 else height - cover_h
    paste_rgba(frame, stage_bottom, play_x, y, play_width, cover_h)


def draw_hit_judgements(frame, skin, judgements, map_time, lane_xs, column_widths, judge_y, note_widths):
    if not lane_xs:
        return

    play_center_x = (lane_xs[0] + lane_xs[-1] + column_widths[-1]) // 2
    cfg = skin.get("cfg", {})
    skin_scale = frame.shape[0] / 480
    judgement_y = int((cfg.get("combo_position") or 130) * skin_scale)

    for judgement in judgements:
        display_time = judgement.get("display_time", judgement["time"])
        age = map_time - display_time

        if age < 0 or age > 450:
            continue

        value = judgement["value"]
        key = judgement.get("image_key") or hit_image_key(value, judgement.get("diff"))
        img = skin.get("hit_images", {}).get(key)

        if img is None:
            continue

        alpha_scale = max(0.0, 1.0 - age / 450)
        draw_img = img.copy()

        if len(draw_img.shape) == 3 and draw_img.shape[2] == 4:
            draw_img[:, :, 3] = (draw_img[:, :, 3].astype(np.float32) * alpha_scale).astype(np.uint8)

        y = judgement_y - int(age * 0.04)
        paste_rgba_centered(frame, draw_img, play_center_x, y, scale=1.0, max_width=180)


def draw_judgement_counter(frame, judgements, map_time, width):
    counts = {"300g": 0, "300": 0, "200": 0, "100": 0, "50": 0, "0": 0}

    for judgement in judgements:
        if not judgement.get("counts_accuracy", True):
            continue

        display_time = judgement.get("display_time", judgement["time"])

        if display_time > map_time:
            break

        key = judgement.get("image_key") or hit_image_key(judgement["value"], judgement.get("diff"))
        counts[key] = counts.get(key, 0) + 1

    labels = [
        ("300g", counts["300g"], (120, 235, 255)),
        ("300", counts["300"], (220, 240, 255)),
        ("200", counts["200"], (100, 220, 120)),
        ("100", counts["100"], (240, 220, 80)),
        ("50", counts["50"], (240, 150, 70)),
        ("Miss", counts["0"], (230, 80, 80)),
    ]

    x = width - 175
    y = 105

    for label, count, color in labels:
        cv2.putText(frame, f"{label}: {count}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
        y += 24


def mania_pp_value(star_rating, counts):
    if star_rating is None:
        return None

    perfect = counts.get("300g", 0)
    great = counts.get("300", 0)
    good = counts.get("200", 0)
    ok = counts.get("100", 0)
    meh = counts.get("50", 0)
    miss = counts.get("0", 0)
    total_hits = perfect + great + good + ok + meh + miss

    if total_hits <= 0:
        return 0.0

    score_accuracy = ((perfect * 320) + (great * 300) + (good * 200) + (ok * 100) + (meh * 50)) / (total_hits * 320)
    difficulty = (
        8.0
        * pow(max(star_rating - 0.15, 0.05), 2.2)
        * max(0, 5 * score_accuracy - 4)
        * (1 + 0.1 * min(1, total_hits / 1500))
    )

    return difficulty


def judgement_counts_at(judgements, map_time):
    counts = {"300g": 0, "300": 0, "200": 0, "100": 0, "50": 0, "0": 0}

    for judgement in judgements:
        if not judgement.get("counts_accuracy", True):
            continue

        display_time = judgement.get("display_time", judgement["time"])

        if display_time > map_time:
            break

        key = judgement.get("image_key") or hit_image_key(judgement["value"], judgement.get("diff"))
        counts[key] = counts.get(key, 0) + 1

    return counts


def draw_pp_counter(frame, judgements, map_time, width, star_rating):
    counts = judgement_counts_at(judgements, map_time)
    pp = mania_pp_value(star_rating, counts)
    text = "pp: N/A" if pp is None else f"pp: {pp:.2f}"
    cv2.putText(frame, text, (width - 185, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)


def draw_star_rating(frame, star_rating, height):
    text = "SR: N/A" if star_rating is None else f"SR: {star_rating:.2f}*"
    cv2.putText(frame, text, (24, height - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2)


def draw_timeline(frame, map_time, start_map_time, end_map_time, width, height):
    x1 = 24
    x2 = width - 24
    y = height - 24
    progress = (map_time - start_map_time) / max(1, end_map_time - start_map_time)
    progress = max(0.0, min(1.0, progress))
    current_x = int(x1 + (x2 - x1) * progress)
    cv2.line(frame, (x1, y), (x2, y), (80, 80, 86), 4)
    cv2.line(frame, (x1, y), (current_x, y), (90, 200, 230), 4)
    cv2.circle(frame, (current_x, y), 6, (230, 230, 230), -1)


def latest_judgement(judgements, lane, kind, time):
    for judgement in judgements:
        if judgement["lane"] == lane and judgement["kind"] == kind and judgement["time"] == time:
            return judgement

    return None


def frame_worker(frame_id):
    width = CTX["width"]
    height = CTX["height"]
    fps = CTX["fps"]
    start_map_time = CTX["start_map_time"]
    end_map_time = CTX["end_map_time"]
    notes = CTX["notes"]
    keys = CTX["keys"]
    title = CTX["title"]
    output_dir = CTX["output_dir"]
    scroll_time_ms = CTX["scroll_time_ms"]
    motion_blur = CTX.get("motion_blur", 0)
    speed_multiplier = CTX["speed_multiplier"]
    mirror = CTX["mirror"]
    skin = CTX["skin"]
    judgements = CTX["judgements"]
    events = CTX["events"]
    event_lanes = CTX.get("event_lanes", [])
    note_times = CTX.get("note_times", [])
    judgement_lookup = CTX.get("judgement_lookup", {})
    star_rating = CTX.get("star_rating")
    replay_accuracy = CTX.get("replay_accuracy", 100.0)

    real_elapsed = int(frame_id * 1000 / fps)
    map_time = start_map_time + int(real_elapsed * speed_multiplier)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (12, 12, 15)

    cfg = skin["cfg"]

    skin_scale = height / 480
    scale_x = skin_scale
    scale_y = skin_scale

    column_widths = cfg["column_widths"] or [70] * keys
    column_spacing = cfg["column_spacing"] or [0] * (keys - 1)

    column_widths = [int(w * scale_x) for w in column_widths]
    column_spacing = [int(s * scale_x) for s in column_spacing]

    play_width = sum(column_widths) + sum(column_spacing)

    if cfg["column_start"] is not None:
        play_x = int((width - play_width) / 2)
    else:
        play_x = (width - play_width) // 2

    if cfg["hit_position"] is not None:
        judge_y = int(cfg["hit_position"] * scale_y)
    else:
        judge_y = height - 120

    top_y = 0

    lane_xs = []
    cur = play_x

    for i in range(keys):
        lane_xs.append(cur)
        cur += column_widths[i]

        if i < len(column_spacing):
            cur += column_spacing[i]

    note_widths = [int(w * 0.94) for w in column_widths]

    if skin.get("stage_left") is not None:
        stage_img = skin["stage_left"]
        paste_rgba(frame, stage_img, 0, height - stage_img.shape[0])

    if skin.get("stage_right") is not None:
        stage_img = skin["stage_right"]
        paste_rgba(frame, stage_img, width - stage_img.shape[1], height - stage_img.shape[0])

    cv2.rectangle(frame, (play_x - 16, top_y), (play_x + play_width + 16, height), (0, 0, 0), -1)

    for lane in range(keys):
        colour = skin_colour_to_bgra(cfg["colours"].get(f"Colour{lane + 1}"))
        fill_rgba_rect(frame, lane_xs[lane], top_y, lane_xs[lane] + column_widths[lane], height, colour)

    cv2.putText(frame, title[:95], (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)

    combo = 0
    accuracy = 100.0

    for j in judgements:
        if not j.get("counts_accuracy", True):
            continue

        display_time = j.get("display_time", j["time"])

        if display_time <= map_time:
            combo = j["combo"]
            accuracy = j["accuracy"]
        else:
            break

    visible_margin = 500
    visible_start = map_time - visible_margin
    effective_scroll_time_ms = scroll_time_ms * speed_multiplier
    visible_end = map_time + effective_scroll_time_ms + visible_margin

    pressed = [False] * keys

    if event_lanes:
        for lane in range(keys):
            lane_events = event_lanes[lane] if lane < len(event_lanes) else ([], [])
            lane_times, lane_states = lane_events
            event_i = bisect_right(lane_times, map_time) - 1
            pressed[lane] = bool(lane_states[event_i]) if event_i >= 0 else False
    else:
        for event in events:
            if event["time"] <= map_time:
                lane = keys - 1 - event["lane"] if mirror else event["lane"]

                if 0 <= lane < keys:
                    pressed[lane] = event["pressed"]
            else:
                break

    if cfg["keys_under_notes"]:
        draw_receptors(frame, skin, pressed, keys, lane_xs, column_widths, judge_y, note_widths)

    draw_stage_lights(frame, skin, pressed, lane_xs, column_widths, height, cfg, skin_scale)

    start_i = max(0, bisect_left(note_times, visible_start) - 128) if note_times else 0

    for note in notes[start_i:]:
        note_time = note["time"]
        end_time = note["end_time"] if note["end_time"] is not None else note_time

        if end_time < visible_start:
            continue

        if note_time > visible_end:
            break

        lane = keys - 1 - note["lane"] if mirror else note["lane"]

        if lane < 0 or lane >= keys:
            continue

        tap_judgement = judgement_lookup.get((lane, "tap", note_time))
        head_judgement = judgement_lookup.get((lane, "ln_head", note_time))
        tail_judgement = judgement_lookup.get((lane, "ln_tail", end_time))

        if tap_judgement and map_time >= tap_judgement.get("display_time", tap_judgement["time"]):
            continue

        lane_x = lane_xs[lane]
        lane_width = column_widths[lane]
        center_x = lane_x + lane_width // 2

        note_w = note_widths[lane]
        note_h = max(12, int(lane_width * 0.24))
        note_x = center_x - note_w // 2
        hit_y = judge_y - note_w // 2
        lane_scroll_speed = max(0.01, (hit_y - top_y) / effective_scroll_time_ms)

        y_head = int(hit_y - (note_time - map_time) * lane_scroll_speed)
        y_tail = int(hit_y - (end_time - map_time) * lane_scroll_speed)

        if note["end_time"] is not None:
            visible_top = min(y_head, y_tail)
            visible_bottom = max(y_head, y_tail)

            if map_time >= note_time:
                visible_bottom = hit_y

            if visible_bottom >= top_y and visible_top <= height:
                body_img = skin["ln_bodies"][lane]
                head_img = skin["ln_heads"][lane]
                tail_img = skin["ln_tails"][lane]

                if body_img is not None:
                    body_h = max(1, visible_bottom - visible_top)
                    paste_ln_body(frame, body_img, center_x, visible_top, visible_top + body_h, note_w)
                else:
                    cv2.rectangle(
                        frame,
                        (note_x, visible_top),
                        (note_x + note_w, visible_bottom),
                        (150, 150, 150),
                        -1,
                    )

                if head_img is not None:
                    tail_display_time = tail_judgement.get("display_time", tail_judgement["time"]) if tail_judgement else end_time
                    head_display_time = head_judgement.get("display_time", head_judgement["time"]) if head_judgement else note_time
                    head_hit = head_judgement is not None and head_judgement["value"] > 0

                    if map_time < head_display_time:
                        paste_rgba_centered(frame, head_img, center_x, y_head, scale=skin_scale, max_width=note_w)
                    elif head_hit and map_time < tail_display_time:
                        paste_rgba_centered(frame, head_img, center_x, hit_y, scale=skin_scale, max_width=note_w)

                if (
                    tail_img is not None
                    and y_tail >= top_y
                    and y_tail <= height
                    and (tail_judgement is None or map_time < tail_judgement.get("display_time", tail_judgement["time"]))
                ):
                    if has_visible_alpha(tail_img):
                        paste_rgba_centered(frame, tail_img, center_x, y_tail, scale=skin_scale, max_width=note_w)

        else:
            note_img = skin["notes"][lane]

            if note_img is not None:
                paste_rgba_centered(frame, note_img, center_x, y_head, scale=skin_scale, max_width=note_w)
            else:
                cv2.rectangle(
                    frame,
                    (note_x, y_head - note_h // 2),
                    (note_x + note_w, y_head + note_h // 2),
                    (90, 160, 220),
                    -1,
                )

    if motion_blur > 0:
        apply_motion_blur(frame, play_x - 16, top_y, play_width + 32, height, motion_blur)

    draw_stage_bottom(frame, skin, play_x, play_width, height, skin_scale)

    line_colour = skin_colour_to_bgra(cfg["colours"].get("ColourColumnLine")) or (65, 55, 55, 255)

    for i in range(keys + 1):
        x = play_x + sum(column_widths[:i]) + sum(column_spacing[:max(0, i - 1)])
        line_width = cfg["column_line_widths"][i] if i < len(cfg["column_line_widths"]) else 1

        if line_width > 0 and line_colour[3] > 0:
            cv2.line(frame, (x, top_y), (x, height), line_colour[:3], max(1, line_width))

    if cfg["judgement_line"]:
        cv2.line(frame, (play_x, judge_y), (play_x + play_width, judge_y), (225, 225, 225), 2)

    if not cfg["keys_under_notes"]:
        draw_receptors(frame, skin, pressed, keys, lane_xs, column_widths, judge_y, note_widths)

    draw_hit_judgements(frame, skin, judgements, map_time, lane_xs, column_widths, judge_y, note_widths)
    draw_judgement_counter(frame, judgements, map_time, width)
    draw_pp_counter(frame, judgements, map_time, width, star_rating)
    draw_star_rating(frame, star_rating, height)
    draw_timeline(frame, map_time, start_map_time, end_map_time, width, height)

    combo_y = int((cfg.get("combo_position") or 58) * skin_scale)
    score_y = int((cfg.get("score_position") or 98) * skin_scale)

    cv2.putText(frame, f"{combo}x", (width - 185, combo_y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (240, 240, 240), 2)
    cv2.putText(frame, f"{accuracy:.2f}%", (width - 190, score_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

    out = Path(output_dir) / f"frame_{frame_id:07d}.png"
    cv2.imwrite(str(out), frame)

    return frame_id


def make_audio_args(speed_multiplier, nightcore_pitch):
    if speed_multiplier == 1.0:
        return []

    if nightcore_pitch and speed_multiplier == 1.5:
        return ["-filter:a", "asetrate=44100*1.5,aresample=44100"]

    if speed_multiplier == 1.5:
        return ["-filter:a", "atempo=1.5"]

    if speed_multiplier == 0.75:
        return ["-filter:a", "atempo=0.75"]

    return ["-filter:a", f"atempo={speed_multiplier}"]


def ffmpeg_encoder_names():
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except Exception:
        return ""


def vaapi_device():
    for device in ("/dev/dri/renderD128", "/dev/dri/renderD129"):
        if Path(device).exists():
            return device

    return None


def video_encode_commands(fps, frames_dir, silent_video):
    encoders = ffmpeg_encoder_names()
    input_args = [
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%07d.png"),
    ]

    device = vaapi_device()

    if device and "h264_vaapi" in encoders:
        yield [
            "ffmpeg", "-y",
            "-vaapi_device", device,
            *input_args,
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi",
            "-qp", "18",
            str(silent_video),
        ]

    if "h264_qsv" in encoders:
        yield [
            "ffmpeg", "-y",
            *input_args,
            "-vf", "format=nv12",
            "-c:v", "h264_qsv",
            "-global_quality", "18",
            str(silent_video),
        ]

    if "h264_amf" in encoders:
        yield [
            "ffmpeg", "-y",
            *input_args,
            "-c:v", "h264_amf",
            "-quality", "quality",
            "-qp_i", "18",
            "-qp_p", "18",
            str(silent_video),
        ]

    yield [
        "ffmpeg", "-y",
        *input_args,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "18",
        str(silent_video),
    ]


def encode_silent_video(fps, frames_dir, silent_video):
    last_error = None

    for cmd in video_encode_commands(fps, frames_dir, silent_video):
        try:
            subprocess.run(cmd, check=True)
            return cmd
        except subprocess.CalledProcessError as e:
            last_error = e

    if last_error:
        raise last_error


def write_debug_report(
    output_file,
    replay_file,
    replay_accuracy,
    simulated_accuracy,
    bad_judgements,
    notes,
    events,
    mod_settings,
    auto_offset_ms,
    overall_difficulty,
    is_convert,
    scroll_speed_value,
    scroll_time_ms,
    motion_blur,
    star_rating,
    video_encoder_cmd=None,
):
    replay = get_replay(replay_file) if replay_file else None

    data = {
        "replay_file": replay_file,
        "output_file": output_file,
        "mods": mod_settings,
        "auto_offset_ms": auto_offset_ms,
        "replay_accuracy": replay_accuracy,
        "simulated_accuracy": simulated_accuracy,
        "accuracy_difference": simulated_accuracy - replay_accuracy,
        "note_count": len(notes),
        "event_count": len(events),
        "overall_difficulty": overall_difficulty,
        "is_convert": is_convert,
        "hit_windows": mania_hit_windows(overall_difficulty, is_convert),
        "scroll_speed": scroll_speed_value,
        "scroll_time_ms": scroll_time_ms,
        "motion_blur": motion_blur,
        "star_rating": star_rating,
        "video_encoder": video_encoder_cmd,
        "first_bad_judgements": bad_judgements[:100],
    }

    if replay:
        data["replay_counts"] = {
            "count_300": getattr(replay, "count_300", 0),
            "count_100": getattr(replay, "count_100", 0),
            "count_50": getattr(replay, "count_50", 0),
            "count_miss": getattr(replay, "count_miss", 0),
            "count_geki": getattr(replay, "count_geki", 0),
            "count_katu": getattr(replay, "count_katu", 0),
            "max_combo": getattr(replay, "max_combo", 0),
            "score": getattr(replay, "score", 0),
        }

    debug_path = Path(output_file).with_suffix(".debug.json")

    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {minutes}m"

    if minutes:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"


def render_video(osu_file, skin_folder, output_file, replay_file, scroll_speed_value, resolution, motion_blur=0, progress_callback=None):
    if progress_callback:
        progress_callback(0, "Preparing render: reading beatmap and replay...")

    beatmap = parse_osu(osu_file)

    width, height = map(int, resolution.split("x"))
    fps = 60

    mod_settings = get_mod_settings(replay_file) if replay_file else {
        "speed_multiplier": 1.0,
        "mirror": False,
        "mods": "NM",
        "nightcore_pitch": False,
    }

    speed_multiplier = mod_settings["speed_multiplier"]
    mirror = mod_settings["mirror"]
    nightcore_pitch = mod_settings["nightcore_pitch"]
    overall_difficulty = beatmap.overall_difficulty
    is_convert = beatmap.mode != 3

    notes = [{"lane": n.lane, "time": n.time, "end_time": n.end_time} for n in beatmap.notes]
    star_rating = read_mania_star_rating(osu_file, beatmap.md5_hash, mod_settings["mods_int"]) if replay_file else None

    if progress_callback:
        progress_callback(0, "Preparing render: analysing replay timing...")

    events = get_replay_events(replay_file, beatmap.keys) if replay_file else []
    replay_accuracy = get_stable_mania_accuracy(replay_file) if replay_file else 100.0

    if events:
        judgements, bad_judgements = build_judgements(notes, events, beatmap.keys, mirror, overall_difficulty, is_convert)
        simulated_accuracy = judgements[-1]["accuracy"] if judgements else 100.0

        if abs(simulated_accuracy - replay_accuracy) < 0.01 and not bad_judgements:
            best_offset = 0
        else:
            best_offset, events, judgements, bad_judgements = find_best_offset(
                notes,
                events,
                beatmap.keys,
                mirror,
                overall_difficulty,
                is_convert,
            )
    else:
        best_offset, judgements, bad_judgements = 0, [], []
        simulated_accuracy = 100.0

    simulated_accuracy = judgements[-1]["accuracy"] if judgements else 100.0

    start_map_time = max(0, notes[0]["time"] - 2000)
    end_map_time = notes[-1]["time"] + 3000

    real_duration_ms = int((end_map_time - start_map_time) / speed_multiplier)
    total_frames = max(1, int(real_duration_ms / 1000 * fps))
    scroll_time_ms = mania_scroll_time_ms(scroll_speed_value)

    workers = max(1, min(os.cpu_count() or 1, 8))
    megapixels = (width * height) / 1_000_000
    estimated_render_fps = max(4.0, workers * (9.0 / max(1.0, megapixels)))
    estimated_seconds = total_frames / estimated_render_fps

    if progress_callback:
        progress_callback(
            0,
            f"Preparing render: {total_frames} frames | Estimated render time: ~{format_duration(estimated_seconds)}"
        )

    temp_dir = Path(tempfile.mkdtemp(prefix=".mania-render-", dir=str(Path(output_file).parent)))
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    skin = load_mania_skin(skin_folder, beatmap.keys)

    event_lanes = []

    for lane in range(beatmap.keys):
        lane_events = []

        for event in events:
            event_lane = beatmap.keys - 1 - event["lane"] if mirror else event["lane"]

            if event_lane == lane:
                lane_events.append((event["time"], event["pressed"]))

        event_lanes.append((
            [event_time for event_time, _ in lane_events],
            [pressed for _, pressed in lane_events],
        ))

    judgement_lookup = {
        (j["lane"], j["kind"], j["time"]): j
        for j in judgements
    }

    ctx = {
        "width": width,
        "height": height,
        "fps": fps,
        "start_map_time": start_map_time,
        "end_map_time": end_map_time,
        "notes": notes,
        "keys": beatmap.keys,
        "title": f"{beatmap.artist} - {beatmap.title} [{beatmap.version}] | {mod_settings['mods']}",
        "output_dir": str(frames_dir),
        "scroll_time_ms": scroll_time_ms,
        "motion_blur": int(motion_blur),
        "speed_multiplier": speed_multiplier,
        "mirror": mirror,
        "skin": skin,
        "judgements": judgements,
        "events": events,
        "event_lanes": event_lanes,
        "note_times": [note["time"] for note in notes],
        "judgement_lookup": judgement_lookup,
        "star_rating": star_rating,
        "replay_accuracy": replay_accuracy,
    }

    write_debug_report(
        output_file=output_file,
        replay_file=replay_file,
        replay_accuracy=replay_accuracy,
        simulated_accuracy=simulated_accuracy,
        bad_judgements=bad_judgements,
        notes=notes,
        events=events,
        mod_settings=mod_settings,
        auto_offset_ms=best_offset,
        overall_difficulty=overall_difficulty,
        is_convert=is_convert,
        scroll_speed_value=scroll_speed_value,
        scroll_time_ms=scroll_time_ms,
        motion_blur=motion_blur,
        star_rating=star_rating,
    )

    start = time.time()
    done = 0

    if progress_callback:
        progress_callback(0, f"Frame: 0/{total_frames} | ETA: calculating...")

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(ctx,)) as executor:
        next_frame = 0
        futures = set()
        max_pending = workers * 3

        while next_frame < total_frames and len(futures) < max_pending:
            futures.add(executor.submit(frame_worker, next_frame))
            next_frame += 1

        while futures:
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)

            for future in completed:
                future.result()
                done += 1

                if next_frame < total_frames:
                    futures.add(executor.submit(frame_worker, next_frame))
                    next_frame += 1

                elapsed = time.time() - start
                render_fps = done / elapsed if elapsed > 0 else 0
                remaining = int((total_frames - done) / render_fps) if render_fps > 0 else 0

                if progress_callback:
                    progress_callback(
                        int(done / total_frames * 85),
                        f"Frame: {done}/{total_frames} | Render: {render_fps:.1f} fps | ETA: {remaining}s"
                    )

    silent_video = temp_dir / "silent.mp4"

    encoder_cmd = encode_silent_video(fps, frames_dir, silent_video)
    debug_path = Path(output_file).with_suffix(".debug.json")

    if debug_path.exists():
        try:
            with open(debug_path, "r", encoding="utf-8") as f:
                debug_data = json.load(f)

            debug_data["video_encoder"] = encoder_cmd

            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

    if progress_callback:
        progress_callback(90, "Encoding audio/video...")

    audio_path = Path(osu_file).parent / beatmap.audio_file

    if audio_path.exists():
        cmd = [
            "ffmpeg", "-y",
            "-i", str(silent_video),
            "-ss", str(start_map_time / 1000),
            "-i", str(audio_path),
        ]

        cmd += make_audio_args(speed_multiplier, nightcore_pitch)

        cmd += [
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_file,
        ]
    else:
        cmd = ["ffmpeg", "-y", "-i", str(silent_video), "-c:v", "copy", output_file]

    subprocess.run(cmd, check=True)
    shutil.rmtree(temp_dir)

    if progress_callback:
        progress_callback(100, "Done")
