import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from bisect import bisect_left, bisect_right
import os
import time
import json

from osu_mania_replay_renderer.beatmap_parser import parse_osu
from osu_mania_replay_renderer.osu_finder import get_mod_settings, get_stable_mania_accuracy, get_replay, mod_settings_from_replay
from osu_mania_replay_renderer.skin_loader import load_mania_skin
from osu_mania_replay_renderer.replay_parser import get_replay_events
from osu_mania_replay_renderer.osu_db_reader import read_mania_star_rating
from osu_mania_replay_renderer.renderer_media import (
    encode_silent_video,
    ffmpeg_binary,
    ffmpeg_encoder_names,
    format_duration,
    frame_input_args,
    make_audio_args,
    vaapi_device,
    video_encode_commands,
)
from osu_mania_replay_renderer.scoring import (
    build_event_lanes,
    build_judgements,
    find_best_offset,
    add_stable_ln_ticks,
    hit_image_key,
    judgement_counts,
    mania_accuracy_from_counts,
    mania_hit_windows,
    reconcile_judgements_with_replay,
    replay_judgement_counts,
    stable_mania_accuracy_from_replay,
)

CTX = {}
RESIZE_CACHE = {}
ALPHA_BBOX_CACHE = {}
GPU_COMPOSITOR = None
GPU_COMPOSITING_ACTIVE = True
FRAME_STREAM = None
PREVIOUS_RENDER_FRAME = None
PREVIOUS_RENDER_FRAME_ID = None
MANIA_MAX_TIME_RANGE_MS = 11485.0
MANIA_MIN_TIME_RANGE_MS = 290.0


class RenderCancelled(RuntimeError):
    pass


def ensure_not_cancelled(cancel_callback):
    if cancel_callback and cancel_callback():
        raise RenderCancelled("Render cancelled")


def terminate_process_pool(executor, futures):
    for future in futures:
        future.cancel()

    processes = list((getattr(executor, "_processes", None) or {}).values())

    for process in processes:
        if process.is_alive():
            process.terminate()

    executor.shutdown(wait=True, cancel_futures=True)


def draw_ui_text(frame, text, origin, scale=0.6, color=(235, 235, 235), thickness=1, anchor="left"):
    """Draw readable antialiased overlay text with a subtle dark edge."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin

    if anchor == "right":
        x -= text_w
    elif anchor == "center":
        x -= text_w // 2

    cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 2, line_type)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, line_type)


def draw_song_header(frame, title, mapper, player, bar_height=None):
    """Draw the compact black metadata strip used by legacy replay playback."""
    height, width = frame.shape[:2]
    ui_scale = max(0.75, height / 1080.0)
    bar_height = int(bar_height) if bar_height is not None else max(42, int(58 * ui_scale))
    overlay = frame[:bar_height].copy()
    overlay[:] = (0, 0, 0)
    cv2.addWeighted(overlay, 0.78, frame[:bar_height], 0.22, 0, frame[:bar_height])

    title_scale = max(0.55, bar_height / 68.0)
    max_title_width = width - int(24 * ui_scale)
    while title_scale > 0.38:
        (title_width, title_height), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 1)
        if title_width <= max_title_width and title_height <= bar_height * 0.48:
            break
        title_scale -= 0.04

    title_y = max(int(22 * ui_scale), int(bar_height * 0.43))
    draw_ui_text(frame, title, (int(12 * ui_scale), title_y), title_scale, (246, 246, 248), 1)

    beatmap_y = bar_height - int(39 * ui_scale)
    player_y = bar_height - int(12 * ui_scale)
    draw_ui_text(
        frame,
        f"Beatmap by {mapper}",
        (int(12 * ui_scale), beatmap_y),
        0.70 * ui_scale,
        (240, 242, 246),
        1,
    )
    draw_ui_text(
        frame,
        f"Played by {player}",
        (int(12 * ui_scale), player_y),
        0.60 * ui_scale,
        (215, 218, 224),
        1,
    )
    cv2.line(
        frame,
        (0, bar_height - 1),
        (width, bar_height - 1),
        (112, 118, 132),
        max(1, int(ui_scale)),
        cv2.LINE_AA,
    )


def format_clock(milliseconds):
    seconds = max(0, int(milliseconds // 1000))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}:{seconds:02d}"


def overlay_scale(height):
    return max(0.58, min(1.8, height / 900))


def side_stats_dimensions(height):
    ui_scale = overlay_scale(height)
    return max(96, int(186 * ui_scale)), max(140, int(259 * ui_scale))


def layout_point(layout_positions, key, width, height):
    value = (layout_positions or {}).get(key)

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None

    try:
        x = max(0.0, min(1.0, float(value[0])))
        y = max(0.0, min(1.0, float(value[1])))
    except (TypeError, ValueError):
        return None

    return int(x * width), int(y * height)


def resize_texture(img, width, height, interpolation=cv2.INTER_AREA):
    width = max(1, int(width))
    height = max(1, int(height))

    if img.shape[1] == width and img.shape[0] == height:
        return img

    if not img.flags.writeable:
        pointer = int(img.__array_interface__["data"][0])
        key = (pointer, img.shape, img.strides, width, height, interpolation)
        cached = RESIZE_CACHE.get(key)

        if cached is not None:
            return cached

        resized = cv2.resize(img, (width, height), interpolation=interpolation)
        resized.flags.writeable = False
        RESIZE_CACHE[key] = resized
        return resized

    return cv2.resize(img, (width, height), interpolation=interpolation)


def paste_rgba(frame, img, x, y, w=None, h=None):
    if img is None:
        return

    if GPU_COMPOSITOR is not None and GPU_COMPOSITING_ACTIVE:
        target_width = w if w is not None else img.shape[1]
        target_height = h if h is not None else img.shape[0]
        GPU_COMPOSITOR.queue(img, x, y, target_width, target_height)
        return

    if w and h:
        img = resize_texture(img, w, h)

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


def paste_additive(frame, img, x, y, w=None, h=None):
    if img is None:
        return

    target_width = w if w is not None else img.shape[1]
    target_height = h if h is not None else img.shape[0]

    if GPU_COMPOSITOR is not None and GPU_COMPOSITING_ACTIVE:
        GPU_COMPOSITOR.queue(img, x, y, target_width, target_height, "additive")
        return

    if w is not None and h is not None:
        img = resize_texture(img, target_width, target_height)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        alpha_full = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha_full], axis=2)

    if x >= frame.shape[1] or y >= frame.shape[0] or x + img.shape[1] <= 0 or y + img.shape[0] <= 0:
        return

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + img.shape[1], frame.shape[1])
    y2 = min(y + img.shape[0], frame.shape[0])
    part = img[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha = part[:, :, 3:4].astype(np.float32) / 255.0
    glow = part[:, :, :3].astype(np.float32) * alpha
    destination = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = np.clip(destination + glow, 0, 255).astype(np.uint8)


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

    x = int(cx - w / 2)
    y = int(cy - h / 2)
    paste_rgba(frame, img, x, y, w, h)


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

    x = int(cx - w / 2)
    y = int(bottom_y - h)
    paste_rgba(frame, img, x, y, w, h)


def alpha_bbox(img, threshold=8):
    if img is None or len(img.shape) < 3 or img.shape[2] < 4:
        return None

    cache_key = None

    if not img.flags.writeable:
        pointer = int(img.__array_interface__["data"][0])
        cache_key = (pointer, img.shape, img.strides, threshold)

        if cache_key in ALPHA_BBOX_CACHE:
            return ALPHA_BBOX_CACHE[cache_key]

    ys, xs = np.where(img[:, :, 3] > threshold)

    if len(xs) == 0:
        if cache_key is not None:
            ALPHA_BBOX_CACHE[cache_key] = None
        return None

    result = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

    if cache_key is not None:
        ALPHA_BBOX_CACHE[cache_key] = result

    return result


def has_visible_alpha(img):
    if img is None:
        return False

    if len(img.shape) < 3 or img.shape[2] < 4:
        return True

    return bool(np.any(img[:, :, 3] > 8))


def has_meaningful_visible_alpha(img):
    if not has_visible_alpha(img):
        return False

    if img.ndim < 3 or img.shape[2] < 4:
        return True

    visible_pixels = int(np.count_nonzero(img[:, :, 3] > 8))
    return visible_pixels >= max(4, int(img.shape[0] * img.shape[1] * 0.002))


def dominant_visible_colour(img):
    if not has_meaningful_visible_alpha(img):
        return None

    if img.ndim < 3 or img.shape[2] < 3:
        return None

    mask = img[:, :, 3] > 16 if img.shape[2] == 4 else np.ones(img.shape[:2], dtype=bool)
    bgr = np.mean(img[:, :, :3][mask], axis=0)
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def paste_rgba_centered_sized(frame, img, cx, cy, width, height, crop_alpha=False):
    if img is None:
        return

    if crop_alpha:
        bbox = alpha_bbox(img)

        if bbox:
            x1, y1, x2, y2 = bbox
            img = img[y1:y2, x1:x2]

    x = int(cx - width / 2)
    y = int(cy - height / 2)
    paste_rgba(frame, img, x, y, width, height)


def flush_gpu(frame):
    if GPU_COMPOSITOR is not None:
        GPU_COMPOSITOR.flush(frame)


def set_gpu_compositing_active(active):
    global GPU_COMPOSITING_ACTIVE
    GPU_COMPOSITING_ACTIVE = bool(active)


def select_skin_glyph(variants, height):
    usable = {
        density: image
        for density, image in variants.items()
        if has_meaningful_visible_alpha(image)
        and image.shape[1] <= image.shape[0] * 4
    }

    if not usable:
        return None, 1.0

    # @2x is a density marker, not a resolution threshold. osu! prefers the
    # sharpest source and converts it back to legacy logical coordinates.
    preferred_density = max(usable)
    image = usable.get(preferred_density)

    if image is None:
        preferred_density, image = next(iter(usable.items()))

    return image, preferred_density


def draw_skin_text(
    frame,
    text,
    glyphs,
    center_x,
    y,
    overlap,
    coordinate_scale,
    vertical_anchor="top",
    vertical_scale=1.0,
    alpha=1.0,
    tint=None,
):
    selected = []

    for character in str(text):
        image, density = select_skin_glyph(glyphs.get(character, {}), frame.shape[0])

        if image is None:
            return False

        scale = coordinate_scale / density
        selected.append((
            image,
            max(1, int(image.shape[1] * scale)),
            max(1, int(image.shape[0] * scale * vertical_scale)),
        ))

    if not selected:
        return False

    overlap_px = int(overlap * coordinate_scale)
    total_width = sum(width for _, width, _ in selected) - overlap_px * max(0, len(selected) - 1)
    x = int(center_x - total_width / 2)
    max_height = max(height for _, _, height in selected)
    top_y = int(y - max_height / 2) if vertical_anchor == "center" else int(y)

    for image, width, height in selected:
        draw_image = image

        if alpha < 0.999 or tint is not None:
            draw_image = image.copy()

            if tint is not None and draw_image.ndim == 3 and draw_image.shape[2] >= 3:
                target = np.array((tint[2], tint[1], tint[0]), dtype=np.float32) / 255.0
                draw_image[:, :, :3] = np.clip(draw_image[:, :, :3].astype(np.float32) * target, 0, 255).astype(np.uint8)

            if draw_image.ndim == 3 and draw_image.shape[2] == 4:
                draw_image[:, :, 3] = np.clip(draw_image[:, :, 3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        glyph_y = top_y + (max_height - height) // 2
        paste_rgba(frame, draw_image, x, glyph_y, width, height)
        x += width - overlap_px

    return True


def measure_skin_text(text, glyphs, overlap, coordinate_scale, frame_height):
    widths = []
    heights = []

    for character in str(text):
        image, density = select_skin_glyph(glyphs.get(character, {}), frame_height)

        if image is None:
            return None

        scale = coordinate_scale / density
        widths.append(max(1, int(image.shape[1] * scale)))
        heights.append(max(1, int(image.shape[0] * scale)))

    if not widths:
        return None

    overlap_px = int(overlap * coordinate_scale)
    return sum(widths) - overlap_px * max(0, len(widths) - 1), max(heights)


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

    paste_rgba(
        frame,
        source,
        int(cx - target_width / 2),
        int(top_y),
        max(1, target_width),
        max(1, body_h),
    )


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


def apply_temporal_motion_blur(frame, previous_frame, strength):
    strength = max(0, int(strength))

    if strength <= 0 or previous_frame is None or previous_frame.shape != frame.shape:
        return

    difference = cv2.absdiff(frame, previous_frame)
    change = np.max(difference, axis=2)
    mask = np.where(change > 6, 255, 0).astype(np.uint8)
    kernel_size = max(3, strength * 2 + 1)
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    alpha = (mask.astype(np.float32) / 255.0 * min(0.58, 0.18 + strength * 0.05))[:, :, None]
    blended = frame.astype(np.float32) * (1.0 - alpha) + previous_frame.astype(np.float32) * alpha
    frame[:] = np.clip(blended, 0, 255).astype(np.uint8)


def create_vignette_mask(width, height, strength):
    strength = max(0.0, min(1.0, float(strength)))

    if strength <= 0:
        return None

    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    distance = np.clip((xx * xx + yy * yy) / 1.6, 0.0, 1.0)
    return np.clip((1.0 - distance * strength) * 255.0, 0, 255).astype(np.uint8)


def apply_vignette(frame, mask):
    if mask is None:
        return

    frame[:] = (frame.astype(np.uint16) * mask[:, :, None].astype(np.uint16) // 255).astype(np.uint8)



def init_worker(ctx):
    global CTX, RESIZE_CACHE, ALPHA_BBOX_CACHE, GPU_COMPOSITOR, GPU_COMPOSITING_ACTIVE
    global PREVIOUS_RENDER_FRAME, PREVIOUS_RENDER_FRAME_ID
    cv2.setNumThreads(1)
    CTX = ctx
    RESIZE_CACHE = {}
    ALPHA_BBOX_CACHE = {}
    GPU_COMPOSITOR = None
    GPU_COMPOSITING_ACTIVE = True
    PREVIOUS_RENDER_FRAME = None
    PREVIOUS_RENDER_FRAME_ID = None

    if ctx.get("gpu_compositing"):
        from osu_mania_replay_renderer.gpu_compositor import create_gpu_compositor

        GPU_COMPOSITOR = create_gpu_compositor()


def process_pool_smoke_worker(value):
    """Small picklable worker used to validate frozen multiprocessing builds."""
    return os.getpid(), CTX.get("pool_smoke"), value * value


def run_process_pool_smoke_test():
    context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=context,
        initializer=init_worker,
        initargs=({"pool_smoke": "ready"},),
    ) as executor:
        child_pid, marker, result = executor.submit(process_pool_smoke_worker, 7).result(timeout=30)

    return child_pid != os.getpid() and marker == "ready" and result == 49


def draw_receptors(frame, skin, pressed, keys, lane_xs, column_widths, judge_y, note_widths):
    for lane in range(keys):
        lane_x = lane_xs[lane]
        lane_width = column_widths[lane]
        center_x = lane_x + lane_width // 2
        receptor_size = note_widths[lane]
        receptor_y = judge_y

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
            flush_gpu(frame)
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


def draw_hit_lighting(
    frame,
    skin,
    display_judgements,
    judgement_times,
    map_time,
    lane_xs,
    column_widths,
    judge_y,
    skin_scale,
    active_hold_lanes,
):
    normal = skin.get("hit_lighting_normal")
    long = skin.get("hit_lighting_long")
    cfg = skin.get("cfg", {})
    light_fps = max(1, int(cfg.get("light_frame_per_second") or 24))
    normal_frames = skin.get("hit_lighting_normal_frames") or []
    long_frames = skin.get("hit_lighting_long_frames") or []
    normal_duration = int(len(normal_frames) * 1000 / light_fps) if normal_frames else 120
    long_duration = int(len(long_frames) * 1000 / light_fps) if long_frames else 120
    lighting_duration = max(24, normal_duration, long_duration)

    if not has_visible_alpha(normal) and not has_visible_alpha(long):
        return

    lane_effects = {lane: ("long", 0) for lane in active_hold_lanes}
    latest_i = bisect_right(judgement_times, map_time) - 1
    visible_since = map_time - lighting_duration

    for index in range(latest_i, -1, -1):
        judgement = display_judgements[index]
        display_time = judgement.get("display_time", judgement["time"])

        if display_time < visible_since:
            break

        if judgement.get("stable_tick") or judgement.get("value", 0) <= 0:
            continue

        lane = judgement.get("lane")

        if lane is None or lane < 0 or lane >= len(lane_xs):
            continue

        kind = judgement.get("kind")
        effect = "long" if kind == "ln_head" else "normal"
        age = max(0, map_time - display_time)
        effect_duration = long_duration if effect == "long" else normal_duration
        if age <= max(24, effect_duration):
            previous = lane_effects.get(lane)
            if previous is None or age < previous[1]:
                lane_effects[lane] = (effect, age)

    normal_widths = cfg.get("lighting_n_widths")
    long_widths = cfg.get("lighting_l_widths")

    for lane, (effect, age) in lane_effects.items():
        frames = skin.get("hit_lighting_long_frames" if effect == "long" else "hit_lighting_normal_frames") or []
        if frames:
            frame_index = min(len(frames) - 1, int(age * light_fps / 1000.0))
            image = frames[frame_index]
        else:
            image = long if effect == "long" and has_visible_alpha(long) else normal

        if not has_visible_alpha(image):
            continue

        density_key = "hit_lighting_long_density" if effect == "long" else "hit_lighting_normal_density"
        density = max(1.0, float(skin.get(density_key, 1.0)))
        configured_widths = long_widths if effect == "long" else normal_widths

        if configured_widths and lane < len(configured_widths):
            target_width = max(1, int(configured_widths[lane] * skin_scale))
        else:
            target_width = max(1, int(image.shape[1] * skin_scale / density))

        target_height = max(1, int(image.shape[0] * target_width / max(1, image.shape[1])))
        center_x = lane_xs[lane] + column_widths[lane] // 2
        # LightingN/LightingL are centred on the judgement line according to
        # skin.ini, not on the note cap's visual centre.
        hit_center_y = judge_y
        paste_additive(
            frame,
            image,
            int(center_x - target_width / 2),
            int(hit_center_y - target_height / 2),
            target_width,
            target_height,
        )


def draw_stage_bottom(frame, skin, play_x, play_width, height, skin_scale):
    stage_bottom = skin.get("stage_bottom")

    if stage_bottom is None or not has_visible_alpha(stage_bottom):
        return

    # Stable uses a bottom-centred origin (top-centred for UpsideDown) and does
    # not stretch this element to the stage width. Transparent canvas padding is
    # meaningful and must be preserved for lane-cover positioning.
    cover_w = max(1, int(stage_bottom.shape[1] * skin_scale))
    cover_h = max(1, int(stage_bottom.shape[0] * skin_scale))
    cover_x = int(play_x + play_width / 2 - cover_w / 2)
    upside_down = bool(skin.get("cfg", {}).get("upside_down", False))
    cover_y = 0 if upside_down else height - cover_h
    paste_rgba(frame, stage_bottom, cover_x, cover_y, cover_w, cover_h)


def draw_hit_judgements(
    frame,
    skin,
    judgements,
    judgement_times,
    map_time,
    lane_xs,
    column_widths,
    judge_y,
    note_widths,
    custom_position=None,
):
    if not lane_xs:
        return

    play_center_x = (lane_xs[0] + lane_xs[-1] + column_widths[-1]) // 2
    cfg = skin.get("cfg", {})
    skin_scale = frame.shape[0] / 480
    score_position = cfg.get("score_position") if cfg.get("score_position") is not None else 300
    judgement_y = int(score_position * skin_scale)

    if custom_position is not None:
        play_center_x, judgement_y = custom_position
    latest_i = bisect_right(judgement_times, map_time) - 1

    # Stable LN combo ticks are internal combo events, not hit judgements. Do
    # not let their value=0 surface as a fake MISS while a hold is active.
    while latest_i >= 0 and not judgements[latest_i].get("counts_accuracy", True):
        latest_i -= 1

    if latest_i < 0:
        return

    latest = judgements[latest_i]
    display_time = latest.get("display_time", latest["time"])

    if map_time - display_time > 1000:
        return

    value = latest["value"]
    key = latest.get("image_key") or hit_image_key(value, latest.get("diff"))
    age = max(0, map_time - display_time)
    frames = skin.get("hit_image_frames", {}).get(key) or []
    if frames:
        frame_index = int(age * 60 / 1000.0) % len(frames)
        img = frames[frame_index]
    else:
        img = skin.get("hit_images", {}).get(key)

    if img is not None and has_visible_alpha(img):
        density = skin.get("hit_image_densities", {}).get(key, 1.0)
        resolution_scale = frame.shape[0] / 768.0 / max(1.0, density)
        paste_rgba_centered(
            frame,
            img,
            play_center_x,
            judgement_y,
            scale=resolution_scale,
        )


def draw_judgement_counter(frame, judgements, map_time, width, top_y, right_x=None, counts=None):
    if counts is None:
        counts = judgement_counts_at(judgements, map_time)
    else:
        counts = dict(counts)

    labels = [
        ("300g", counts["300g"], (120, 235, 255)),
        ("300", counts["300"], (220, 240, 255)),
        ("200", counts["200"], (100, 220, 120)),
        ("100", counts["100"], (240, 220, 80)),
        ("50", counts["50"], (240, 150, 70)),
        ("Miss", counts["0"], (230, 80, 80)),
    ]

    ui_scale = overlay_scale(frame.shape[0])
    x = right_x if right_x is not None else width - int(24 * ui_scale)
    y = top_y

    for label, count, color in labels:
        draw_ui_text(frame, f"{label}: {count}", (x, y), 0.48 * ui_scale, color, 1, "right")
        y += int(24 * ui_scale)

    return y


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


def rosu_mania_performance(osu_file, mods_int, counts):
    """Calculate mania stars/pp locally when osu!.db has no star rating.

    osu!.db only contains cached star ratings for maps that osu! has indexed.
    Replays shared through Telegram often come with an .osz that we extract to
    /tmp, so the database lookup returns None and the overlay showed pp: N/A.
    rosu-pp-py lets us calculate the same information directly from the .osu.
    """
    try:
        from rosu_pp_py import Beatmap, Difficulty, Performance
    except Exception:
        return None, None

    try:
        beatmap = Beatmap(path=str(osu_file))
        difficulty = Difficulty(mods=int(mods_int or 0)).calculate(beatmap)
        pp = Performance(
            mods=int(mods_int or 0),
            n_geki=int(counts.get("300g", 0)),
            n300=int(counts.get("300", 0)),
            n_katu=int(counts.get("200", 0)),
            n100=int(counts.get("100", 0)),
            n50=int(counts.get("50", 0)),
            misses=int(counts.get("0", 0)),
        ).calculate(beatmap)
        return float(getattr(difficulty, "stars", 0.0) or 0.0), float(getattr(pp, "pp", 0.0) or 0.0)
    except Exception:
        return None, None


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


def draw_pp_counter(frame, judgements, map_time, width, y, star_rating, right_x=None, counts=None):
    counts = counts if counts is not None else judgement_counts_at(judgements, map_time)
    pp = mania_pp_value(star_rating, counts)
    text = "pp: N/A" if pp is None else f"pp: {pp:.2f}"
    ui_scale = overlay_scale(frame.shape[0])
    x = right_x if right_x is not None else width - int(24 * ui_scale)
    draw_ui_text(frame, text, (x, y), 0.56 * ui_scale, (220, 220, 220), 1, "right")

    return y + int(28 * ui_scale)


def stable_key_bpm(times, states, map_time):
    # Show a responsive per-key tapping rate from the last actual press
    # intervals. This intentionally does not count events in fixed buckets:
    # the overlay is a "current rhythm" readout, so it should update as soon
    # as the second press exists and then fade only after inactivity.
    end_i = bisect_right(times, map_time)
    start_i = max(0, bisect_left(times, map_time - 4000))
    press_times = [times[i] for i in range(start_i, end_i) if states[i]]

    if len(press_times) < 2:
        return 0

    idle_ms = map_time - press_times[-1]
    if idle_ms > 1800:
        return 0

    intervals = [
        interval
        for interval in np.diff(press_times[-6:])
        if 20 <= interval <= 1800
    ]

    if not intervals:
        return 0

    weights = np.arange(1, len(intervals) + 1, dtype=np.float32)
    weighted_interval = float(np.average(np.asarray(intervals, dtype=np.float32), weights=weights))
    bpm = 60000.0 / max(1.0, weighted_interval)

    if idle_ms > 900:
        bpm *= max(0.0, 1.0 - ((idle_ms - 900) / 900.0))

    return min(999, int(round(bpm)))


def draw_key_input_overlay(frame, event_lanes, pressed, map_time, width, y, custom_position=None):
    ui_scale = overlay_scale(frame.shape[0])
    right_x = width - int(24 * ui_scale)
    window_ms = 1400
    lane_count = max(1, len(event_lanes))
    lane_w = max(20, int(27 * ui_scale))
    lane_gap = max(4, int(7 * ui_scale))
    total_w = lane_count * lane_w + (lane_count - 1) * lane_gap
    history_h = max(112, int(154 * ui_scale))
    key_h = max(17, int(22 * ui_scale))
    panel_h = int(18 * ui_scale) + history_h + int(7 * ui_scale) + key_h + int(28 * ui_scale)

    if custom_position is not None:
        center_x, center_y = custom_position
        x1 = int(center_x - total_w / 2)
        y = int(center_y - panel_h / 2)
        x1 = max(4, min(width - total_w - 4, x1))
        right_x = x1 + total_w
    else:
        x1 = right_x - total_w

    history_top = y + int(18 * ui_scale)
    history_bottom = history_top + history_h
    key_top = history_bottom + int(7 * ui_scale)

    cv2.rectangle(
        frame,
        (x1 - int(6 * ui_scale), y - int(10 * ui_scale)),
        (right_x + int(5 * ui_scale), key_top + key_h + int(28 * ui_scale)),
        (0, 0, 0),
        -1,
    )
    draw_ui_text(frame, "INPUT / BPM", (right_x, y), 0.42 * ui_scale, (190, 190, 195), 1, "right")

    for lane, (times, states) in enumerate(event_lanes):
        lane_x = x1 + lane * (lane_w + lane_gap)
        start_i = max(0, bisect_left(times, map_time - window_ms) - 1)
        end_i = bisect_right(times, map_time)

        for event_i in range(start_i, end_i):
            if not states[event_i]:
                continue

            press_time = times[event_i]
            press_y = history_bottom - int((map_time - press_time) / window_ms * history_h)
            press_y = max(history_top, min(history_bottom, press_y))
            cv2.line(
                frame,
                (lane_x + 2, press_y),
                (lane_x + lane_w - 2, press_y),
                (214, 218, 205),
                max(1, int(2 * ui_scale)),
                cv2.LINE_AA,
            )

            release_time = times[event_i + 1] if event_i + 1 < len(times) else map_time
            hold_duration = max(0, release_time - press_time)

            if hold_duration >= 120:
                release_time = min(map_time, release_time)
                release_y = history_bottom - int((map_time - release_time) / window_ms * history_h)
                release_y = max(history_top, min(history_bottom, release_y))
                top, bottom = sorted((press_y, release_y))
                if bottom > top:
                    hold_overlay = frame[top:bottom + 1, lane_x + 3:lane_x + lane_w - 2]
                    hold_overlay[:] = (
                        hold_overlay.astype(np.uint16) * 2 + np.array((135, 140, 128), dtype=np.uint16)
                    ) // 3

        is_pressed = bool(pressed[lane]) if lane < len(pressed) else False
        key_colour = (90, 205, 245) if is_pressed else (30, 31, 34)
        border_colour = (120, 225, 255) if is_pressed else (150, 126, 55)
        cv2.rectangle(frame, (lane_x, key_top), (lane_x + lane_w, key_top + key_h), key_colour, -1)
        cv2.rectangle(frame, (lane_x, key_top), (lane_x + lane_w, key_top + key_h), border_colour, max(1, int(ui_scale)), cv2.LINE_AA)
        draw_ui_text(
            frame,
            str(lane + 1),
            (lane_x + lane_w // 2, key_top + key_h - int(5 * ui_scale)),
            0.32 * ui_scale,
            (235, 235, 230),
            1,
            "center",
        )

        # A median interval produces a continuous, timing-accurate rate and is
        # robust against a single jack/chord outlier. Unlike fixed-window
        # counting it is not quantised in 20/30 BPM blocks.
        bpm = stable_key_bpm(times, states, map_time)
        draw_ui_text(
            frame,
            f"{bpm:03d}",
            (lane_x + lane_w // 2, key_top + key_h + int(17 * ui_scale)),
            0.32 * ui_scale,
            (205, 205, 210),
            1,
            "center",
        )

    return key_top + key_h + int(30 * ui_scale)


def draw_star_rating(frame, star_rating, height, custom_position=None):
    text = "SR: N/A" if star_rating is None else f"SR: {star_rating:.2f}*"
    ui_scale = overlay_scale(height)
    if custom_position is not None:
        draw_ui_text(frame, text, custom_position, 0.56 * ui_scale, anchor="center")
    else:
        draw_ui_text(frame, text, (int(24 * ui_scale), height - int(140 * ui_scale)), 0.56 * ui_scale)


def draw_timeline(frame, map_time, start_map_time, end_map_time, width, height, y, custom_position=None):
    progress = (map_time - start_map_time) / max(1, end_map_time - start_map_time)
    progress = max(0.0, min(1.0, progress))
    ui_scale = overlay_scale(height)
    radius = max(11, int(16 * ui_scale))
    right_x = width - int(24 * ui_scale)
    center = custom_position if custom_position is not None else (right_x - radius, y + radius)
    thickness = max(2, int(3 * ui_scale))
    cv2.circle(frame, center, radius, (38, 38, 43), -1, cv2.LINE_AA)

    if progress > 0:
        cv2.ellipse(
            frame,
            center,
            (radius, radius),
            -90,
            0,
            progress * 360,
            (145, 145, 150),
            -1,
            cv2.LINE_AA,
        )

    cv2.circle(frame, center, radius, (205, 205, 210), thickness, cv2.LINE_AA)

    elapsed = map_time - start_map_time
    duration = end_map_time - start_map_time
    draw_ui_text(
        frame,
        f"{format_clock(elapsed)} / {format_clock(duration)}",
        (center[0] - radius - int(9 * ui_scale), center[1] + int(5 * ui_scale)),
        0.42 * ui_scale,
        (215, 215, 220),
        1,
        "right",
    )


def build_difficulty_profile(notes, start_time, end_time, sample_count=220):
    sample_count = max(60, int(sample_count))
    duration = max(1, end_time - start_time)
    raw = np.zeros(sample_count, dtype=np.float32)
    grouped = {}

    for note in notes:
        grouped.setdefault(note["time"], []).append(note)

    previous_time = None

    for note_time, chord in sorted(grouped.items()):
        index = min(sample_count - 1, max(0, int((note_time - start_time) / duration * (sample_count - 1))))
        interval = 500 if previous_time is None else max(35, note_time - previous_time)
        speed = min(8.0, 500.0 / interval)
        chord_weight = 1.0 + 0.55 * (len(chord) - 1)
        ln_weight = 1.0 + 0.2 * sum(1 for note in chord if note.get("end_time") is not None)
        raw[index] += speed * chord_weight * ln_weight
        previous_time = note_time

    kernel = np.array([0.08, 0.18, 0.34, 0.55, 1.0, 0.55, 0.34, 0.18, 0.08], dtype=np.float32)
    smoothed = np.convolve(raw, kernel, mode="same")
    ceiling = float(np.percentile(smoothed[smoothed > 0], 96)) if np.any(smoothed > 0) else 1.0
    return np.clip(smoothed / max(ceiling, 0.001), 0.0, 1.0).tolist()


def draw_difficulty_graph(
    frame,
    profile,
    map_time,
    start_time,
    end_time,
    width,
    height,
    play_x,
    play_width,
    custom_position=None,
):
    if not profile:
        return

    ui_scale = overlay_scale(height)
    gap = int(28 * ui_scale)
    right_start = play_x + play_width + gap
    max_graph_w = int(416 * ui_scale)
    graph_h = max(52, int(82 * ui_scale))

    if custom_position is not None:
        center_x, center_y = custom_position
        graph_w = min(max_graph_w, width - 8)
        x1 = max(0, min(width - graph_w, int(center_x - graph_w / 2)))
        x2 = x1 + graph_w
        y1 = max(0, min(height - graph_h, int(center_y - graph_h / 2)))
        y2 = y1 + graph_h
    else:
        x2 = width
        x1 = max(right_start, width - max_graph_w)
        y2 = height
        y1 = y2 - graph_h

    if x2 - x1 < 100:
        return

    xs = np.linspace(x1, x2, len(profile)).astype(np.int32)
    ys = np.array([y2 - int(value * graph_h) for value in profile], dtype=np.int32)
    points = np.column_stack((xs, ys))
    progress = max(0.0, min(1.0, (map_time - start_time) / max(1, end_time - start_time)))
    passed_i = min(len(points) - 1, max(0, int(progress * (len(points) - 1))))

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

    if passed_i > 0:
        passed_poly = np.vstack((points[:passed_i + 1], [points[passed_i, 0], y2], [x1, y2]))
        cv2.fillPoly(frame, [passed_poly], (18, 78, 34))

    if passed_i < len(points) - 1:
        future_poly = np.vstack(([points[passed_i, 0], y2], points[passed_i:], [x2, y2]))
        cv2.fillPoly(frame, [future_poly], (34, 34, 38))

    if passed_i > 0:
        cv2.polylines(frame, [points[:passed_i + 1]], False, (70, 220, 100), max(1, int(2 * ui_scale)), cv2.LINE_AA)

    if passed_i < len(points) - 1:
        cv2.polylines(frame, [points[passed_i:]], False, (135, 135, 142), max(1, int(2 * ui_scale)), cv2.LINE_AA)

    marker_x = int(x1 + (x2 - x1) * progress)
    cv2.line(frame, (marker_x, y1), (marker_x, y2), (225, 225, 228), 1, cv2.LINE_AA)


def replay_rank(accuracy, mods_int=0):
    if accuracy >= 100.0 - 1e-9:
        rank = "X"
    elif accuracy > 95.0:
        rank = "S"
    elif accuracy > 90.0:
        rank = "A"
    elif accuracy > 80.0:
        rank = "B"
    elif accuracy > 70.0:
        rank = "C"
    else:
        rank = "D"

    if rank in ("X", "S") and mods_int & (8 | 1024 | 1048576):
        rank += "H"

    return rank


def paste_legacy_ranking_asset(frame, image, density, x, y, origin="top_left", crop_alpha=False, max_size=None):
    if image is None or not has_visible_alpha(image):
        return

    if crop_alpha:
        bbox = alpha_bbox(image)

        if bbox:
            x1, y1, x2, y2 = bbox
            image = image[y1:y2, x1:x2]

    interface_scale = frame.shape[0] / 768
    image_scale = interface_scale / max(1.0, density)
    target_w = max(1, int(image.shape[1] * image_scale))
    target_h = max(1, int(image.shape[0] * image_scale))

    if max_size:
        max_w, max_h = max_size
        ratio = min(1.0, max_w / target_w, max_h / target_h)
        target_w = max(1, int(target_w * ratio))
        target_h = max(1, int(target_h * ratio))
    draw_x = int(x)
    draw_y = int(y)

    if origin == "center":
        draw_x -= target_w // 2
        draw_y -= target_h // 2
    elif origin == "top_right":
        draw_x -= target_w

    paste_rgba(frame, image, draw_x, draw_y, target_w, target_h)


def draw_ranking_hit_asset(frame, skin, key, x, y, interface_scale):
    variants = skin.get("ranking_hit_images", {}).get(key, {})
    image, density = select_skin_glyph(variants, frame.shape[0])

    if image is None:
        image = skin.get("hit_images", {}).get(key)
        density = 1.0

    if not has_visible_alpha(image):
        return False

    bbox = alpha_bbox(image)

    if bbox:
        x1, y1, x2, y2 = bbox
        tiny_asset = x2 - x1 < 6 or y2 - y1 < 6

        if key != "300g" and tiny_asset:
            return False

        image = image[y1:y2, x1:x2]
    else:
        tiny_asset = False

    if key == "300g" and tiny_asset:
        target_h = max(1, int(image.shape[0] * interface_scale / max(1.0, density)))
    else:
        target_h = max(1, int(28 * interface_scale))
    target_w = max(1, int(image.shape[1] * target_h / max(1, image.shape[0])))
    max_w = int(112 * interface_scale)

    if target_w > max_w:
        target_h = max(1, int(target_h * max_w / target_w))
        target_w = max_w

    paste_rgba(frame, image, int(x), int(y - target_h / 2), target_w, target_h)
    return True


def draw_result_mods(frame, skin, mods, rank_x, rank_y, interface_scale):
    acronyms = [mod for mod in str(mods).split() if mod and mod != "NM"]

    if not acronyms:
        return

    icon_size = max(32, int(48 * interface_scale))
    gap = max(5, int(6 * interface_scale))
    total_h = len(acronyms) * icon_size + max(0, len(acronyms) - 1) * gap
    x = frame.shape[1] - icon_size - int(12 * interface_scale)
    y = int(rank_y - total_h / 2)

    for acronym in acronyms:
        icon = skin.get("mod_icons", {}).get(acronym)

        if has_meaningful_visible_alpha(icon):
            paste_rgba(frame, icon, x, y, icon_size, icon_size)
        else:
            cv2.rectangle(frame, (x, y), (x + icon_size, y + icon_size), (28, 31, 38), -1)
            cv2.rectangle(frame, (x, y), (x + icon_size, y + icon_size), (210, 214, 220), 1, cv2.LINE_AA)
            draw_ui_text(
                frame,
                acronym,
                (x + icon_size // 2, y + int(icon_size * 0.64)),
                0.38 * interface_scale,
                (242, 242, 245),
                1,
                "center",
            )

        y += icon_size + gap


def prepare_results_background(image, width, height, opacity=0.62):
    if image is None:
        return None

    source_h, source_w = image.shape[:2]
    scale = max(width / max(1, source_w), height / max(1, source_h))
    resized_w = max(1, int(source_w * scale))
    resized_h = max(1, int(source_h * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x = max(0, (resized_w - width) // 2)
    y = max(0, (resized_h - height) // 2)
    cropped = resized[y:y + height, x:x + width, :3]
    return np.clip(cropped.astype(np.float32) * opacity, 0, 255).astype(np.uint8)


def draw_results_screen(
    frame,
    skin,
    title,
    mapper,
    player,
    mods,
    counts,
    accuracy,
    max_combo,
    score,
    pp,
    rank,
    perfect=False,
    background=None,
):
    height, width = frame.shape[:2]
    interface_scale = height / 768
    skin_version = skin.get("cfg", {}).get("skin_version", 1.0)
    is_v2 = skin_version >= 2.0
    panel = skin.get("ranking_panel")

    if background is not None and background.shape[:2] == frame.shape[:2]:
        frame[:] = background
    else:
        frame[:] = (24, 27, 36)

    panel_y = (102 if is_v2 else 74) * interface_scale
    draw_song_header(frame, title, mapper, player, panel_y)

    if panel is not None:
        paste_legacy_ranking_asset(
            frame,
            panel,
            skin.get("ranking_panel_density", 1.0),
            0,
            panel_y,
        )
    elements = skin.get("ranking_elements", {})
    element_densities = skin.get("ranking_element_densities", {})
    rank_y = (320 if is_v2 else 272) * interface_scale
    rank_x = width - 192 * interface_scale
    paste_legacy_ranking_asset(
        frame,
        skin.get("ranking_ranks", {}).get(rank),
        skin.get("ranking_rank_densities", {}).get(rank, 1.0),
        rank_x,
        rank_y,
        "center",
        crop_alpha=True,
        max_size=(int(250 * interface_scale), int(250 * interface_scale)),
    )
    draw_result_mods(frame, skin, mods, rank_x, rank_y, interface_scale)
    paste_legacy_ranking_asset(
        frame,
        elements.get("maxcombo"),
        element_densities.get("maxcombo", 1.0),
        8 * interface_scale,
        (480 if is_v2 else 500) * interface_scale,
    )
    paste_legacy_ranking_asset(
        frame,
        elements.get("accuracy"),
        element_densities.get("accuracy", 1.0),
        291 * interface_scale,
        (480 if is_v2 else 500) * interface_scale,
    )
    paste_legacy_ranking_asset(
        frame,
        elements.get("graph"),
        element_densities.get("graph", 1.0),
        256 * interface_scale,
        (608 if is_v2 else 576) * interface_scale,
    )

    if perfect:
        paste_legacy_ranking_asset(
            frame,
            elements.get("perfect"),
            element_densities.get("perfect", 1.0),
            (416 if is_v2 else 320) * interface_scale,
            688 * interface_scale,
            "center",
        )

    paste_legacy_ranking_asset(
        frame,
        elements.get("title"),
        element_densities.get("title", 1.0),
        width - 32 * interface_scale,
        0,
        "top_right",
    )

    text_scale = max(0.55, interface_scale)

    rows = [
        (("300", counts.get("300", 0)), ("300g", counts.get("300g", 0))),
        (("200", counts.get("200", 0)), ("100", counts.get("100", 0))),
        (("50", counts.get("50", 0)), ("Miss", counts.get("0", 0))),
    ]
    row_ys = (274, 368, 462)
    score_glyphs = skin.get("score_glyphs", {})
    score_overlap = skin.get("cfg", {}).get("score_overlap", 0)

    for row_y, row in zip(row_ys, rows):
        for label_x, value_x, (label, value) in zip((21, 340), (204, 522), row):
            image_key = "0" if label == "Miss" else label

            if not draw_ranking_hit_asset(
                frame,
                skin,
                image_key,
                label_x * interface_scale,
                (row_y - 9) * interface_scale,
                interface_scale,
            ):
                draw_ui_text(frame, label, (int(label_x * interface_scale), int(row_y * interface_scale)), 0.54 * text_scale, (185, 210, 235), 1)

            value_center_x = value_x - 34

            if not draw_skin_text(frame, str(value), score_glyphs, value_center_x * interface_scale, (row_y - 25) * interface_scale, score_overlap, interface_scale):
                draw_ui_text(frame, str(value), (int(value_x * interface_scale), int(row_y * interface_scale)), 0.72 * text_scale, (250, 250, 252), 1, "right")

    if not draw_skin_text(frame, str(max_combo), score_glyphs, 140 * interface_scale, 540 * interface_scale, score_overlap, interface_scale):
        draw_ui_text(frame, str(max_combo), (int(145 * interface_scale), int(575 * interface_scale)), 0.68 * text_scale, (250, 250, 252), 1, "center")

    accuracy_number = f"{accuracy:.2f}"
    accuracy_metrics = measure_skin_text(accuracy_number, score_glyphs, score_overlap, interface_scale, height)

    if accuracy_metrics:
        number_w, number_h = accuracy_metrics
        suffix_scale = 0.82 * interface_scale
        (percent_w, percent_h), _ = cv2.getTextSize("%", cv2.FONT_HERSHEY_SIMPLEX, suffix_scale, 1)
        gap = int(8 * interface_scale)
        group_w = number_w + gap + percent_w
        group_x = int(430 * interface_scale - group_w / 2)
        number_center_x = group_x + number_w / 2
        accuracy_y = int(540 * interface_scale)
        draw_skin_text(frame, accuracy_number, score_glyphs, number_center_x, accuracy_y, score_overlap, interface_scale)
        draw_ui_text(
            frame,
            "%",
            (group_x + number_w + gap, accuracy_y + max(number_h, percent_h)),
            suffix_scale,
            (255, 255, 255),
            1,
        )
    else:
        draw_ui_text(frame, f"{accuracy_number}%", (int(430 * interface_scale), int(575 * interface_scale)), 0.68 * text_scale, (255, 255, 255), 1, "center")

    if not draw_skin_text(frame, f"{score:07d}", score_glyphs, 354 * interface_scale, 130 * interface_scale, score_overlap, interface_scale):
        draw_ui_text(frame, f"{score:,}", (int(354 * interface_scale), int(160 * interface_scale)), 0.72 * text_scale, (250, 250, 252), 1, "center")

    if pp is None:
        draw_ui_text(frame, "pp  N/A", (int(rank_x), int(575 * interface_scale)), 0.58 * text_scale, (235, 240, 245), 1, "center")
    else:
        pp_value = f"{pp:.2f}"
        pp_metrics = measure_skin_text(pp_value, score_glyphs, score_overlap, interface_scale, height)
        suffix_scale = 0.82 * interface_scale
        (suffix_w, suffix_h), _ = cv2.getTextSize("pp", cv2.FONT_HERSHEY_SIMPLEX, suffix_scale, 1)
        gap = int(8 * interface_scale)

        if pp_metrics:
            number_w, number_h = pp_metrics
            group_w = number_w + gap + suffix_w
            group_x = int(rank_x - group_w / 2)
            number_center_x = group_x + number_w / 2
            pp_y = int(540 * interface_scale)
            draw_skin_text(frame, pp_value, score_glyphs, number_center_x, pp_y, score_overlap, interface_scale)
            draw_ui_text(
                frame,
                "pp",
                (group_x + number_w + gap, pp_y + max(number_h, suffix_h)),
                suffix_scale,
                (245, 245, 248),
                1,
            )
        else:
            draw_ui_text(frame, f"{pp_value} pp", (int(rank_x), int(575 * interface_scale)), 0.58 * text_scale, (235, 240, 245), 1, "center")

    if skin.get("ranking_ranks", {}).get(rank) is None:
        draw_ui_text(frame, rank, (int(rank_x), int(rank_y)), 3.0 * text_scale, (245, 245, 250), 2, "center")


def write_render_frame(output_dir, frame_id, frame):
    flush_gpu(frame)
    stream_mode = CTX.get("frame_stream_mode", "mjpeg")
    jpeg_quality = int(CTX.get("frame_jpeg_quality", 94))

    if FRAME_STREAM is not None:
        if stream_mode == "raw":
            FRAME_STREAM.write(np.ascontiguousarray(frame).data)
            return

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )

        if not ok:
            raise RuntimeError(f"Could not encode frame {frame_id}")

        FRAME_STREAM.write(memoryview(encoded))
        return

    out = Path(output_dir) / f"frame_{frame_id:07d}.jpg"
    cv2.imwrite(
        str(out),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )


def latest_judgement(judgements, lane, kind, time):
    for judgement in judgements:
        if judgement["lane"] == lane and judgement["kind"] == kind and judgement["time"] == time:
            return judgement

    return None


def frame_worker(frame_id):
    global PREVIOUS_RENDER_FRAME, PREVIOUS_RENDER_FRAME_ID
    width = CTX["width"]
    height = CTX["height"]
    fps = CTX["fps"]
    start_map_time = CTX["start_map_time"]
    end_map_time = CTX["end_map_time"]
    gameplay_end_time = CTX.get("gameplay_end_time", end_map_time)
    results_start_time = CTX.get("results_start_time", end_map_time + 1)
    notes = CTX["notes"]
    keys = CTX["keys"]
    title = CTX["title"]
    mapper = CTX.get("mapper", "Unknown")
    player = CTX.get("player", "Unknown")
    mods = CTX.get("mods", "")
    output_dir = CTX["output_dir"]
    scroll_time_ms = CTX["scroll_time_ms"]
    motion_blur = CTX.get("motion_blur", 0)
    speed_multiplier = CTX["speed_multiplier"]
    mirror = CTX["mirror"]
    skin = CTX["skin"]
    judgements = CTX["judgements"]
    display_judgements = CTX.get("display_judgements", judgements)
    judgement_times = CTX.get("judgement_times") or [j.get("display_time", j["time"]) for j in display_judgements]
    combo_changed_times = CTX.get("combo_changed_times", [])
    cumulative_counts = CTX.get("cumulative_counts", [])
    events = CTX["events"]
    event_lanes = CTX.get("event_lanes", [])
    ln_hold_lanes = CTX.get("ln_hold_lanes", [])
    note_times = CTX.get("note_times", [])
    judgement_lookup = CTX.get("judgement_lookup", {})
    star_rating = CTX.get("star_rating")
    difficulty_profile = CTX.get("difficulty_profile", [])
    show_side_overlay = CTX.get("show_side_overlay", True)
    show_strain_graph = CTX.get("show_strain_graph", True)
    colour_combo_during_holds = CTX.get("colour_combo_during_holds", True)
    layout_positions = CTX.get("layout_positions", {})
    vignette_mask = CTX.get("vignette_mask")

    real_elapsed = int(frame_id * 1000 / fps)
    map_time = start_map_time + int(real_elapsed * speed_multiplier)

    if map_time >= results_start_time:
        frame = CTX["results_frame"].copy()
        write_render_frame(output_dir, frame_id, frame)
        return frame_id

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    set_gpu_compositing_active(False)

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
        play_x = int(cfg["column_start"] * scale_x)
    else:
        play_x = (width - play_width) // 2

    custom_playfield = layout_point(layout_positions, "playfield", width, height)

    if custom_playfield is not None:
        play_x = max(0, min(width - play_width, custom_playfield[0] - play_width // 2))

    raw_hit_position = cfg["hit_position"] if cfg["hit_position"] is not None else 402
    hit_position = int(raw_hit_position)
    judge_y = int(hit_position * scale_y)

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

    flush_gpu(frame)
    cv2.rectangle(frame, (play_x - 16, top_y), (play_x + play_width + 16, height), (0, 0, 0), -1)

    for lane in range(keys):
        colour = skin_colour_to_bgra(cfg["colours"].get(f"Colour{lane + 1}"))
        fill_rgba_rect(frame, lane_xs[lane], top_y, lane_xs[lane] + column_widths[lane], height, colour)

    # From here until the stage bottom is drawn most work is repeated skin
    # texture composition: receptors, lights, notes and LN pieces. Keep that
    # as a single GPU batch and read it back once before CPU-only overlays.
    set_gpu_compositing_active(True)

    ui_scale = overlay_scale(height)
    state_i = bisect_right(judgement_times, map_time) - 1

    if state_i >= 0:
        state = display_judgements[state_i]
        combo = state["combo"]
        accuracy = state["accuracy"]
        combo_changed_at = combo_changed_times[state_i]
        current_counts = cumulative_counts[state_i]
    else:
        combo = 0
        accuracy = 100.0
        combo_changed_at = None
        current_counts = {"300g": 0, "300": 0, "200": 0, "100": 0, "50": 0, "0": 0}

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
                lane = event["lane"]

                if 0 <= lane < keys:
                    pressed[lane] = event["pressed"]
            else:
                break

    if cfg["keys_under_notes"]:
        draw_receptors(frame, skin, pressed, keys, lane_xs, column_widths, judge_y, note_widths)

    draw_stage_lights(frame, skin, pressed, lane_xs, column_widths, height, cfg, skin_scale)

    start_i = max(0, bisect_left(note_times, visible_start) - 128) if note_times else 0
    active_ln_hold = False
    active_hold_lanes = []

    for lane, lane_holds in enumerate(ln_hold_lanes):
        if lane >= len(pressed) or not pressed[lane] or not lane_holds[0]:
            continue

        hold_starts, hold_ends = lane_holds
        hold_i = bisect_right(hold_starts, map_time) - 1

        if hold_i >= 0 and map_time < hold_ends[hold_i]:
            active_ln_hold = True
            active_hold_lanes.append(lane)

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
        hit_y = judge_y
        lane_scroll_speed = max(0.01, (hit_y - top_y) / effective_scroll_time_ms)

        y_head = int(hit_y - (note_time - map_time) * lane_scroll_speed)
        y_tail = int(hit_y - (end_time - map_time) * lane_scroll_speed)

        if note["end_time"] is not None:
            head_display_time = head_judgement.get("display_time", head_judgement["time"]) if head_judgement else note_time
            tail_display_time = tail_judgement.get("display_time", tail_judgement["time"]) if tail_judgement else end_time
            head_hit = head_judgement is not None and head_judgement["value"] > 0
            missed_ln_scrolling_out = head_judgement is not None and head_judgement["value"] <= 0 and map_time >= head_display_time
            head_visual_y = hit_y - note_w * 0.5 if map_time >= head_display_time and head_hit else y_head
            tail_visual_y = min(y_tail, y_head) if missed_ln_scrolling_out else min(y_tail, hit_y - note_w * 0.5)
            visible_top = min(head_visual_y, tail_visual_y)
            visible_bottom = max(head_visual_y, tail_visual_y)

            if visible_bottom >= top_y and visible_top <= height:
                body_img = skin["ln_bodies"][lane]
                head_img = skin["ln_heads"][lane]
                tail_img = skin["ln_tails"][lane]
                tail_explicit = True
                tail_explicit_flags = skin.get("ln_tail_explicit")
                if tail_explicit_flags is not None and lane < len(tail_explicit_flags):
                    tail_explicit = bool(tail_explicit_flags[lane])

                visual_span = abs(float(head_visual_y) - float(tail_visual_y))
                if body_img is not None and visual_span > 2:
                    cap_pad = 0
                    body_top = max(top_y, int(visible_top - cap_pad))
                    body_bottom = min(height, int(visible_bottom + cap_pad))
                    body_h = max(1, body_bottom - body_top)
                    paste_ln_body(frame, body_img, center_x, body_top, body_top + body_h, note_w)
                elif body_img is None:
                    flush_gpu(frame)
                    cv2.rectangle(
                        frame,
                        (note_x, visible_top),
                        (note_x + note_w, visible_bottom),
                        (150, 150, 150),
                        -1,
                    )

                if head_img is not None:
                    if head_hit and head_display_time <= map_time < tail_display_time and pressed[lane]:
                        active_ln_hold = True

                    if map_time < head_display_time:
                        paste_rgba_centered(frame, head_img, center_x, y_head, scale=skin_scale, max_width=note_w)
                    elif head_hit and map_time < tail_display_time:
                        paste_rgba_centered(frame, head_img, center_x, head_visual_y, scale=skin_scale, max_width=note_w)

                if (
                    tail_img is not None
                    and tail_visual_y >= top_y
                    and tail_visual_y <= height
                    and not missed_ln_scrolling_out
                    and (tail_judgement is None or map_time < tail_judgement.get("display_time", tail_judgement["time"]))
                    and (tail_explicit or body_img is None)
                ):
                    if has_visible_alpha(tail_img):
                        paste_rgba_centered(frame, tail_img, center_x, tail_visual_y, scale=skin_scale, max_width=note_w)

        else:
            note_img = skin["notes"][lane]

            if note_img is not None:
                paste_rgba_centered(frame, note_img, center_x, y_head, scale=skin_scale, max_width=note_w)
            else:
                flush_gpu(frame)
                cv2.rectangle(
                    frame,
                    (note_x, y_head - note_h // 2),
                    (note_x + note_w, y_head + note_h // 2),
                    (90, 160, 220),
                    -1,
                )

    draw_stage_bottom(frame, skin, play_x, play_width, height, skin_scale)
    flush_gpu(frame)
    set_gpu_compositing_active(False)

    if show_strain_graph:
        draw_difficulty_graph(
            frame,
            difficulty_profile,
            map_time,
            start_map_time,
            gameplay_end_time,
            width,
            height,
            play_x,
            play_width,
            layout_point(layout_positions, "strain_graph", width, height),
        )

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

    draw_hit_lighting(
        frame,
        skin,
        display_judgements,
        judgement_times,
        map_time,
        lane_xs,
        column_widths,
        judge_y,
        skin_scale,
        active_hold_lanes,
    )

    if vignette_mask is not None:
        flush_gpu(frame)
        apply_vignette(frame, vignette_mask)
    draw_hit_judgements(
        frame,
        skin,
        display_judgements,
        judgement_times,
        map_time,
        lane_xs,
        column_widths,
        judge_y,
        note_widths,
        layout_point(layout_positions, "judgement", width, height),
    )

    if combo > 0:
        combo_center_y = int((cfg.get("combo_position") if cfg.get("combo_position") is not None else 111) * skin_scale)
        play_center_x = play_x + play_width // 2
        custom_combo = layout_point(layout_positions, "combo", width, height)

        if custom_combo is not None:
            play_center_x, combo_center_y = custom_combo
        combo_age = max(0, map_time - combo_changed_at) if combo_changed_at is not None else 200
        bounce = 1.0

        if combo_age < 160:
            bounce += 0.12 * np.sin(np.pi * combo_age / 160.0)

        combo_scale = skin_scale * 0.72 * bounce
        combo_tint = None
        combo_colours = cfg.get("combo_colours", [])

        if colour_combo_during_holds and active_ln_hold:
            combo_tint = dominant_visible_colour(skin.get("hit_images", {}).get("300"))

            if combo_tint is None and combo_colours:
                combo_tint = combo_colours[(combo - 1) % len(combo_colours)]

        native_combo_drawn = draw_skin_text(
            frame,
            str(combo),
            skin.get("combo_glyphs", {}),
            play_center_x,
            combo_center_y,
            cfg.get("combo_overlap", 0),
            combo_scale,
            vertical_anchor="center",
            tint=combo_tint,
        )

        if not native_combo_drawn:
            draw_ui_text(frame, str(combo), (play_center_x, combo_center_y), 0.58 * ui_scale * bounce, anchor="center")

    flush_gpu(frame)

    if show_side_overlay:
        custom_stats = layout_point(layout_positions, "side_stats", width, height)

        if custom_stats is not None:
            stats_width, stats_height = side_stats_dimensions(height)
            right_x = min(width - 4, max(stats_width, custom_stats[0] + stats_width // 2))
            stats_y = max(4, custom_stats[1] - stats_height // 2)
        else:
            right_x = width - int(24 * ui_scale)
            stats_y = int(72 * ui_scale)
        draw_ui_text(frame, f"{combo}x", (right_x, stats_y), 0.88 * ui_scale, (242, 242, 245), 1, "right")
        stats_y += int(32 * ui_scale)
        draw_ui_text(frame, f"{accuracy:.2f}%", (right_x, stats_y), 0.58 * ui_scale, (225, 225, 230), 1, "right")
        stats_y += int(38 * ui_scale)
        stats_y = draw_judgement_counter(
            frame,
            judgements,
            map_time,
            width,
            stats_y,
            right_x,
            current_counts,
        )
        stats_y = draw_pp_counter(
            frame,
            judgements,
            map_time,
            width,
            stats_y + int(10 * ui_scale),
            star_rating,
            right_x,
            current_counts,
        )
        stats_y = draw_key_input_overlay(
            frame,
            event_lanes,
            pressed,
            map_time,
            width,
            stats_y + int(8 * ui_scale),
            layout_point(layout_positions, "key_input", width, height),
        )
        draw_timeline(
            frame,
            map_time,
            start_map_time,
            gameplay_end_time,
            width,
            height,
            stats_y + int(8 * ui_scale),
            layout_point(layout_positions, "timeline", width, height),
        )
        draw_star_rating(
            frame,
            star_rating,
            height,
            layout_point(layout_positions, "star_rating", width, height),
        )

    flush_gpu(frame)
    unblurred_frame = frame.copy() if motion_blur > 0 else None

    if motion_blur > 0 and PREVIOUS_RENDER_FRAME_ID == frame_id - 1:
        apply_temporal_motion_blur(frame, PREVIOUS_RENDER_FRAME, motion_blur)

    if motion_blur > 0:
        PREVIOUS_RENDER_FRAME = unblurred_frame
        PREVIOUS_RENDER_FRAME_ID = frame_id

    write_render_frame(output_dir, frame_id, frame)

    return frame_id


def frame_batch_worker(frame_ids):
    global FRAME_STREAM, PREVIOUS_RENDER_FRAME, PREVIOUS_RENDER_FRAME_ID

    frame_ids = list(frame_ids)

    if not frame_ids:
        return 0, 0, ""

    segment_path = Path(CTX["segments_dir"]) / f"segment_{frame_ids[0]:07d}.mjpg"
    PREVIOUS_RENDER_FRAME = None
    PREVIOUS_RENDER_FRAME_ID = None

    with open(segment_path, "wb", buffering=1024 * 1024) as stream:
        FRAME_STREAM = stream

        try:
            for frame_id in frame_ids:
                frame_worker(frame_id)
        finally:
            FRAME_STREAM = None

    return frame_ids[0], frame_ids[-1] + 1, str(segment_path)


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
    video_encoder_attempts=None,
    judgement_counts_reconciled=False,
    simulated_judgement_counts=None,
    visual_options=None,
    missing_skin_elements=None,
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
        "video_encoder_attempts": video_encoder_attempts or [],
        "judgement_counts_reconciled": judgement_counts_reconciled,
        "simulated_judgement_counts": simulated_judgement_counts or {},
        "missing_skin_elements": missing_skin_elements or [],
        "visual_options": visual_options or {},
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


def render_video(
    osu_file,
    skin_folder,
    output_file,
    replay_file,
    scroll_speed_value,
    resolution,
    motion_blur=0,
    progress_callback=None,
    show_side_overlay=True,
    show_strain_graph=True,
    vignette_strength=0,
    results_background_opacity=0.62,
    results_duration=4.5,
    show_results_screen=True,
    colour_combo_during_holds=True,
    gpu_compositing=True,
    layout_positions=None,
    cancel_callback=None,
):
    ensure_not_cancelled(cancel_callback)
    prepare_started = time.time()

    def report_prepare(percent, text, expected_remaining=None):
        if not progress_callback:
            return

        if expected_remaining is not None:
            text = f"{text} | render starts in ~{format_duration(expected_remaining)}"

        progress_callback(percent, text)

    report_prepare(0, "Preparing render: reading beatmap and replay...", 8)

    beatmap = parse_osu(osu_file)
    ensure_not_cancelled(cancel_callback)
    report_prepare(1, f"Preparing render: loaded beatmap with {len(beatmap.notes):,} objects", 7)

    width, height = map(int, resolution.split("x"))
    try:
        fps = int(os.environ.get("MANIA_RENDERER_FPS", "60"))
    except ValueError:
        fps = 60

    fps = max(15, min(240, fps))

    try:
        output_fps = int(os.environ.get("MANIA_RENDERER_OUTPUT_FPS", str(fps)))
    except ValueError:
        output_fps = fps

    output_fps = max(15, min(240, output_fps))

    try:
        frame_jpeg_quality = int(os.environ.get("MANIA_RENDERER_FRAME_JPEG_QUALITY", "94"))
    except ValueError:
        frame_jpeg_quality = 94

    frame_jpeg_quality = max(60, min(100, frame_jpeg_quality))
    frame_stream_mode = os.environ.get("MANIA_RENDERER_FRAME_STREAM", "mjpeg").strip().lower()

    if frame_stream_mode not in {"mjpeg", "raw"}:
        frame_stream_mode = "mjpeg"
    background_path = Path(osu_file).parent / beatmap.background_file
    background_image = cv2.imread(str(background_path), cv2.IMREAD_COLOR) if background_path.exists() else None
    results_background = prepare_results_background(background_image, width, height, results_background_opacity)

    replay = get_replay(replay_file) if replay_file else None
    mod_settings = mod_settings_from_replay(replay) if replay else {
        "mods_int": 0,
        "speed_multiplier": 1.0,
        "mirror": False,
        "mods": "NM",
        "nightcore_pitch": False,
        "score_v2": False,
    }

    speed_multiplier = mod_settings["speed_multiplier"]
    mirror = mod_settings["mirror"]
    nightcore_pitch = mod_settings["nightcore_pitch"]
    score_v2 = mod_settings.get("score_v2", False)
    overall_difficulty = beatmap.overall_difficulty
    is_convert = beatmap.mode != 3

    notes = [{"lane": n.lane, "time": n.time, "end_time": n.end_time} for n in beatmap.notes]
    star_rating = read_mania_star_rating(osu_file, beatmap.md5_hash, mod_settings["mods_int"]) if replay_file else None

    report_prepare(2, "Preparing render: decoding replay inputs...", 6)

    raw_events = get_replay_events(replay_file, beatmap.keys) if replay_file else []
    events = raw_events
    replay_accuracy = stable_mania_accuracy_from_replay(replay, score_v2) if replay else 100.0
    target_counts = replay_judgement_counts(replay) if replay else None
    ensure_not_cancelled(cancel_callback)
    report_prepare(3, f"Preparing render: matching {len(raw_events):,} replay inputs...", 5)

    if events:
        judgements, bad_judgements = build_judgements(notes, events, beatmap.keys, mirror, overall_difficulty, is_convert, score_v2)
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
                target_counts,
                score_v2,
            )
    else:
        best_offset, judgements, bad_judgements = 0, [], []
        simulated_accuracy = 100.0

    if judgements and not score_v2:
        # Judgement matching may apply an automatic timing correction. Stable LN
        # ticks, however, are driven by the original replay key state at each
        # beatmap tick, so keep them on the unshifted replay timeline.
        add_stable_ln_ticks(judgements, notes, raw_events, beatmap.keys, mirror)

    counts_reconciled = bool(judgements and target_counts and reconcile_judgements_with_replay(judgements, target_counts))
    bad_judgements = [
        j for j in judgements
        if j.get("counts_accuracy", True) and j["value"] < 300
    ][:100]
    simulated_accuracy = judgements[-1]["accuracy"] if judgements else 100.0
    ensure_not_cancelled(cancel_callback)
    report_prepare(4, "Preparing render: building overlays and timing cache...", 4)

    start_map_time = max(0, notes[0]["time"] - 2000)
    gameplay_end_time = max((note["end_time"] if note["end_time"] is not None else note["time"]) for note in notes)
    results_start_time = gameplay_end_time + 1000 if show_results_screen else gameplay_end_time + 2001
    end_map_time = (
        results_start_time + int(max(1.0, results_duration) * 1000)
        if show_results_screen
        else gameplay_end_time + 2000
    )
    replay_score = int(getattr(replay, "score", 0)) if replay else 0
    replay_max_combo = int(getattr(replay, "max_combo", 0)) if replay else 0
    replay_perfect = bool(getattr(replay, "perfect", False)) if replay else False
    final_counts = judgement_counts(judgements)
    rosu_star_rating, rosu_final_pp = rosu_mania_performance(osu_file, mod_settings["mods_int"], final_counts)
    if star_rating is None and rosu_star_rating is not None:
        star_rating = rosu_star_rating
    final_pp = rosu_final_pp if rosu_final_pp is not None else mania_pp_value(star_rating, final_counts)
    rank = replay_rank(simulated_accuracy, mod_settings["mods_int"])
    difficulty_profile = build_difficulty_profile(notes, start_map_time, gameplay_end_time) if show_strain_graph else []
    vignette_mask = create_vignette_mask(width, height, float(vignette_strength) / 100.0)
    report_prepare(5, "Preparing render: checking GPU and encoder...", 3)

    real_duration_ms = int((end_map_time - start_map_time) / speed_multiplier)
    total_frames = max(1, int(real_duration_ms / 1000 * fps))
    test_limit_seconds = 0.0
    try:
        test_limit_seconds = float(os.environ.get("MANIA_RENDERER_LIMIT_SECONDS", "0") or 0)
    except ValueError:
        test_limit_seconds = 0.0

    if test_limit_seconds > 0:
        limited_frames = max(1, int(test_limit_seconds * fps))
        if limited_frames < total_frames:
            total_frames = limited_frames
            end_map_time = start_map_time + int(total_frames * 1000 / fps * speed_multiplier)
            results_start_time = end_map_time + 1
            gameplay_end_time = min(gameplay_end_time, end_map_time)
    scroll_time_ms = mania_scroll_time_ms(scroll_speed_value)

    gpu_renderer = None
    gpu_error = None

    if gpu_compositing:
        from osu_mania_replay_renderer.gpu_compositor import detect_gpu_renderer, gpu_unavailable_reason

        gpu_renderer = detect_gpu_renderer()
        gpu_error = gpu_unavailable_reason() if gpu_renderer is None else None

    if gpu_compositing and os.environ.get("MANIA_RENDERER_REQUIRE_GPU") and gpu_renderer is None:
        message = "GPU frame compositing was required, but no OpenGL GPU context could be created."

        if gpu_error:
            message += f" Reason: {gpu_error}"

        raise RuntimeError(message)

    cpu_count = os.cpu_count() or 1

    if gpu_renderer:
        default_workers = max(1, min(cpu_count, 4))
        worker_env = os.environ.get("MANIA_RENDERER_GPU_WORKERS")
    else:
        default_workers = max(1, min(cpu_count, 8))
        worker_env = os.environ.get("MANIA_RENDERER_CPU_WORKERS")

    try:
        workers = int(worker_env) if worker_env else default_workers
    except ValueError:
        workers = default_workers

    workers = max(1, min(workers, cpu_count))
    megapixels = (width * height) / 1_000_000
    estimated_render_fps = max(4.0, workers * ((12.0 if gpu_renderer else 9.0) / max(1.0, megapixels)))
    estimated_seconds = total_frames / estimated_render_fps
    render_backend = f"GPU frame compositing: {gpu_renderer}" if gpu_renderer else "CPU frame compositing"

    if progress_callback:
        progress_callback(
            6,
            f"Preparing render: {total_frames} frames | {render_backend} | render ETA ~{format_duration(estimated_seconds)}"
        )

    temp_dir = Path(tempfile.mkdtemp(prefix=".mania-render-", dir=str(Path(output_file).parent)))
    segments_dir = temp_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    def ensure_active():
        try:
            ensure_not_cancelled(cancel_callback)
        except RenderCancelled:
            shutil.rmtree(temp_dir, ignore_errors=True)
            Path(output_file).unlink(missing_ok=True)
            Path(output_file).with_suffix(".debug.json").unlink(missing_ok=True)
            raise

    skin = load_mania_skin(skin_folder, beatmap.keys)
    ensure_active()
    report_prepare(7, f"Preparing render: loaded skin, starting renderer after {format_duration(time.time() - prepare_started)}", 1)
    results_frame = np.zeros((height, width, 3), dtype=np.uint8)
    draw_results_screen(
        results_frame,
        skin,
        f"{beatmap.artist} - {beatmap.title} [{beatmap.version}]",
        beatmap.creator,
        getattr(replay, "username", "Unknown") if replay else "Unknown",
        mod_settings["mods"],
        final_counts,
        simulated_accuracy,
        replay_max_combo,
        replay_score,
        final_pp,
        rank,
        replay_perfect,
        results_background,
    )
    results_frame.flags.writeable = False

    # Replay key bits already represent physical lanes after applying Mirror.
    # Mirroring them again would light the opposite key in the overlay.
    event_lanes = build_event_lanes(events, beatmap.keys)

    judgement_lookup = {
        (j["lane"], j["kind"], j["time"]): j
        for j in judgements
    }
    ln_hold_lanes = []

    for lane in range(beatmap.keys):
        lane_holds = []

        for note in notes:
            if note["end_time"] is None:
                continue

            render_lane = beatmap.keys - 1 - note["lane"] if mirror else note["lane"]

            if render_lane != lane:
                continue

            head = judgement_lookup.get((lane, "ln_head", note["time"]))

            if head is not None and head["value"] > 0:
                lane_holds.append((head.get("display_time", head["time"]), note["end_time"]))

        lane_holds.sort()
        ln_hold_lanes.append((
            [start for start, _ in lane_holds],
            [end for _, end in lane_holds],
        ))

    display_judgements = sorted(judgements, key=lambda j: j.get("display_time", j["time"]))
    cumulative_counts = []
    running_counts = {"300g": 0, "300": 0, "200": 0, "100": 0, "50": 0, "0": 0}
    combo_changed_times = []
    previous_combo = 0
    last_combo_change = None

    for judgement in display_judgements:
        display_time = judgement.get("display_time", judgement["time"])

        if judgement["combo"] > previous_combo:
            last_combo_change = display_time

        previous_combo = judgement["combo"]
        combo_changed_times.append(last_combo_change)

        if judgement.get("counts_accuracy", True):
            key = judgement.get("image_key") or hit_image_key(judgement["value"], judgement.get("diff"))
            running_counts[key] = running_counts.get(key, 0) + 1

        cumulative_counts.append(running_counts.copy())

    ctx = {
        "width": width,
        "height": height,
        "fps": fps,
        "output_fps": output_fps,
        "start_map_time": start_map_time,
        "end_map_time": end_map_time,
        "gameplay_end_time": gameplay_end_time,
        "results_start_time": results_start_time,
        "notes": notes,
        "keys": beatmap.keys,
        "title": f"{beatmap.artist} - {beatmap.title} [{beatmap.version}]",
        "mapper": beatmap.creator,
        "player": getattr(replay, "username", "Unknown") if replay else "Unknown",
        "mods": mod_settings["mods"],
        # output_dir remains available for direct single-frame tests; normal
        # renders stream each worker batch into a contiguous MJPEG segment.
        "output_dir": str(temp_dir),
        "segments_dir": str(segments_dir),
        "scroll_time_ms": scroll_time_ms,
        "scroll_speed_value": float(scroll_speed_value),
        "motion_blur": int(motion_blur),
        "speed_multiplier": speed_multiplier,
        "mirror": mirror,
        "score_v2": score_v2,
        "skin": skin,
        "judgements": judgements,
        "display_judgements": display_judgements,
        "judgement_times": [j.get("display_time", j["time"]) for j in display_judgements],
        "combo_changed_times": combo_changed_times,
        "cumulative_counts": cumulative_counts,
        "events": events,
        "event_lanes": event_lanes,
        "ln_hold_lanes": ln_hold_lanes,
        "note_times": [note["time"] for note in notes],
        "judgement_lookup": judgement_lookup,
        "star_rating": star_rating,
        "replay_accuracy": replay_accuracy,
        "final_accuracy": simulated_accuracy,
        "final_counts": final_counts,
        "final_pp": final_pp,
        "replay_score": replay_score,
        "replay_max_combo": replay_max_combo,
        "replay_perfect": replay_perfect,
        "replay_rank": rank,
        "difficulty_profile": difficulty_profile,
        "results_background": results_background,
        "results_frame": results_frame,
        "show_side_overlay": bool(show_side_overlay),
        "show_strain_graph": bool(show_strain_graph),
        "colour_combo_during_holds": bool(colour_combo_during_holds),
        "layout_positions": layout_positions or {},
        "gpu_compositing": bool(gpu_renderer),
        "vignette_mask": vignette_mask,
        "frame_jpeg_quality": frame_jpeg_quality,
        "frame_stream_mode": frame_stream_mode,
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
        judgement_counts_reconciled=counts_reconciled,
        simulated_judgement_counts=judgement_counts(judgements),
        missing_skin_elements=skin.get("missing_elements", []),
        visual_options={
            "show_side_overlay": bool(show_side_overlay),
            "show_strain_graph": bool(show_strain_graph),
            "vignette_strength": int(vignette_strength),
            "results_background_opacity": float(results_background_opacity),
            "results_duration": float(results_duration),
            "show_results_screen": bool(show_results_screen),
            "colour_combo_during_holds": bool(colour_combo_during_holds),
            "layout_positions": layout_positions or {},
            "gpu_compositing": bool(gpu_renderer),
            "gpu_renderer": gpu_renderer,
            "gpu_error": gpu_error,
            "fps": fps,
            "output_fps": output_fps,
            "test_limit_seconds": test_limit_seconds,
            "frame_jpeg_quality": frame_jpeg_quality,
            "frame_stream_mode": frame_stream_mode,
        },
    )

    fast_gpu_setting = os.environ.get("MANIA_RENDERER_FAST_GPU_ENGINE", "1").strip().lower()
    use_fast_gpu_engine = (
        fast_gpu_setting not in {"0", "false", "no", "off"}
        and bool(gpu_renderer)
        and int(motion_blur) <= 0
    )

    if use_fast_gpu_engine:
        if progress_callback:
            progress_callback(7, "Fast GPU engine: starting OpenGL render pipeline...")

        try:
            from osu_mania_replay_renderer.fast_gpu_renderer import render_fast_gpu

            render_fast_gpu(
                ctx,
                osu_file,
                beatmap,
                output_file,
                total_frames,
                start_map_time,
                total_frames / fps,
                nightcore_pitch,
                progress_callback,
                cancel_callback,
            )
            shutil.rmtree(temp_dir, ignore_errors=True)
            return
        except Exception as error:
            if os.environ.get("MANIA_RENDERER_REQUIRE_FAST_GPU"):
                shutil.rmtree(temp_dir, ignore_errors=True)
                Path(output_file).unlink(missing_ok=True)
                raise

            if progress_callback:
                progress_callback(0, f"Fast GPU engine failed ({error}); falling back to classic renderer...")

    start = time.time()
    done = 0

    if progress_callback:
        progress_callback(0, f"Frame: 0/{total_frames} | ETA: calculating...")

    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=init_worker,
        initargs=(ctx,),
    )

    try:
        next_frame = 0
        futures = set()

        try:
            configured_batch_size = int(os.environ.get("MANIA_RENDERER_BATCH_SIZE", "0"))
        except ValueError:
            configured_batch_size = 0

        if configured_batch_size > 0:
            batch_size = configured_batch_size
        else:
            batch_size = max(16, min(120, total_frames // max(1, workers * 4)))

        max_pending = workers * 2
        segments = []

        def submit_batch(start):
            stop = min(total_frames, start + batch_size)
            return executor.submit(frame_batch_worker, range(start, stop)), stop

        while next_frame < total_frames and len(futures) < max_pending:
            ensure_not_cancelled(cancel_callback)
            future, next_frame = submit_batch(next_frame)
            futures.add(future)

        while futures:
            ensure_not_cancelled(cancel_callback)
            completed, futures = wait(futures, timeout=0.2, return_when=FIRST_COMPLETED)

            if not completed:
                continue

            for future in completed:
                segment_start, segment_stop, segment_path = future.result()
                done += segment_stop - segment_start
                segments.append((segment_start, segment_path))

                if next_frame < total_frames:
                    ensure_not_cancelled(cancel_callback)
                    new_future, next_frame = submit_batch(next_frame)
                    futures.add(new_future)

                elapsed = time.time() - start
                render_fps = done / elapsed if elapsed > 0 else 0
                remaining = int((total_frames - done) / render_fps) if render_fps > 0 else 0

                if progress_callback:
                    progress_callback(
                        int(done / total_frames * 85),
                        f"Frame: {done}/{total_frames} | Render: {render_fps:.1f} fps | ETA: {remaining}s"
                    )
    except RenderCancelled:
        terminate_process_pool(executor, futures)
        shutil.rmtree(temp_dir, ignore_errors=True)
        Path(output_file).unlink(missing_ok=True)
        Path(output_file).with_suffix(".debug.json").unlink(missing_ok=True)
        raise
    except Exception:
        terminate_process_pool(executor, futures)
        shutil.rmtree(temp_dir, ignore_errors=True)
        Path(output_file).unlink(missing_ok=True)
        raise
    else:
        executor.shutdown(wait=True)

    frame_stream = temp_dir / ("frames.raw" if frame_stream_mode == "raw" else "frames.mjpg")
    ensure_active()

    if progress_callback:
        progress_callback(85, "Joining frame stream...")

    with open(frame_stream, "wb", buffering=4 * 1024 * 1024) as output_stream:
        for _, segment_path in sorted(segments):
            with open(segment_path, "rb") as segment_stream:
                shutil.copyfileobj(segment_stream, output_stream, length=4 * 1024 * 1024)

    shutil.rmtree(segments_dir)
    ensure_active()
    silent_video = temp_dir / "silent.mp4"

    encoder_cmd, encoder_attempts = encode_silent_video(
        fps,
        frame_stream,
        silent_video,
        progress_callback,
        frame_stream_mode=frame_stream_mode,
        width=width,
        height=height,
        output_fps=output_fps,
    )
    ensure_active()
    debug_path = Path(output_file).with_suffix(".debug.json")

    if debug_path.exists():
        try:
            with open(debug_path, "r", encoding="utf-8") as f:
                debug_data = json.load(f)

            debug_data["video_encoder"] = encoder_cmd
            debug_data["video_encoder_attempts"] = encoder_attempts

            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

    if progress_callback:
        encoder_name = encoder_cmd[encoder_cmd.index("-c:v") + 1]
        progress_callback(90, f"Encoding audio/video with {encoder_name}...")

    audio_path = Path(osu_file).parent / beatmap.audio_file

    if audio_path.exists():
        cmd = [
            ffmpeg_binary(), "-y",
            "-i", str(silent_video),
            "-ss", str(start_map_time / 1000),
            "-i", str(audio_path),
        ]

        cmd += make_audio_args(speed_multiplier, nightcore_pitch)

        cmd += [
            "-c:v", "copy",
            "-c:a", "aac",
            "-t", str(total_frames / fps),
            output_file,
        ]
    else:
        cmd = [ffmpeg_binary(), "-y", "-i", str(silent_video), "-c:v", "copy", output_file]

    subprocess.run(cmd, check=True)
    ensure_active()
    shutil.rmtree(temp_dir)

    if progress_callback:
        progress_callback(100, "Done")
