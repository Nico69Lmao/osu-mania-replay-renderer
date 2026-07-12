from bisect import bisect_left, bisect_right
from pathlib import Path
import os
import platform
import queue
import subprocess
import threading
import time

import cv2
import numpy as np


def _windows_gpu_names():
    if platform.system().lower() != "windows":
        return ""

    commands = [
        ["wmic", "path", "win32_VideoController", "get", "name"],
        ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
    ]

    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
        except Exception:
            continue

        text = f"{result.stdout}\n{result.stderr}".strip()
        if text:
            return text.lower()

    return ""


VERTEX_SHADER = """
#version 330
in vec2 in_position;
in vec3 in_colour;
out vec3 colour;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    colour = in_colour;
}
"""


FRAGMENT_SHADER = """
#version 330
in vec3 colour;
out vec4 frag_colour;

void main() {
    frag_colour = vec4(colour, 1.0);
}
"""

TEXTURE_VERTEX_SHADER = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 uv;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    uv = in_uv;
}
"""


TEXTURE_FRAGMENT_SHADER = """
#version 330
uniform sampler2D source;
in vec2 uv;
out vec4 colour;

void main() {
    colour = texture(source, uv);
}
"""


TEXT_CACHE = {}
LN_BODY_SOURCE_CACHE = {}
TINT_CACHE = {}
ALPHA_CACHE = {}


def _rgb_from_skin_colour(values, default=(0, 0, 0), default_alpha=255):
    if not values or len(values) < 3:
        return (*default, default_alpha)
    alpha = values[3] if len(values) > 3 else default_alpha
    return int(values[0]), int(values[1]), int(values[2]), int(alpha)


def _gl_colour_from_skin_colour(values, default=(0.0, 0.0, 0.0)):
    r, g, b, a = _rgb_from_skin_colour(values, tuple(int(c * 255) for c in default))
    alpha = max(0, min(255, a)) / 255.0
    return (r / 255.0 * alpha, g / 255.0 * alpha, b / 255.0 * alpha)


def _alpha_bbox(image, threshold=8):
    if image is None or image.ndim < 3 or image.shape[2] < 4:
        return None
    ys, xs = np.where(image[:, :, 3] > threshold)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _has_visible_alpha(image):
    if image is None:
        return False
    if image.ndim < 3 or image.shape[2] < 4:
        return True
    return bool(np.any(image[:, :, 3] > 8))


def _animated_frame(frames, map_time, start_time=0, fps=60):
    if not frames:
        return None

    fps = max(1, int(fps or 60))
    age = max(0, float(map_time) - float(start_time))
    index = int(age * fps / 1000.0) % len(frames)
    return frames[index]


def _animated_or_static(frames, static_image, map_time, start_time=0, fps=60):
    image = _animated_frame(frames, map_time, start_time, fps)
    return static_image if image is None else image


def _ln_body_source(image):
    if image is None:
        return None
    pointer = int(image.__array_interface__["data"][0])
    key = (pointer, image.shape, image.strides, "gpu-safe")
    cached = LN_BODY_SOURCE_CACHE.get(key)
    if cached is not None:
        return cached

    # Some mania skins ship absurdly tall LN body strips (Nico's is ~40000px).
    # Uploading those whole strips to OpenGL can fail or murder performance.
    # Keep a single reusable slice and let the GPU stretch/tile it per-frame.
    #
    # LN body textures are repeated/stretched as the visible body strip.
    # Transparent vertical padding in NoteImage#L is common (Span's noteL1
    # starts with ~60 transparent rows) and must not become a real gap between
    # cap and body.  Do not crop horizontal padding though: that padding is
    # part of the skin's intended width/proportion once scaled to ColumnWidth.
    bbox = _alpha_bbox(image)
    if bbox:
        _, y1, _, y2 = bbox
        source = image[y1:y2, :]
        if source.size == 0:
            source = image
    else:
        source = image
    max_safe_height = int(os.environ.get("MANIA_RENDERER_LN_BODY_TEXTURE_HEIGHT", "2048"))
    source = source[:max(1, min(source.shape[0], max_safe_height))]
    if not source.flags["C_CONTIGUOUS"]:
        source = np.ascontiguousarray(source)
    source.flags.writeable = False
    LN_BODY_SOURCE_CACHE[key] = source
    return source


def _visible_source(image):
    if image is None:
        return None
    pointer = int(image.__array_interface__["data"][0])
    key = (pointer, image.shape, image.strides, "visible")
    cached = LN_BODY_SOURCE_CACHE.get(key)
    if cached is not None:
        return cached
    bbox = _alpha_bbox(image)
    if not bbox:
        return image
    x1, y1, x2, y2 = bbox
    source = image[y1:y2, x1:x2]
    if source.size == 0:
        source = image
    if not source.flags["C_CONTIGUOUS"]:
        source = np.ascontiguousarray(source)
    source.flags.writeable = False
    LN_BODY_SOURCE_CACHE[key] = source
    return source


def _receptor_source(image):
    if image is None:
        return None
    bbox = _alpha_bbox(image)
    if not bbox:
        return image
    _, y1, _, y2 = bbox
    visible_h = max(1, y2 - y1)
    full_h = max(1, image.shape[0])
    # KeyImage# uses Origin: Bottom, but many mania skins store receptors in a
    # very tall transparent canvas only to position the visible ring/circle.
    # Scaling that whole canvas makes receptors gigantic or misaligned.  If the
    # visible alpha is much smaller than the canvas, scale/draw from the visible
    # alpha bbox; otherwise keep the full canvas (Span-like skins rely on it).
    if visible_h / full_h < 0.65:
        return _visible_source(image)
    return image


def _receptor_wants_visible_square(image):
    if image is None:
        return False
    bbox = _alpha_bbox(image)
    if not bbox:
        return False
    _, y1, _, y2 = bbox
    visible_h = max(1, y2 - y1)
    full_h = max(1, image.shape[0])
    return visible_h / full_h < 0.65


def _tinted_image(image, tint_rgb):
    if image is None or tint_rgb is None:
        return image
    pointer = int(image.__array_interface__["data"][0])
    key = (pointer, image.shape, image.strides, tuple(int(v) for v in tint_rgb))
    cached = TINT_CACHE.get(key)
    if cached is not None:
        return cached
    tinted = image.copy()
    if tinted.ndim == 3 and tinted.shape[2] >= 3:
        target = np.array((tint_rgb[2], tint_rgb[1], tint_rgb[0]), dtype=np.float32)
        if tinted.shape[2] == 4:
            visible = tinted[:, :, 3] > 0
            tinted[:, :, :3][visible] = target.astype(np.uint8)
        else:
            tinted[:, :, :3] = target.astype(np.uint8)
    tinted.flags.writeable = False
    TINT_CACHE[key] = tinted
    return tinted


def _alpha_scaled_image(image, opacity):
    if image is None or opacity >= 0.995:
        return image
    opacity = max(0.0, min(1.0, float(opacity)))
    pointer = int(image.__array_interface__["data"][0])
    bucket = int(round(opacity * 16))
    key = (pointer, image.shape, image.strides, bucket)
    cached = ALPHA_CACHE.get(key)
    if cached is not None:
        return cached
    faded = image.copy()
    if faded.ndim == 3 and faded.shape[2] >= 4:
        faded[:, :, 3] = np.clip(faded[:, :, 3].astype(np.float32) * (bucket / 16.0), 0, 255).astype(np.uint8)
    faded.flags.writeable = False
    ALPHA_CACHE[key] = faded
    return faded


def _rect(out, row, x1, y1, x2, y2, width, height, colour):
    left = x1 / width * 2.0 - 1.0
    right = x2 / width * 2.0 - 1.0
    top = y1 / height * 2.0 - 1.0
    bottom = y2 / height * 2.0 - 1.0
    r, g, b = colour
    out[row:row + 6] = (
        (left, top, r, g, b),
        (right, top, r, g, b),
        (right, bottom, r, g, b),
        (left, top, r, g, b),
        (right, bottom, r, g, b),
        (left, bottom, r, g, b),
    )
    return row + 6


def _draw_text(frame, text, x, y, scale=0.7, colour=(245, 245, 245), thickness=1, anchor="left"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(text)
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    if anchor == "right":
        x -= tw
    elif anchor == "center":
        x -= tw // 2

    if frame.ndim == 3 and frame.shape[2] == 4:
        shadow = (0, 0, 0, 230)
        text_colour = (int(colour[0]), int(colour[1]), int(colour[2]), 255)
    else:
        shadow = (0, 0, 0)
        text_colour = colour

    cv2.putText(frame, text, (int(x) + 2, int(y) + 2), font, scale, shadow, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (int(x), int(y)), font, scale, text_colour, thickness, cv2.LINE_AA)


def _text_sprite(text, scale=1.0, colour=(255, 255, 255), thickness=2):
    key = (str(text), round(float(scale), 3), tuple(colour), int(thickness))
    cached = TEXT_CACHE.get(key)
    if cached is not None:
        return cached

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(text)
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = max(6, thickness * 4)
    sprite = np.zeros((th + baseline + pad * 2, tw + pad * 2, 4), dtype=np.uint8)
    origin = (pad, pad + th)
    cv2.putText(sprite, text, (origin[0] + 2, origin[1] + 2), font, scale, (0, 0, 0, 220), thickness + 2, cv2.LINE_AA)
    cv2.putText(sprite, text, origin, font, scale, (*colour, 255), thickness, cv2.LINE_AA)
    sprite.flags.writeable = False
    TEXT_CACHE[key] = sprite
    return sprite


def _key_label_sprite(text, scale=0.42, colour=(246, 246, 238)):
    key = ("key-label", str(text), round(float(scale), 3), tuple(colour))
    cached = TEXT_CACHE.get(key)
    if cached is not None:
        return cached

    font = cv2.FONT_HERSHEY_DUPLEX
    text = str(text)
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    sprite = np.zeros((th + baseline + pad * 2, tw + pad * 2, 4), dtype=np.uint8)
    origin = (pad, pad + th)
    cv2.putText(sprite, text, (origin[0] + 1, origin[1] + 1), font, scale, (0, 0, 0, 190), thickness + 1, cv2.LINE_AA)
    cv2.putText(sprite, text, origin, font, scale, (*colour, 255), thickness, cv2.LINE_AA)
    sprite.flags.writeable = False
    TEXT_CACHE[key] = sprite
    return sprite


class TexturePainter:
    def __init__(self, ctx, moderngl, width, height):
        self.ctx = ctx
        self.moderngl = moderngl
        self.width = width
        self.height = height
        self.program = ctx.program(vertex_shader=TEXTURE_VERTEX_SHADER, fragment_shader=TEXTURE_FRAGMENT_SHADER)
        self.program["source"].value = 0
        self.buffer = ctx.buffer(reserve=16 * 4)
        self.vao = ctx.vertex_array(self.program, [(self.buffer, "2f 2f", "in_position", "in_uv")])
        self.vertex_data = np.empty(16, dtype="f4")
        self.batch_buffer = ctx.buffer(reserve=6 * 4 * 4 * 4096)
        self.batch_vao = ctx.vertex_array(self.program, [(self.batch_buffer, "2f 2f", "in_position", "in_uv")])
        self.texture_cache = {}
        self.cv_texture_cache = {}
        self.named_textures = {}

    def texture_for(self, image):
        pointer = int(image.__array_interface__["data"][0])
        key = (pointer, image.shape, image.strides)
        cached = self.texture_cache.get(key)
        if cached is not None:
            return cached
        contiguous = np.ascontiguousarray(image)
        texture = self.ctx.texture((contiguous.shape[1], contiguous.shape[0]), contiguous.shape[2], contiguous, alignment=1)
        texture.filter = (self.moderngl.LINEAR, self.moderngl.LINEAR)
        self.texture_cache[key] = texture
        return texture

    def update_named(self, name, image):
        contiguous = np.ascontiguousarray(image)
        texture = self.named_textures.get(name)
        size = (contiguous.shape[1], contiguous.shape[0])
        components = contiguous.shape[2]
        if texture is None or texture.size != size or texture.components != components:
            if texture is not None:
                texture.release()
            texture = self.ctx.texture(size, components, contiguous, alignment=1)
            texture.filter = (self.moderngl.LINEAR, self.moderngl.LINEAR)
            self.named_textures[name] = texture
        else:
            texture.write(contiguous, alignment=1)
        return texture

    def draw_texture(self, texture, x, y, w=None, h=None, anchor="left"):
        w = int(w if w is not None else texture.size[0])
        h = int(h if h is not None else texture.size[1])
        if anchor == "right":
            x -= w
        elif anchor == "center":
            x -= w // 2
        x = int(x)
        y = int(y)
        if x >= self.width or y >= self.height or x + w <= 0 or y + h <= 0:
            return

        left = x / self.width * 2.0 - 1.0
        right = (x + w) / self.width * 2.0 - 1.0
        top = y / self.height * 2.0 - 1.0
        bottom = (y + h) / self.height * 2.0 - 1.0
        self.vertex_data[:] = (
            left, top, 0.0, 0.0,
            right, top, 1.0, 0.0,
            left, bottom, 0.0, 1.0,
            right, bottom, 1.0, 1.0,
        )
        self.buffer.write(self.vertex_data)
        texture.use(0)
        self.vao.render(self.moderngl.TRIANGLE_STRIP)

    def draw_texture_clipped(self, texture, x, y, w, h):
        w = int(w)
        h = int(h)
        x = int(x)
        y = int(y)
        if w <= 0 or h <= 0 or x >= self.width or y >= self.height or x + w <= 0 or y + h <= 0:
            return

        draw_x1 = max(0, x)
        draw_y1 = max(0, y)
        draw_x2 = min(self.width, x + w)
        draw_y2 = min(self.height, y + h)

        u1 = (draw_x1 - x) / w
        v1 = (draw_y1 - y) / h
        u2 = (draw_x2 - x) / w
        v2 = (draw_y2 - y) / h

        left = draw_x1 / self.width * 2.0 - 1.0
        right = draw_x2 / self.width * 2.0 - 1.0
        top = draw_y1 / self.height * 2.0 - 1.0
        bottom = draw_y2 / self.height * 2.0 - 1.0
        self.vertex_data[:] = (
            left, top, u1, v1,
            right, top, u2, v1,
            left, bottom, u1, v2,
            right, bottom, u2, v2,
        )
        self.buffer.write(self.vertex_data)
        texture.use(0)
        self.vao.render(self.moderngl.TRIANGLE_STRIP)

    def draw_texture_repeat_y(self, texture, x, y, w, h, unit_h, anchor_bottom=False):
        w = int(w)
        h = int(h)
        unit_h = max(1, float(unit_h))
        x = int(x)
        y = int(y)
        if w <= 0 or h <= 0 or x >= self.width or y >= self.height or x + w <= 0 or y + h <= 0:
            return
        repeats = h / unit_h
        if anchor_bottom:
            v1 = 1.0 - repeats
            v2 = 1.0
        else:
            v1 = 0.0
            v2 = repeats
        try:
            texture.repeat_x = False
            texture.repeat_y = True
        except Exception:
            pass
        left = x / self.width * 2.0 - 1.0
        right = (x + w) / self.width * 2.0 - 1.0
        top = y / self.height * 2.0 - 1.0
        bottom = (y + h) / self.height * 2.0 - 1.0
        self.vertex_data[:] = (
            left, top, 0.0, v1,
            right, top, 1.0, v1,
            left, bottom, 0.0, v2,
            right, bottom, 1.0, v2,
        )
        self.buffer.write(self.vertex_data)
        texture.use(0)
        self.vao.render(self.moderngl.TRIANGLE_STRIP)

    def draw_cv_repeat_y(self, image, x, y, w, h, unit_h, anchor_bottom=False):
        if image is None:
            return
        self.draw_texture_repeat_y(self.cv_texture_for(image), x, y, w, h, unit_h, anchor_bottom)

    def draw_texture_crop(self, texture, x, y, crop_w, crop_h):
        crop_w = int(crop_w)
        crop_h = int(crop_h)
        if crop_w <= 0 or crop_h <= 0:
            return
        crop_w = min(crop_w, texture.size[0])
        crop_h = min(crop_h, texture.size[1])
        x = int(x)
        y = int(y)
        if x >= self.width or y >= self.height or x + crop_w <= 0 or y + crop_h <= 0:
            return
        left = x / self.width * 2.0 - 1.0
        right = (x + crop_w) / self.width * 2.0 - 1.0
        top = y / self.height * 2.0 - 1.0
        bottom = (y + crop_h) / self.height * 2.0 - 1.0
        u2 = crop_w / max(1, texture.size[0])
        v2 = crop_h / max(1, texture.size[1])
        self.vertex_data[:] = (
            left, top, 0.0, 0.0,
            right, top, u2, 0.0,
            left, bottom, 0.0, v2,
            right, bottom, u2, v2,
        )
        self.buffer.write(self.vertex_data)
        texture.use(0)
        self.vao.render(self.moderngl.TRIANGLE_STRIP)

    def draw_cv_crop(self, image, x, y, crop_w, crop_h):
        if image is None:
            return
        self.draw_texture_crop(self.cv_texture_for(image), x, y, crop_w, crop_h)

    def draw_texture_batch(self, texture, rects):
        if not rects:
            return
        vertices = np.empty((len(rects) * 6, 4), dtype="f4")
        out = 0
        for x, y, w, h in rects:
            w = int(w)
            h = int(h)
            if w <= 0 or h <= 0:
                continue
            x = int(x)
            y = int(y)
            if x >= self.width or y >= self.height or x + w <= 0 or y + h <= 0:
                continue
            left = x / self.width * 2.0 - 1.0
            right = (x + w) / self.width * 2.0 - 1.0
            top = y / self.height * 2.0 - 1.0
            bottom = (y + h) / self.height * 2.0 - 1.0
            vertices[out:out + 6] = (
                (left, top, 0.0, 0.0),
                (right, top, 1.0, 0.0),
                (left, bottom, 0.0, 1.0),
                (right, top, 1.0, 0.0),
                (right, bottom, 1.0, 1.0),
                (left, bottom, 0.0, 1.0),
            )
            out += 6
        if out == 0:
            return
        data = vertices[:out]
        self.batch_buffer.orphan(data.nbytes)
        self.batch_buffer.write(data)
        texture.use(0)
        self.batch_vao.render(self.moderngl.TRIANGLES, vertices=out)

    def draw_cv_rects(self, rects, preserve_order=False):
        if not rects:
            return
        if preserve_order:
            for image, x, y, w, h in rects:
                if image is not None:
                    self.draw_cv_image(image, x, y, w, h)
            return
        groups = {}
        for image, x, y, w, h in rects:
            if image is None:
                continue
            texture = self.cv_texture_for(image)
            key = id(texture)
            if key not in groups:
                groups[key] = (texture, [])
            groups[key][1].append((x, y, w, h))
        for texture, texture_rects in groups.values():
            self.draw_texture_batch(texture, texture_rects)

    def draw_image(self, image, x, y, anchor="left"):
        self.draw_texture(self.texture_for(image), x, y, anchor=anchor)

    def cv_texture_for(self, image):
        pointer = int(image.__array_interface__["data"][0])
        key = (pointer, image.shape, image.strides)
        cached = self.cv_texture_cache.get(key)
        if cached is not None:
            return cached
        contiguous = np.ascontiguousarray(image)
        if contiguous.ndim == 3 and contiguous.shape[2] == 4:
            converted = cv2.cvtColor(contiguous, cv2.COLOR_BGRA2RGBA)
            transparent = converted[:, :, 3] <= 8
            if np.any(transparent):
                converted = converted.copy()
                converted[transparent] = 0
        elif contiguous.ndim == 3 and contiguous.shape[2] == 3:
            converted = cv2.cvtColor(contiguous, cv2.COLOR_BGR2RGB)
        else:
            converted = contiguous
        texture = self.ctx.texture((converted.shape[1], converted.shape[0]), converted.shape[2], converted, alignment=1)
        texture.filter = (self.moderngl.LINEAR, self.moderngl.LINEAR)
        self.cv_texture_cache[key] = texture
        return texture

    def draw_cv_image(self, image, x, y, w=None, h=None, anchor="left"):
        if image is None:
            return
        texture = self.cv_texture_for(image)
        self.draw_texture(texture, x, y, w=w, h=h, anchor=anchor)

    def draw_cv_clipped(self, image, x, y, w, h):
        if image is None:
            return
        self.draw_texture_clipped(self.cv_texture_for(image), x, y, w, h)

    def draw_cv_centered(self, image, cx, cy, scale=1.0, max_width=None):
        if image is None:
            return
        h, w = image.shape[:2]
        tw = max(1, int(w * scale))
        th = max(1, int(h * scale))
        if max_width and tw > max_width:
            ratio = max_width / tw
            tw = max(1, int(tw * ratio))
            th = max(1, int(th * ratio))
        self.draw_cv_image(image, int(cx - tw / 2), int(cy - th / 2), tw, th)

    def draw_cv_bottom_centered(self, image, cx, bottom_y, scale=1.0, max_width=None):
        if image is None:
            return
        h, w = image.shape[:2]
        tw = max(1, int(w * scale))
        th = max(1, int(h * scale))
        if max_width and tw > max_width:
            ratio = max_width / tw
            tw = max(1, int(tw * ratio))
            th = max(1, int(th * ratio))
        self.draw_cv_image(image, int(cx - tw / 2), int(bottom_y - th), tw, th)

    def draw_skin_text(self, text, glyphs, center_x, y, overlap, coordinate_scale, frame_height, vertical_anchor="top", tint=None):
        from osu_mania_replay_renderer.renderer import select_skin_glyph

        selected = []
        for character in str(text):
            image, density = select_skin_glyph(glyphs.get(character, {}), frame_height)
            if image is None:
                return False
            scale = coordinate_scale / density
            selected.append((image, max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
        if not selected:
            return False
        overlap_px = int(overlap * coordinate_scale)
        total_width = sum(w for _, w, _ in selected) - overlap_px * max(0, len(selected) - 1)
        x = int(center_x - total_width / 2)
        max_h = max(h for _, _, h in selected)
        top_y = int(y - max_h / 2) if vertical_anchor == "center" else int(y)
        for image, w, h in selected:
            draw_image = _tinted_image(image, tint)
            glyph_y = top_y + (max_h - h) // 2
            self.draw_cv_image(draw_image, x, glyph_y, w, h)
            x += w - overlap_px
        return True


def _blit_sprite(frame, sprite, x, y, anchor="center"):
    h, w = sprite.shape[:2]
    if anchor == "right":
        x -= w
    elif anchor == "center":
        x -= w // 2
    x = int(x)
    y = int(y)
    fh, fw = frame.shape[:2]
    if x >= fw or y >= fh or x + w <= 0 or y + h <= 0:
        return
    sx1 = max(0, -x)
    sy1 = max(0, -y)
    sx2 = min(w, fw - x)
    sy2 = min(h, fh - y)
    roi = frame[y + sy1:y + sy2, x + sx1:x + sx2]
    spr = sprite[sy1:sy2, sx1:sx2]
    cv2.copyTo(spr[:, :, :3], spr[:, :, 3], roi)


def _pressed_at(event_lanes, lane, map_time):
    times, states = event_lanes[lane]
    i = bisect_right(times, map_time) - 1
    return bool(states[i]) if i >= 0 else False


def _first_release_after(event_lanes, lane, start_time, max_time=None):
    if lane < 0 or lane >= len(event_lanes):
        return None

    times, states = event_lanes[lane]
    i = bisect_left(times, start_time)

    while i < len(times):
        event_time = times[i]

        if max_time is not None and event_time > max_time:
            return None

        if not states[i]:
            return event_time

        i += 1

    return None


def _start_writer(ffmpeg, buffer_size, count):
    free_buffers = queue.Queue(maxsize=count)
    filled_buffers = queue.Queue(maxsize=count)
    for _ in range(count):
        free_buffers.put(bytearray(buffer_size))

    def writer():
        while True:
            item = filled_buffers.get()
            if item is None:
                break
            ffmpeg.stdin.write(item)
            free_buffers.put(item)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    return free_buffers, filled_buffers, thread


def _copy_bgr_to_rgb_bytes(frame_bgr, target):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    target[:] = memoryview(np.ascontiguousarray(rgb))


def render_fast_gpu(ctx, osu_file, beatmap, output_file, total_frames, audio_start_ms, audio_duration_s, nightcore_pitch, progress_callback=None, cancel_callback=None):
    import moderngl
    from osu_mania_replay_renderer.renderer import (
        ffmpeg_binary,
        ffmpeg_encoder_names,
        make_audio_args,
        mania_pp_value,
        mania_scroll_time_ms,
        format_clock,
        stable_key_bpm,
        vaapi_device,
    )

    width = ctx["width"]
    height = ctx["height"]
    fps = ctx["fps"]
    start_map_time = ctx["start_map_time"]
    gameplay_end_time = ctx["gameplay_end_time"]
    results_start_time = ctx["results_start_time"]
    speed_multiplier = ctx["speed_multiplier"]
    keys = ctx["keys"]
    notes = ctx["notes"]
    note_times = ctx["note_times"]
    judgements = ctx["display_judgements"]
    judgement_times = ctx["judgement_times"]
    cumulative_counts = ctx["cumulative_counts"]
    combo_changed_times = ctx["combo_changed_times"]
    event_lanes = ctx["event_lanes"]
    mirror = ctx["mirror"]
    star_rating = ctx["star_rating"]
    title = ctx["title"]
    mapper = ctx["mapper"]
    player = ctx["player"]
    mods = ctx["mods"]
    difficulty_profile = ctx["difficulty_profile"]
    results_frame = ctx["results_frame"]
    judgement_lookup = ctx["judgement_lookup"]
    ln_hold_lanes = ctx.get("ln_hold_lanes", [])
    colour_combo_during_holds = ctx.get("colour_combo_during_holds", True)
    skin = ctx["skin"]
    cfg = skin["cfg"]
    scroll_time = mania_scroll_time_ms(float(ctx.get("scroll_speed_value", os.environ.get("MANIA_RENDERER_FAST_SCROLL_SPEED", "30")))) * speed_multiplier

    device = vaapi_device()
    encoders = ffmpeg_encoder_names()

    output_path = Path(output_file)
    output_path.unlink(missing_ok=True)
    silent_video = output_path.with_suffix(".fast-gpu-silent.mp4")
    silent_video.unlink(missing_ok=True)

    qp = os.environ.get("MANIA_RENDERER_VAAPI_QP", "32")
    requested_encoder = os.environ.get("MANIA_RENDERER_FAST_ENCODER", "").strip()
    system_name = platform.system().lower()
    gpu_names = _windows_gpu_names()

    if requested_encoder:
        encoder = requested_encoder
    elif system_name == "windows" and any(token in gpu_names for token in ("amd", "radeon")) and "h264_amf" in encoders:
        encoder = "h264_amf"
    elif system_name == "windows" and "nvidia" in gpu_names and "h264_nvenc" in encoders:
        encoder = "h264_nvenc"
    elif system_name == "windows":
        encoder = "libx264"
    elif device and "h264_vaapi" in encoders:
        encoder = "h264_vaapi"
    elif "h264_nvenc" in encoders:
        encoder = "h264_nvenc"
    elif "h264_amf" in encoders:
        encoder = "h264_amf"
    else:
        encoder = "libx264"

    if progress_callback:
        gpu_label = gpu_names.replace("\r", " ").replace("\n", " ").strip() if gpu_names else "unknown GPU"
        progress_callback(7, f"Fast GPU engine: encoder {encoder}, preparing OpenGL ({gpu_label})...")

    if encoder == "h264_vaapi":
        if device is None:
            raise RuntimeError("h264_vaapi requested, but no VAAPI render device was found.")
        codec_args = [
            "-vaapi_device", device,
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi",
            "-qp", qp,
        ]
        vaapi_async_depth = os.environ.get("MANIA_RENDERER_VAAPI_ASYNC_DEPTH")
        vaapi_quality = os.environ.get("MANIA_RENDERER_VAAPI_QUALITY")
        if vaapi_async_depth:
            codec_args.extend(["-async_depth", vaapi_async_depth])
        if vaapi_quality:
            codec_args.extend(["-quality", vaapi_quality])
    elif encoder == "h264_nvenc":
        codec_args = [
            "-vf", "format=yuv420p",
            "-c:v", "h264_nvenc",
            "-preset", os.environ.get("MANIA_RENDERER_NVENC_PRESET", "p1"),
            "-tune", "ull",
            "-cq", os.environ.get("MANIA_RENDERER_NVENC_CQ", "28"),
        ]
    elif encoder == "h264_amf":
        codec_args = [
            "-vf", "format=yuv420p",
            "-c:v", "h264_amf",
            "-quality", os.environ.get("MANIA_RENDERER_AMF_QUALITY", "speed"),
            "-qp_i", os.environ.get("MANIA_RENDERER_AMF_QP", "28"),
            "-qp_p", os.environ.get("MANIA_RENDERER_AMF_QP", "28"),
        ]
    else:
        codec_args = [
            "-vf", "format=yuv420p",
            "-c:v", "libx264",
            "-preset", os.environ.get("MANIA_RENDERER_X264_PRESET", "ultrafast"),
            "-crf", os.environ.get("MANIA_RENDERER_X264_CRF", "28"),
        ]

    ffmpeg = subprocess.Popen(
        [
            ffmpeg_binary(), "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s:v", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "-",
            *codec_args,
            str(silent_video),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if ffmpeg.poll() is not None:
        raise RuntimeError("FFmpeg exited before the fast GPU renderer could send frames.")

    frame_size = width * height * 3
    default_pipe_buffers = 32 if frame_size <= 8_000_000 else 12
    free_buffers, filled_buffers, writer_thread = _start_writer(
        ffmpeg,
        frame_size,
        int(os.environ.get("MANIA_RENDERER_FAST_PIPE_BUFFERS", str(default_pipe_buffers))),
    )

    default_gl_backend = "egl" if platform.system().lower() == "linux" else ""
    gl_backend = os.environ.get("MANIA_RENDERER_GL_BACKEND", default_gl_backend).strip()
    context_attempts = [{"backend": gl_backend, "require": 330}] if gl_backend else [{"require": 330}]
    if platform.system().lower() == "windows":
        context_attempts.extend((
            {"backend": "wgl", "require": 330},
            {"backend": "standalone", "require": 330},
            {"require": 330},
        ))

    ctx_gl = None
    context_errors = []
    for attempt in context_attempts:
        try:
            ctx_gl = moderngl.create_standalone_context(**attempt)
            break
        except Exception as error:
            context_errors.append(f"{attempt}: {error}")

    if ctx_gl is None:
        raise RuntimeError("Could not create a fast OpenGL context. " + " | ".join(context_errors))

    if progress_callback:
        progress_callback(7, f"Fast GPU engine: OpenGL context ready, rendering {total_frames} frames with {encoder}...")
    program = ctx_gl.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    buffer = ctx_gl.buffer(reserve=max(8 * 1024 * 1024, (len(notes) + keys * 4 + 16) * 6 * 5 * 4))
    vao = ctx_gl.vertex_array(program, [(buffer, "2f 3f", "in_position", "in_colour")])
    texture_painter = TexturePainter(ctx_gl, moderngl, width, height)
    texture = ctx_gl.texture((width, height), 3, dtype="f1")
    framebuffer = ctx_gl.framebuffer(color_attachments=[texture])
    framebuffer.use()
    ctx_gl.viewport = (0, 0, width, height)
    geometry = np.empty((max(4096, (len(notes) + keys * 4 + 16) * 6), 5), dtype="f4")

    skin_scale = height / 480
    column_widths = cfg["column_widths"] or [70] * keys
    column_spacing = cfg["column_spacing"] or [0] * (keys - 1)
    column_widths = [int(w * skin_scale) for w in column_widths]
    column_spacing = [int(s * skin_scale) for s in column_spacing]
    play_width = sum(column_widths) + sum(column_spacing)
    play_x = int(cfg["column_start"] * skin_scale) if cfg["column_start"] is not None else (width - play_width) // 2
    raw_hit_position = cfg["hit_position"] if cfg["hit_position"] is not None else 402
    # skin.ini owns the judgement line. HitPosition is also the position used
    # by StageHint, LightingN and LightingL, so do not clamp or reinterpret it.
    hit_position = int(raw_hit_position)
    judge_y = int(hit_position * skin_scale)
    top_y = 0
    lane_xs = []
    cur = play_x
    for lane in range(keys):
        lane_xs.append(cur)
        cur += column_widths[lane]
        if lane < len(column_spacing):
            cur += column_spacing[lane]
    note_widths = list(column_widths)
    height_scale_width = cfg.get("width_for_note_height_scale")
    if height_scale_width is not None and height_scale_width > 0:
        height_scale_width = int(height_scale_width * skin_scale)
    else:
        height_scale_width = min(column_widths) if column_widths else 1

    def mania_element_size(image, lane, target_width=None):
        target_w = max(1, int(target_width if target_width is not None else column_widths[lane]))
        if image is None:
            return target_w, target_w
        target_h = max(1, int(image.shape[0] * height_scale_width / max(1, image.shape[1])))
        return target_w, target_h

    def mania_receptor_size(source_image, original_image, lane):
        target_w = max(1, int(column_widths[lane]))
        if source_image is None:
            return target_w, target_w
        target_h = max(1, int(source_image.shape[0] * height_scale_width / max(1, source_image.shape[1])))
        # Some stable skins place the visible receptor in a very tall
        # transparent KeyImage canvas.  After alpha-cropping that canvas, the
        # visible ring still often contains glow/outline rows that make a
        # circle look vertically stretched if we blindly preserve the bbox
        # aspect ratio.  osu!mania aligns the KeyImage bottom to HitPosition;
        # for these cropped receptor canvases the playable receptor should be
        # roughly one column tall.
        if _receptor_wants_visible_square(original_image):
            target_h = min(target_h, target_w)
        return target_w, target_h

    visible_margin = 500
    lane_colours = [
        _gl_colour_from_skin_colour(cfg.get("colours", {}).get(f"Colour{lane + 1}"), (0.0, 0.0, 0.0))
        for lane in range(keys)
    ]
    note_colours = [(0.30, 0.80, 1.00), (1.00, 0.45, 0.70), (0.50, 1.00, 0.45), (1.00, 0.85, 0.25)]
    receptor_layouts = []
    receptor_images = []
    receptor_down_images = []
    for lane in range(keys):
        # KeyImage# / KeyImage#D use Origin: Bottom and are scaled from the
        # complete image canvas.  Do not trim transparent padding here: skins
        # use that padding to place/scale the receptor exactly like stable.
        receptor_img = _receptor_source(skin["keys"][lane]) if lane < len(skin.get("keys", [])) else None
        receptor_down_img = _receptor_source(skin["keys_down"][lane]) if lane < len(skin.get("keys_down", [])) else None
        receptor_images.append(receptor_img)
        receptor_down_images.append(receptor_down_img)
        original_receptor_img = skin["keys"][lane] if lane < len(skin.get("keys", [])) else None
        receptor_w, receptor_h = mania_receptor_size(receptor_img, original_receptor_img, lane)
        receptor_layouts.append((receptor_w, receptor_h, judge_y))

    fast_notes = []
    for note in notes:
        note_time = note["time"]
        end_time = note["end_time"] if note["end_time"] is not None else note_time
        lane = keys - 1 - note["lane"] if mirror else note["lane"]
        tap = judgement_lookup.get((lane, "tap", note_time))
        head = judgement_lookup.get((lane, "ln_head", note_time))
        tail = judgement_lookup.get((lane, "ln_tail", end_time))
        head_display_time = head.get("display_time", head["time"]) if head else note_time
        head_hit = bool(head and head.get("value", 0) > 0)
        ln_miss_display_time = head_display_time if note["end_time"] is not None and head is not None and not head_hit else None
        tail_display_time = tail.get("display_time", tail["time"]) if tail else None
        release_cutoff_time = None
        if note["end_time"] is not None:
            # Stable removes the held LN visual as soon as the key is released.
            # If we keep drawing until the note tail scrolls out, skins with
            # large tail caps leave a half-cap "ghost" on the receptor after
            # release.  Use the real replay release event as the visual cutoff;
            # fall back to the judgement display time for misses/no-release.
            release_cutoff_time = _first_release_after(event_lanes, lane, head_display_time)
            if tail_display_time is not None:
                release_cutoff_time = (
                    min(release_cutoff_time, tail_display_time)
                    if release_cutoff_time is not None
                    else tail_display_time
                )
        fast_notes.append((
            note_time,
            end_time,
            lane,
            note["end_time"] is not None,
            tap.get("display_time", tap["time"]) if tap else None,
            head_display_time,
            head_hit,
            tail_display_time,
            ln_miss_display_time,
            release_cutoff_time,
        ))

    overlay_cache = {"static": None, "stats": None, "last_stats_key": None, "key_rects": []}
    draw_overlays = os.environ.get("MANIA_RENDERER_FAST_OVERLAY", "1").strip().lower() not in {"0", "false", "no", "off"}
    results_rgb = bytearray(frame_size)
    _copy_bgr_to_rgb_bytes(results_frame, results_rgb)
    readback_depth = int(os.environ.get("MANIA_RENDERER_READBACK_DEPTH", "3"))
    async_readback = (
        readback_depth > 1
        and os.environ.get("MANIA_RENDERER_ASYNC_READBACK", "1").strip().lower() not in {"0", "false", "no", "off"}
    )
    readback_pool = [ctx_gl.buffer(reserve=frame_size) for _ in range(readback_depth)] if async_readback else []
    pending_readbacks = []

    def flush_one_readback():
        pbo = pending_readbacks.pop(0)
        cpu_buffer = free_buffers.get()
        pbo.read_into(cpu_buffer)
        filled_buffers.put(cpu_buffer)
        readback_pool.append(pbo)

    def flush_all_readbacks():
        while pending_readbacks:
            flush_one_readback()

    def emit_framebuffer():
        if async_readback:
            if not readback_pool:
                flush_one_readback()
            pbo = readback_pool.pop()
            framebuffer.read_into(pbo, components=3, alignment=1)
            pending_readbacks.append(pbo)
            if len(pending_readbacks) >= readback_depth:
                flush_one_readback()
        else:
            cpu_buffer = free_buffers.get()
            framebuffer.read_into(cpu_buffer, components=3, alignment=1)
            filled_buffers.put(cpu_buffer)

    def emit_rgb_bytes(data):
        flush_all_readbacks()
        cpu_buffer = free_buffers.get()
        cpu_buffer[:] = data
        filled_buffers.put(cpu_buffer)

    start = time.time()
    last_report = start

    try:
        for frame_id in range(total_frames):
            if cancel_callback:
                cancel_callback()

            real_elapsed = int(frame_id * 1000 / fps)
            map_time = start_map_time + int(real_elapsed * speed_multiplier)

            if map_time >= results_start_time:
                emit_rgb_bytes(results_rgb)
            else:
                rows = 0
                for lane in range(keys):
                    x1 = lane_xs[lane]
                    x2 = x1 + column_widths[lane] - 2
                    rows = _rect(geometry, rows, x1, top_y, x2, height, width, height, lane_colours[lane])

                line_colour = _gl_colour_from_skin_colour(
                    cfg.get("colours", {}).get("ColourColumnLine"),
                    (0.25, 0.25, 0.25),
                )
                column_line_widths = cfg.get("column_line_widths") or []
                for i in range(keys + 1):
                    line_width = column_line_widths[i] if i < len(column_line_widths) else 1
                    if line_width <= 0:
                        continue
                    x = play_x + sum(column_widths[:i]) + sum(column_spacing[:max(0, i - 1)])
                    rows = _rect(
                        geometry,
                        rows,
                        x,
                        top_y,
                        x + max(1, int(line_width * skin_scale)),
                        height,
                        width,
                        height,
                        line_colour,
                    )

                if cfg.get("judgement_line"):
                    rows = _rect(
                        geometry,
                        rows,
                        play_x,
                        judge_y - max(1, int(skin_scale)),
                        play_x + play_width,
                        judge_y + max(1, int(skin_scale)),
                        width,
                        height,
                        (0.88, 0.88, 0.88),
                    )

                pressed = [_pressed_at(event_lanes, lane, map_time) for lane in range(keys)]
                texture_commands = []
                receptor_commands = []

                for lane in range(keys):
                    render_lane = keys - 1 - lane if mirror else lane
                    active = pressed[render_lane]
                    img_list = receptor_down_images if active else receptor_images
                    receptor_img = img_list[lane] if lane < len(img_list) else None
                    center_x = lane_xs[lane] + column_widths[lane] // 2
                    if receptor_img is not None:
                        receptor_w, receptor_h, _ = receptor_layouts[lane]
                        receptor_commands.append((receptor_img, center_x, judge_y, receptor_w, receptor_h))
                    else:
                        x1 = lane_xs[lane] + 4
                        x2 = lane_xs[lane] + column_widths[lane] - 4
                        rows = _rect(
                            geometry,
                            rows,
                            x1,
                            judge_y - 24,
                            x2,
                            judge_y + 24,
                            width,
                            height,
                            (0.92, 0.92, 0.96) if active else (0.35, 0.35, 0.40),
                        )

                stage_left = skin.get("stage_left")
                if _has_visible_alpha(stage_left):
                    side_h = max(1, int(height))
                    side_w = max(1, int(stage_left.shape[1] * side_h / max(1, stage_left.shape[0])))
                    side_x = int(play_x - side_w)
                    texture_commands.append(("clipped", stage_left, side_x, 0, side_w, side_h))

                stage_right = skin.get("stage_right")
                if _has_visible_alpha(stage_right):
                    side_h = max(1, int(height))
                    side_x = int(play_x + play_width)
                    side_w = max(1, int(stage_right.shape[1] * side_h / max(1, stage_right.shape[0])))
                    texture_commands.append(("clipped", stage_right, side_x, 0, side_w, side_h))

                stage_light_fps = max(1, int(cfg.get("light_frame_per_second") or 60))
                stage_light = _animated_or_static(skin.get("stage_light_frames"), skin.get("stage_light"), map_time, 0, stage_light_fps)
                if stage_light is not None:
                    light_y = int((cfg.get("light_position") or cfg.get("hit_position") or 480) * skin_scale)
                    for lane, active in enumerate(pressed):
                        if not active:
                            continue
                        center_x = lane_xs[lane] + column_widths[lane] // 2
                        light_w = column_widths[lane]
                        light_h = min(height, int(stage_light.shape[0] * skin_scale))
                        top = max(0, light_y - light_h)
                        texture_commands.append(("stretch", stage_light, int(center_x - light_w / 2), top, light_w, light_h))

                if cfg.get("keys_under_notes"):
                    for image, cx, bottom_y, w, h in receptor_commands:
                        texture_commands.append(("bottom_sized_center", image, cx, bottom_y, w, h))

                visible_start = map_time - visible_margin
                visible_end = map_time + scroll_time + visible_margin
                start_i = max(0, bisect_left(note_times, visible_start) - 128)
                lane_scroll_speed = max(0.01, (judge_y - top_y) / scroll_time)
                additive_commands = []
                active_hold_lanes = []

                for lane, lane_holds in enumerate(ln_hold_lanes):
                    if lane >= len(pressed) or not lane_holds[0]:
                        continue
                    hold_starts, hold_ends = lane_holds
                    hold_i = bisect_right(hold_starts, map_time) - 1
                    if hold_i >= 0 and map_time < hold_ends[hold_i]:
                        active_hold_lanes.append(lane)

                for (
                    note_time,
                    end_time,
                    lane,
                    is_long_note,
                    tap_display_time,
                    head_display_time,
                    head_hit,
                    tail_display_time,
                    ln_miss_display_time,
                    release_cutoff_time,
                ) in fast_notes[start_i:]:
                    if end_time < visible_start:
                        continue
                    if note_time > visible_end:
                        break

                    if is_long_note and release_cutoff_time is not None and map_time >= release_cutoff_time:
                        continue

                    if tap_display_time is not None and map_time >= tap_display_time:
                        continue
                    lane_x = lane_xs[lane]
                    lane_width = column_widths[lane]
                    center_x = lane_x + lane_width // 2
                    note_w = note_widths[lane]
                    hit_y = receptor_layouts[lane][2] if lane < len(receptor_layouts) else judge_y - note_w // 2
                    x1 = center_x - note_w / 2
                    x2 = center_x + note_w / 2
                    lane_scroll_speed = max(0.01, (hit_y - top_y) / scroll_time)
                    y_head = hit_y - (note_time - map_time) * lane_scroll_speed
                    y_tail = hit_y - (end_time - map_time) * lane_scroll_speed
                    colour = note_colours[lane % len(note_colours)]

                    if not is_long_note:
                        note_img = _animated_or_static(
                            (skin.get("note_frames") or [[]] * keys)[lane] if lane < len(skin.get("note_frames") or []) else [],
                            skin["notes"][lane],
                            map_time,
                            note_time,
                            60,
                        )
                        note_draw_w, note_draw_h = mania_element_size(note_img, lane, note_w) if note_img is not None else (note_w, max(12, int(note_w * 0.24)))
                        # mania-note* elements use Origin: Bottom.
                        y1 = y_head - note_draw_h
                        y2 = y_head
                    else:
                        head_img = _animated_or_static(
                            (skin.get("ln_head_frames") or [[]] * keys)[lane] if lane < len(skin.get("ln_head_frames") or []) else [],
                            skin["ln_heads"][lane],
                            map_time,
                            note_time,
                            60,
                        )
                        tail_img = _animated_or_static(
                            (skin.get("ln_tail_frames") or [[]] * keys)[lane] if lane < len(skin.get("ln_tail_frames") or []) else [],
                            skin["ln_tails"][lane],
                            map_time,
                            end_time,
                            60,
                        )
                        head_w, head_h = mania_element_size(head_img, lane, note_w) if head_img is not None else (note_w, note_w)
                        tail_w, tail_h = mania_element_size(tail_img, lane, note_w) if tail_img is not None else (note_w, note_w)
                        # mania-note*H/L/T elements also use Origin: Bottom.
                        # While a LN is held, stable keeps the head cap locked
                        # to the judgement point. The tail continues scrolling
                        # towards the judgement point until the release.
                        head_bottom_y = hit_y if map_time >= head_display_time and head_hit else y_head
                        tail_bottom_y = min(y_tail, hit_y)
                        y1 = min(head_bottom_y - head_h, tail_bottom_y - tail_h)
                        y2 = max(head_bottom_y, tail_bottom_y)
                        y1 = max(y1, top_y)
                        y2 = min(y2, height)

                    if y2 >= top_y and y1 <= height:
                        if not is_long_note:
                            if note_img is not None:
                                texture_commands.append(("bottom_sized_center", note_img, center_x, y_head, note_draw_w, note_draw_h))
                            else:
                                rows = _rect(geometry, rows, x1, y1, x2, y2, width, height, colour)
                        else:
                            body_frames = (skin.get("ln_body_frames") or [[]] * keys)[lane] if lane < len(skin.get("ln_body_frames") or []) else []
                            head_frames = (skin.get("ln_head_frames") or [[]] * keys)[lane] if lane < len(skin.get("ln_head_frames") or []) else []
                            tail_frames = (skin.get("ln_tail_frames") or [[]] * keys)[lane] if lane < len(skin.get("ln_tail_frames") or []) else []
                            body_animation_start = head_display_time if map_time >= head_display_time and head_hit else note_time
                            body_img = _animated_or_static(body_frames, skin["ln_bodies"][lane], map_time, body_animation_start, 60)
                            head_img = _animated_or_static(head_frames, skin["ln_heads"][lane], map_time, note_time, 60)
                            tail_img = _animated_or_static(tail_frames, skin["ln_tails"][lane], map_time, end_time, 60)
                            tail_explicit = True
                            tail_explicit_flags = skin.get("ln_tail_explicit")
                            if tail_explicit_flags is not None and lane < len(tail_explicit_flags):
                                tail_explicit = bool(tail_explicit_flags[lane])
                            head_w, head_h = mania_element_size(head_img, lane, note_w) if head_img is not None else (note_w, note_w)
                            tail_w, tail_h = mania_element_size(tail_img, lane, note_w) if tail_img is not None else (note_w, note_w)
                            head_locked = map_time >= head_display_time and head_hit
                            head_bottom_y = hit_y if head_locked else y_head
                            missed_ln_scrolling_out = ln_miss_display_time is not None and map_time >= ln_miss_display_time
                            tail_bottom_y = min(y_tail, y_head) if missed_ln_scrolling_out else min(y_tail, hit_y)
                            visual_span = abs(float(head_bottom_y) - float(tail_bottom_y))
                            # Tiny LNs should not upload/draw NoteImage#L at
                            # all.  On stable, very short long notes visually
                            # collapse into their caps; drawing even a 1px
                            # body creates the wrong little pedestal seen on
                            # circle skins.  Normal/long LNs keep the connected
                            # body behaviour below.
                            min_body_span = max(2.0, min(head_h, tail_h) * 0.90)
                            should_draw_ln_body = body_img is not None and visual_span > min_body_span
                            if should_draw_ln_body:
                                # osu!mania LN pieces use Origin: Bottom.
                                # The tail/head caps are then drawn over the
                                # body.  The body must therefore start at the
                                # upper cap's bottom anchor and stop behind the
                                # lower cap (roughly at its centre), otherwise
                                # skins like SillyFangirl/Nico69 show the body
                                # starting before the circle or leaving a gap.
                                upper_anchor = min(head_bottom_y, tail_bottom_y)
                                lower_anchor = max(head_bottom_y, tail_bottom_y)
                                lower_cap_h = head_h if head_bottom_y >= tail_bottom_y else tail_h
                                body_y1 = upper_anchor
                                if head_locked:
                                    # While the LN is held, stable keeps the
                                    # NoteImage#H cap locked on the receptor.
                                    # The long body should end behind that cap,
                                    # not continue through the receptor circle.
                                    body_y2 = head_bottom_y - max(1.0, head_h * 0.45)
                                else:
                                    body_y2 = lower_anchor - max(1.0, lower_cap_h * 0.5)
                                if body_y2 <= body_y1 and not head_locked:
                                    body_y2 = (head_bottom_y - max(1.0, head_h * 0.25)) if head_locked else lower_anchor
                                body_y1 = max(top_y, body_y1)
                                body_y2 = min(height, body_y2)
                                body_h = int(body_y2 - body_y1)
                                if body_h > 1:
                                    body_source = _ln_body_source(body_img)
                                    body_w, body_unit_h = mania_element_size(body_source, lane, note_w)
                                    body_styles = cfg.get("note_body_styles") or []
                                    body_style = int(
                                        body_styles[lane]
                                        if lane < len(body_styles) and body_styles[lane] is not None
                                        else (cfg.get("note_body_style") if cfg.get("note_body_style") is not None else 1)
                                    )
                                    if body_style == 0:
                                        texture_commands.append(("stretch", body_source, int(x1), int(body_y1), int(note_w), body_h))
                                    else:
                                        # Stable behaviour:
                                        # 1 = tile NoteL from NoteT to NoteH (top anchored in downscroll)
                                        # 2 = tile NoteL from NoteH to NoteT (bottom anchored in downscroll)
                                        texture_commands.append(("tile_y", body_source, int(x1), int(body_y1), int(note_w), body_h, body_unit_h, body_style == 2))
                            elif body_img is None:
                                rows = _rect(geometry, rows, x1, y1, x2, y2, width, height, colour)
                            if head_img is not None:
                                if map_time < head_display_time:
                                    texture_commands.append(("bottom_sized_center", head_img, center_x, y_head, head_w, head_h))
                                elif head_hit and (tail_display_time is None or map_time < tail_display_time) and pressed[lane]:
                                    texture_commands.append(("bottom_sized_center", head_img, center_x, head_bottom_y, head_w, head_h))
                            # If a custom skin does not provide NoteImage#T,
                            # our loader used to fall back to NoteImage#H.
                            # On skins like yelo mania that turns the held LN
                            # head into a second visible cap/tail, creating
                            # "ghost notes" inside long notes.  Keep explicit
                            # tails untouched; suppress only this implicit
                            # custom fallback when a real body is already used.
                            draw_tail = (
                                tail_img is not None
                                and not missed_ln_scrolling_out
                                and (tail_display_time is None or map_time < tail_display_time)
                            )
                            if draw_tail and not tail_explicit and body_img is not None:
                                draw_tail = False
                            if draw_tail:
                                texture_commands.append(("bottom_sized_center", tail_img, center_x, tail_bottom_y, tail_w, tail_h))

                if not cfg.get("keys_under_notes"):
                    for image, cx, bottom_y, w, h in receptor_commands:
                        texture_commands.append(("bottom_sized_center", image, cx, bottom_y, w, h))

                stage_hint = skin.get("stage_hint")
                if _has_visible_alpha(stage_hint):
                    hint_w = max(1, int(play_width))
                    hint_h = max(1, int(stage_hint.shape[0] * hint_w / max(1, stage_hint.shape[1])))
                    texture_commands.append(("sized_center", stage_hint, play_x + play_width // 2, judge_y, hint_w, hint_h))

                stage_bottom = _animated_or_static(
                    skin.get("stage_bottom_frames"),
                    skin.get("stage_bottom"),
                    map_time,
                    0,
                    max(1, int(cfg.get("light_frame_per_second") or 60)),
                )
                if stage_bottom is not None:
                    cover_w = max(1, int(stage_bottom.shape[1] * skin_scale))
                    cover_h = max(1, int(stage_bottom.shape[0] * skin_scale))
                    cover_x = int(play_x + play_width / 2 - cover_w / 2)
                    cover_y = 0 if cfg.get("upside_down", False) else height - cover_h
                    texture_commands.append(("stretch", stage_bottom, cover_x, cover_y, cover_w, cover_h))

                normal_light = skin.get("hit_lighting_normal")
                long_light = skin.get("hit_lighting_long")
                if _has_visible_alpha(normal_light) or _has_visible_alpha(long_light):
                    light_fps = max(1, int(cfg.get("light_frame_per_second") or 24))
                    normal_frames = skin.get("hit_lighting_normal_frames") or []
                    long_frames = skin.get("hit_lighting_long_frames") or []
                    normal_duration = int(len(normal_frames) * 1000 / light_fps) if normal_frames else int(os.environ.get("MANIA_RENDERER_HIT_LIGHTING_MS", "120"))
                    long_duration = int(len(long_frames) * 1000 / light_fps) if long_frames else int(os.environ.get("MANIA_RENDERER_HIT_LIGHTING_MS", "120"))
                    lighting_duration = max(24, normal_duration, long_duration)
                    lane_effects = {}
                    latest_i = bisect_right(judgement_times, map_time) - 1
                    visible_since = map_time - lighting_duration

                    for index in range(latest_i, -1, -1):
                        judgement = judgements[index]
                        display_time = judgement.get("display_time", judgement["time"])
                        if display_time < visible_since:
                            break
                        if judgement.get("stable_tick") or judgement.get("value", 0) <= 0:
                            continue
                        lane = judgement.get("lane")
                        if lane is None or lane < 0 or lane >= len(lane_xs):
                            continue
                        age = max(0, map_time - display_time)
                        effect = "long" if judgement.get("kind") == "ln_head" else "normal"
                        effect_duration = long_duration if effect == "long" else normal_duration
                        if age > max(24, effect_duration):
                            continue
                        previous = lane_effects.get(lane)
                        if previous is None or age < previous[1]:
                            lane_effects[lane] = (effect, age)

                    normal_widths = cfg.get("lighting_n_widths")
                    long_widths = cfg.get("lighting_l_widths")

                    for lane, (effect, age) in lane_effects.items():
                        frames_key = "hit_lighting_long_frames" if effect == "long" else "hit_lighting_normal_frames"
                        frames = skin.get(frames_key) or []
                        if frames:
                            frame_index = min(len(frames) - 1, int(age * light_fps / 1000.0))
                            image = frames[frame_index]
                        else:
                            image = long_light if effect == "long" and _has_visible_alpha(long_light) else normal_light
                        if not _has_visible_alpha(image):
                            continue
                        effect_duration = long_duration if effect == "long" else normal_duration
                        opacity = 1.0 if frames else max(0.0, 1.0 - (age / max(1, effect_duration)))
                        if opacity <= 0.01:
                            continue
                        image = _alpha_scaled_image(image, opacity)
                        density_key = "hit_lighting_long_density" if effect == "long" else "hit_lighting_normal_density"
                        density = max(1.0, float(skin.get(density_key, 1.0)))
                        configured_widths = long_widths if effect == "long" else normal_widths
                        if configured_widths and lane < len(configured_widths):
                            target_width = max(1, int(configured_widths[lane] * skin_scale))
                        else:
                            target_width = max(1, int(image.shape[1] * skin_scale / density))
                        target_height = max(1, int(image.shape[0] * target_width / max(1, image.shape[1])))
                        center_x = lane_xs[lane] + column_widths[lane] // 2
                        # Official skin.ini docs: LightingN/LightingL are
                        # positioned where the centre of the judgement line
                        # crosses the centre of a lane.
                        hit_center_y = judge_y
                        additive_commands.append((image, int(center_x - target_width / 2), int(hit_center_y - target_height / 2), target_width, target_height))

                array = geometry[:rows]
                buffer.orphan(array.nbytes)
                buffer.write(array)
                ctx_gl.clear(0.0, 0.0, 0.0, 1.0)
                vao.render(moderngl.TRIANGLES, vertices=rows)
                if texture_commands:
                    ctx_gl.enable(moderngl.BLEND)
                    ctx_gl.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                    texture_rects = []

                    def flush_texture_rects():
                        nonlocal texture_rects
                        if texture_rects:
                            texture_painter.draw_cv_rects(texture_rects, preserve_order=True)
                            texture_rects = []

                    for command in texture_commands:
                        if command[0] == "center":
                            _, image, cx, cy, scale, max_w = command
                            if image is not None:
                                ih, iw = image.shape[:2]
                                tw = max(1, int(iw * scale))
                                th = max(1, int(ih * scale))
                                if max_w and tw > max_w:
                                    ratio = max_w / tw
                                    tw = max(1, int(tw * ratio))
                                    th = max(1, int(th * ratio))
                                texture_rects.append((image, int(cx - tw / 2), int(cy - th / 2), tw, th))
                        elif command[0] == "bottom_center":
                            _, image, cx, bottom_y, scale, max_w = command
                            if image is not None:
                                ih, iw = image.shape[:2]
                                tw = max(1, int(iw * scale))
                                th = max(1, int(ih * scale))
                                if max_w and tw > max_w:
                                    ratio = max_w / tw
                                    tw = max(1, int(tw * ratio))
                                    th = max(1, int(th * ratio))
                                texture_rects.append((image, int(cx - tw / 2), int(bottom_y - th), tw, th))
                        elif command[0] == "sized_center":
                            _, image, cx, cy, w, h = command
                            texture_rects.append((image, int(cx - w / 2), int(cy - h / 2), w, h))
                        elif command[0] == "bottom_sized_center":
                            _, image, cx, bottom_y, w, h = command
                            texture_rects.append((image, int(cx - w / 2), int(bottom_y - h), w, h))
                        elif command[0] == "tile_y":
                            _, image, x, y, w, h, unit_h, anchor_bottom = command
                            flush_texture_rects()
                            texture_painter.draw_cv_repeat_y(image, x, y, w, h, unit_h, anchor_bottom)
                        elif command[0] == "clipped":
                            _, image, x, y, w, h = command
                            flush_texture_rects()
                            texture_painter.draw_cv_clipped(image, x, y, w, h)
                        else:
                            _, image, x, y, w, h = command
                            texture_rects.append((image, x, y, w, h))
                    flush_texture_rects()
                if additive_commands:
                    ctx_gl.enable(moderngl.BLEND)
                    ctx_gl.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
                    texture_painter.draw_cv_rects(additive_commands)
                    ctx_gl.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

                state_i = bisect_right(judgement_times, map_time) - 1
                if state_i >= 0:
                    state = judgements[state_i]
                    combo = state["combo"]
                    accuracy = state["accuracy"]
                    combo_changed_at = combo_changed_times[state_i]
                    counts = cumulative_counts[state_i]
                else:
                    combo = 0
                    accuracy = 100.0
                    combo_changed_at = None
                    counts = {"300g": 0, "300": 0, "200": 0, "100": 0, "50": 0, "0": 0}

                if draw_overlays and overlay_cache["static"] is None:
                    static = np.zeros((74, width, 4), dtype=np.uint8)
                    _draw_text(static, title, 18, 32, 0.72, (242, 242, 246), 1)
                    _draw_text(static, f"Beatmap by {mapper}  |  Played by {player}  |  {mods}", 20, 61, 0.47, (220, 220, 226), 1)
                    static[:, :, 3] = 255
                    static.flags.writeable = False
                    overlay_cache["static"] = static

                if draw_overlays:
                    ctx_gl.enable(moderngl.BLEND)
                    ctx_gl.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                    texture_painter.draw_image(overlay_cache["static"], 0, 0)

                combo_age = max(0, map_time - combo_changed_at) if combo_changed_at is not None else 200
                bounce = 1.0
                if combo_age < 160:
                    bounce += 0.12 * np.sin(np.pi * combo_age / 160.0)
                if draw_overlays and combo > 0:
                    combo_center_y = int((cfg.get("combo_position") if cfg.get("combo_position") is not None else 111) * skin_scale)
                    combo_tint = (255, 255, 255)
                    if colour_combo_during_holds and active_hold_lanes:
                        hold_colour = cfg.get("colours", {}).get("ColourHold")
                        if hold_colour and len(hold_colour) >= 3:
                            combo_tint = tuple(hold_colour[:3])
                        else:
                            combo_tint = (255, 230, 65)

                    if not texture_painter.draw_skin_text(
                        str(combo),
                        skin.get("combo_glyphs", {}),
                        play_x + play_width // 2,
                        combo_center_y,
                        cfg.get("combo_overlap", 0),
                        skin_scale * 0.72 * bounce,
                        height,
                        vertical_anchor="center",
                        tint=combo_tint,
                    ):
                        sprite = _text_sprite(combo, 1.35, combo_tint or (255, 255, 255), 2)
                        texture_painter.draw_image(sprite, play_x + play_width // 2, combo_center_y, "center")

                if draw_overlays and lane_xs:
                    latest_i = bisect_right(judgement_times, map_time) - 1
                    while latest_i >= 0 and not judgements[latest_i].get("counts_accuracy", True):
                        latest_i -= 1
                    if latest_i >= 0:
                        latest = judgements[latest_i]
                        display_time = latest.get("display_time", latest["time"])
                        if map_time - display_time <= 1000:
                            key = latest.get("image_key")
                            if key is None:
                                value = latest["value"]
                                key = "300g" if value == 305 else "300" if value >= 300 else "200" if value >= 200 else "100" if value >= 100 else "50" if value >= 50 else "0"
                            age = max(0, map_time - display_time)
                            frames = skin.get("hit_image_frames", {}).get(key) or []
                            if frames:
                                frame_index = int(age * 60 / 1000.0) % len(frames)
                                img = frames[frame_index]
                            else:
                                img = skin.get("hit_images", {}).get(key)
                            if img is not None:
                                score_position = cfg.get("score_position") if cfg.get("score_position") is not None else 300
                                judgement_y = int(score_position * skin_scale)
                                play_center_x = (lane_xs[0] + lane_xs[-1] + column_widths[-1]) // 2
                                density = skin.get("hit_image_densities", {}).get(key, 1.0)
                                resolution_scale = height / 768.0 / max(1.0, density)
                                texture_painter.draw_cv_centered(img, play_center_x, judgement_y, resolution_scale)

                if draw_overlays:
                    panel_w = min(520, max(360, width // 3))
                    panel_h = height - 74
                    elapsed_text = f"{format_clock(max(0, map_time - start_map_time))} / {format_clock(max(0, gameplay_end_time - start_map_time))}"
                    stats_key = (
                        combo,
                        int(accuracy * 100),
                        counts.get("300g", 0),
                        counts.get("300", 0),
                        counts.get("200", 0),
                        counts.get("100", 0),
                        counts.get("50", 0),
                        counts.get("0", 0),
                        elapsed_text,
                    )
                    if overlay_cache["last_stats_key"] != stats_key:
                        panel = np.zeros((panel_h, panel_w, 4), dtype=np.uint8)
                        right = panel_w - 24
                        y = 34
                        _draw_text(panel, f"{accuracy:.2f}%", right, y, 0.70, (240, 240, 245), 1, "right")
                        y += 54
                        for label, key, colour in (
                            ("300g", "300g", (135, 230, 255)),
                            ("300", "300", (135, 255, 150)),
                            ("200", "200", (245, 220, 90)),
                            ("100", "100", (245, 155, 80)),
                            ("50", "50", (215, 105, 210)),
                            ("miss", "0", (80, 80, 255)),
                        ):
                            _draw_text(panel, f"{label}: {counts.get(key, 0)}", right, y, 0.48, colour, 1, "right")
                            y += 25
                        pp = mania_pp_value(star_rating, counts)
                        _draw_text(panel, "pp: N/A" if pp is None else f"pp: {pp:.2f}", right, y + 8, 0.58, (220, 220, 225), 1, "right")
                        y += 56
                        lane_w_ui = 28
                        lane_gap = 8
                        total_w = keys * lane_w_ui + (keys - 1) * lane_gap
                        x1 = right - total_w
                        history_top = y + 20
                        history_h = 132
                        history_bottom = history_top + history_h
                        _draw_text(panel, "INPUT / BPM", right, y, 0.42, (190, 190, 195), 1, "right")
                        cv2.rectangle(panel, (x1 - 8, history_top - 8), (right + 8, history_bottom + 68), (7, 8, 13, 232), -1)
                        cv2.rectangle(panel, (x1 - 8, history_top - 8), (right + 8, history_bottom + 68), (58, 62, 76, 230), 1, cv2.LINE_AA)
                        for lane, (times, states) in enumerate(event_lanes):
                            lx = x1 + lane * (lane_w_ui + lane_gap)
                            cv2.rectangle(panel, (lx, history_top), (lx + lane_w_ui, history_bottom), (12, 14, 22, 170), -1)
                            key_top = history_bottom + 8
                            cv2.rectangle(panel, (lx, key_top), (lx + lane_w_ui, key_top + 22), (18, 19, 26, 250), -1)
                            cv2.rectangle(panel, (lx, key_top), (lx + lane_w_ui, key_top + 22), (168, 146, 70, 255), 1, cv2.LINE_AA)
                            _draw_text(panel, lane + 1, lx + lane_w_ui // 2, key_top + 16, 0.34, (245, 245, 238), 1, "center")
                        overlay_cache["key_rects"] = [
                            (x1 + lane * (lane_w_ui + lane_gap), history_bottom + 8, lane_w_ui, 22, lane + 1)
                            for lane in range(keys)
                        ]
                        t_y = min(panel_h - 214, history_bottom + 104)
                        overlay_cache["timeline_rect"] = (right - 320, t_y, right, t_y + 8)
                        cv2.rectangle(panel, (right - 320, t_y), (right, t_y + 8), (48, 48, 54, 240), -1)
                        _draw_text(panel, elapsed_text, right, t_y - 8, 0.38, (218, 218, 224), 1, "right")
                        if difficulty_profile:
                            gx1, gy1, gx2, gy2 = right - 420, panel_h - 170, right, panel_h - 10
                            overlay_cache["strain_rect"] = (gx1, gy1, gx2, gy2)
                            cv2.rectangle(panel, (gx1, gy1), (gx2, gy2), (0, 0, 0, 245), -1)
                            values = np.array(difficulty_profile, dtype=np.float32)
                            if values.max() > 0:
                                values /= values.max()
                            points = []
                            for i, _ in enumerate(values):
                                px = int(gx1 + i / max(1, len(values) - 1) * (gx2 - gx1))
                                py = int(gy2 - 12 - values[i] * (gy2 - gy1 - 24))
                                points.append((px, py))
                            overlay_cache["strain_points"] = points
                            if len(points) > 1:
                                cv2.polylines(panel, [np.array(points, dtype=np.int32)], False, (150, 150, 158, 255), 2, cv2.LINE_AA)
                            dyn_w = gx2 - gx1 + 1
                            dyn_h = gy2 - gy1 + 1
                            green_strain = np.zeros((dyn_h, dyn_w, 4), dtype=np.uint8)
                            if len(points) > 1:
                                local_points = np.array([(px - gx1, py - gy1) for px, py in points], dtype=np.int32)
                                cv2.polylines(green_strain, [local_points], False, (95, 225, 120, 255), 2, cv2.LINE_AA)
                            green_strain.flags.writeable = False
                            overlay_cache["strain_dynamic"] = green_strain
                            overlay_cache["strain_dynamic_size"] = (dyn_w, dyn_h)
                        else:
                            overlay_cache["strain_rect"] = None
                            overlay_cache["strain_points"] = None
                            overlay_cache["strain_dynamic"] = None
                            overlay_cache["strain_dynamic_size"] = None
                        overlay_cache["stats"] = panel
                        overlay_cache["last_stats_key"] = stats_key
                        overlay_cache["stats_texture_dirty"] = True
                    panel = overlay_cache["stats"]
                    if panel is not None:
                        panel_x = width - panel.shape[1]
                        panel_y = 74
                        if overlay_cache.get("stats_texture_dirty"):
                            overlay_cache["stats_texture"] = texture_painter.update_named("stats_panel", panel)
                            overlay_cache["stats_texture_dirty"] = False
                        texture_painter.draw_texture(overlay_cache["stats_texture"], panel_x, panel_y)

                        overlay_rows = 0
                        progress = max(0.0, min(1.0, (map_time - start_map_time) / max(1, gameplay_end_time - start_map_time)))
                        timeline_rect = overlay_cache.get("timeline_rect")
                        if timeline_rect:
                            tx1, ty1, tx2, ty2 = timeline_rect
                            tx1 += panel_x
                            tx2 += panel_x
                            ty1 += panel_y
                            ty2 += panel_y
                            overlay_rows = _rect(geometry, overlay_rows, tx1, ty1, tx1 + int((tx2 - tx1) * progress), ty2, width, height, (140 / 255, 225 / 255, 145 / 255))
                        strain_rect = overlay_cache.get("strain_rect")
                        dynamic_strain_width = 0
                        if strain_rect:
                            gx1, gy1, gx2, gy2 = strain_rect
                            cursor_x = panel_x + gx1 + int((gx2 - gx1) * progress)
                            dyn_w, _ = overlay_cache.get("strain_dynamic_size") or (0, 0)
                            dynamic_strain_width = max(0, min(dyn_w, int(dyn_w * progress)))
                            overlay_rows = _rect(geometry, overlay_rows, cursor_x - 1, panel_y + gy1, cursor_x + 1, panel_y + gy2, width, height, (230 / 255, 230 / 255, 234 / 255))
                        history_window = 1050
                        for lane, (times, states) in enumerate(event_lanes):
                            if lane >= keys:
                                continue
                            if not overlay_cache.get("key_rects"):
                                continue
                            kx, ky, kw, kh, _ = overlay_cache["key_rects"][lane]
                            history_bottom = ky - 5
                            history_h = 112
                            history_top = history_bottom - history_h
                            start_i = max(0, bisect_left(times, map_time - history_window) - 1)
                            end_i = bisect_right(times, map_time)
                            for event_i in range(start_i, end_i):
                                if not states[event_i]:
                                    continue
                                press_time = times[event_i]
                                release_time = times[event_i + 1] if event_i + 1 < len(times) else map_time
                                release_time = min(release_time, map_time)
                                if release_time < map_time - history_window or press_time > map_time:
                                    continue
                                age_press = max(0, min(history_window, map_time - press_time))
                                age_release = max(0, min(history_window, map_time - release_time))
                                py_press = history_bottom - int(age_press / history_window * history_h)
                                py_release = history_bottom - int(age_release / history_window * history_h)
                                y_top = max(history_top, min(history_bottom, min(py_press, py_release)))
                                y_bottom = max(history_top, min(history_bottom, max(py_press, py_release)))
                                if y_bottom - y_top < 4:
                                    y_top -= 2
                                    y_bottom += 2
                                alpha_boost = 1.0 - min(1.0, age_press / history_window) * 0.35
                                ax1 = panel_x + kx + 3
                                ax2 = panel_x + kx + kw - 3
                                ay1 = panel_y + y_top
                                ay2 = panel_y + y_bottom
                                overlay_rows = _rect(geometry, overlay_rows, ax1, ay1, ax2, ay2, width, height, (0.86 * alpha_boost, 0.90 * alpha_boost, 0.88 * alpha_boost))
                        for lane, (kx, ky, kw, kh, label) in enumerate(overlay_cache.get("key_rects", [])):
                            if lane < len(event_lanes):
                                times, states = event_lanes[lane]
                                bpm_text = f"{stable_key_bpm(times, states, map_time):03d}"
                                texture_painter.draw_image(
                                    _key_label_sprite(bpm_text, 0.32, (205, 205, 210)),
                                    panel_x + kx + kw // 2,
                                    panel_y + ky + kh + 17,
                                    "center",
                                )
                            if not (bool(pressed[lane]) if lane < len(pressed) else False):
                                continue
                            ax = panel_x + kx
                            base_y = panel_y + ky
                            press_age = 90
                            if lane < len(event_lanes):
                                times, states = event_lanes[lane]
                                event_i = bisect_right(times, map_time) - 1
                                if event_i >= 0 and states[event_i]:
                                    press_age = max(0, map_time - times[event_i])
                            press_t = max(0.0, min(1.0, press_age / 85.0))
                            press_ease = 1.0 - (1.0 - press_t) * (1.0 - press_t)
                            offset = int(round(max(2, kh * 0.18) * press_ease))
                            ay = base_y + offset
                            overlay_rows = _rect(geometry, overlay_rows, ax - 2, base_y - 2, ax + kw + 2, base_y + kh + 2, width, height, (5 / 255, 7 / 255, 12 / 255))
                            overlay_rows = _rect(geometry, overlay_rows, ax - 2, ay - 2, ax + kw + 2, ay + kh + 2, width, height, (40 / 255, 115 / 255, 145 / 255))
                            overlay_rows = _rect(geometry, overlay_rows, ax, ay, ax + kw, ay + kh, width, height, (90 / 255, 205 / 255, 245 / 255))
                            overlay_rows = _rect(geometry, overlay_rows, ax, ay, ax + kw, ay + 1, width, height, (180 / 255, 238 / 255, 255 / 255))
                            overlay_rows = _rect(geometry, overlay_rows, ax, ay + kh - 1, ax + kw, ay + kh, width, height, (32 / 255, 125 / 255, 170 / 255))
                            overlay_rows = _rect(geometry, overlay_rows, ax, ay, ax + 1, ay + kh, width, height, (180 / 255, 238 / 255, 255 / 255))
                            overlay_rows = _rect(geometry, overlay_rows, ax + kw - 1, ay, ax + kw, ay + kh, width, height, (32 / 255, 125 / 255, 170 / 255))
                        if overlay_rows:
                            if dynamic_strain_width > 0 and overlay_cache.get("strain_dynamic") is not None:
                                ctx_gl.enable(moderngl.BLEND)
                                ctx_gl.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                                sx1, sy1, _, _ = overlay_cache["strain_rect"]
                                _, dynamic_strain_h = overlay_cache.get("strain_dynamic_size") or (0, 0)
                                texture_painter.draw_cv_crop(
                                    overlay_cache["strain_dynamic"],
                                    panel_x + sx1,
                                    panel_y + sy1,
                                    dynamic_strain_width,
                                    dynamic_strain_h,
                                )
                            ctx_gl.disable(moderngl.BLEND)
                            active_array = geometry[:overlay_rows]
                            buffer.orphan(active_array.nbytes)
                            buffer.write(active_array)
                            vao.render(moderngl.TRIANGLES, vertices=overlay_rows)
                            ctx_gl.enable(moderngl.BLEND)
                            ctx_gl.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                            for lane, (kx, ky, kw, kh, label) in enumerate(overlay_cache.get("key_rects", [])):
                                if bool(pressed[lane]) if lane < len(pressed) else False:
                                    press_age = 90
                                    if lane < len(event_lanes):
                                        times, states = event_lanes[lane]
                                        event_i = bisect_right(times, map_time) - 1
                                        if event_i >= 0 and states[event_i]:
                                            press_age = max(0, map_time - times[event_i])
                                    press_t = max(0.0, min(1.0, press_age / 85.0))
                                    press_ease = 1.0 - (1.0 - press_t) * (1.0 - press_t)
                                    offset = int(round(max(2, kh * 0.18) * press_ease))
                                    texture_painter.draw_image(_key_label_sprite(label, 0.40, (245, 245, 238)), panel_x + kx + kw // 2, panel_y + ky + offset + 2, "center")
                        elif dynamic_strain_width > 0 and overlay_cache.get("strain_dynamic") is not None:
                            ctx_gl.enable(moderngl.BLEND)
                            ctx_gl.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                            sx1, sy1, _, _ = overlay_cache["strain_rect"]
                            _, dynamic_strain_h = overlay_cache.get("strain_dynamic_size") or (0, 0)
                            texture_painter.draw_cv_crop(
                                overlay_cache["strain_dynamic"],
                                panel_x + sx1,
                                panel_y + sy1,
                                dynamic_strain_width,
                                dynamic_strain_h,
                            )

                emit_framebuffer()

            now = time.time()
            if now - last_report >= 2.0 or frame_id + 1 == total_frames:
                done = frame_id + 1
                render_fps = done / max(0.001, now - start)
                if progress_callback:
                    progress_callback(
                        min(85, 7 + int(done / total_frames * 78)),
                        f"Fast GPU frame: {done}/{total_frames} | Render: {render_fps:.1f} fps"
                    )
                last_report = now
    finally:
        flush_all_readbacks()
        filled_buffers.put(None)
        writer_thread.join()
        if ffmpeg.stdin:
            ffmpeg.stdin.close()

    if ffmpeg.wait() != 0:
        raise RuntimeError("Fast GPU FFmpeg encode failed")

    audio_path = Path(osu_file).parent / beatmap.audio_file
    if audio_path.exists():
        cmd = [
            ffmpeg_binary(), "-y",
            "-i", str(silent_video),
            "-ss", str(audio_start_ms / 1000),
            "-i", str(audio_path),
            *make_audio_args(ctx["speed_multiplier"], nightcore_pitch),
            "-c:v", "copy",
            "-c:a", "aac",
            "-t", str(audio_duration_s),
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        silent_video.unlink(missing_ok=True)
    else:
        silent_video.replace(output_path)

    if progress_callback:
        progress_callback(100, f"Done: {output_path}")

    return True
