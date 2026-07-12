import os

import numpy as np


VERTEX_SHADER = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 uv;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    uv = in_uv;
}
"""


FRAGMENT_SHADER = """
#version 330
uniform sampler2D source;
in vec2 uv;
out vec4 colour;

void main() {
    colour = texture(source, uv);
}
"""


class GpuCompositor:
    """Batches legacy skin texture draws through a headless OpenGL context."""

    def __init__(self):
        import moderngl

        self.moderngl = moderngl
        self.context_backend = None
        self.context = self._create_context(moderngl)
        self.program = self.context.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.vertex_buffer = self.context.buffer(reserve=16 * 4)
        self.vertex_array = self.context.vertex_array(
            self.program,
            [(self.vertex_buffer, "2f 2f", "in_position", "in_uv")],
        )
        self.program["source"].value = 0
        self.texture_cache = {}
        self.commands = []
        self.frame_size = None
        self.frame_texture = None
        self.output_texture = None
        self.framebuffer = None
        self.read_buffer = None
        self.vertex_data = np.empty(16, dtype="f4")

    @property
    def renderer_name(self):
        renderer = self.context.info.get("GL_RENDERER", "OpenGL")
        return f"{renderer} ({self.context_backend})" if self.context_backend else renderer

    @staticmethod
    def _context_attempts():
        preferred = os.environ.get("MANIA_RENDERER_GL_BACKEND")
        attempts = []

        if preferred:
            attempts.append(preferred.strip().lower())

        attempts.extend(["egl", "default", "osmesa"])

        seen = set()

        for backend in attempts:
            if not backend or backend in seen:
                continue

            seen.add(backend)
            yield backend

    def _create_context(self, moderngl):
        errors = []

        for backend in self._context_attempts():
            kwargs = {} if backend == "default" else {"backend": backend}

            try:
                context = moderngl.create_standalone_context(require=330, **kwargs)
            except Exception as error:
                errors.append(f"{backend}: {type(error).__name__}: {error}")
                continue

            self.context_backend = backend
            return context

        raise RuntimeError("Could not create an OpenGL context. " + " | ".join(errors))

    def queue(self, image, x, y, width, height, blend_mode="normal"):
        self.commands.append((
            image,
            int(x),
            int(y),
            max(1, int(width)),
            max(1, int(height)),
            blend_mode,
        ))

    def _ensure_frame(self, width, height):
        if self.frame_size == (width, height):
            return

        for resource in (self.framebuffer, self.output_texture, self.frame_texture):
            if resource is not None:
                resource.release()

        self.frame_texture = self.context.texture((width, height), 3, dtype="f1")
        self.frame_texture.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)
        self.output_texture = self.context.texture((width, height), 3, dtype="f1")
        self.output_texture.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)
        self.framebuffer = self.context.framebuffer(color_attachments=[self.output_texture])
        self.frame_size = (width, height)
        self.read_buffer = bytearray(width * height * 3)

    def _texture_for(self, image):
        cacheable = not image.flags.writeable
        pointer = int(image.__array_interface__["data"][0])
        key = (pointer, image.shape, image.strides)
        cached = self.texture_cache.get(key) if cacheable else None

        if cached is not None:
            return cached, False

        contiguous = np.ascontiguousarray(image)
        components = 1 if contiguous.ndim == 2 else contiguous.shape[2]
        texture = self.context.texture(
            (contiguous.shape[1], contiguous.shape[0]),
            components,
            contiguous.tobytes(),
            alignment=1,
        )
        texture.filter = (self.moderngl.LINEAR, self.moderngl.LINEAR)

        if cacheable:
            self.texture_cache[key] = texture

        return texture, not cacheable

    def _draw(self, texture, x, y, width, height, frame_width, frame_height):
        left = x
        top = y
        right = x + width
        bottom = y + height

        clipped_left = max(0, left)
        clipped_top = max(0, top)
        clipped_right = min(frame_width, right)
        clipped_bottom = min(frame_height, bottom)

        if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
            return

        u1 = (clipped_left - left) / width
        u2 = (clipped_right - left) / width
        v1 = (clipped_top - top) / height
        v2 = (clipped_bottom - top) / height
        x1 = clipped_left / frame_width * 2.0 - 1.0
        x2 = clipped_right / frame_width * 2.0 - 1.0
        y1 = 1.0 - clipped_top / frame_height * 2.0
        y2 = 1.0 - clipped_bottom / frame_height * 2.0
        self.vertex_data[:] = (x1, y1, u1, v1, x1, y2, u1, v2, x2, y1, u2, v1, x2, y2, u2, v2)
        self.vertex_buffer.write(self.vertex_data)
        texture.use(0)
        self.vertex_array.render(self.moderngl.TRIANGLE_STRIP)

    def flush(self, frame):
        if not self.commands:
            return

        height, width = frame.shape[:2]
        self._ensure_frame(width, height)
        self.frame_texture.write(np.ascontiguousarray(frame), alignment=1)
        self.framebuffer.use()
        self.context.viewport = (0, 0, width, height)
        self.context.disable(self.moderngl.BLEND)
        self._draw(self.frame_texture, 0, 0, width, height, width, height)
        self.context.enable(self.moderngl.BLEND)
        self.context.blend_func = self.moderngl.SRC_ALPHA, self.moderngl.ONE_MINUS_SRC_ALPHA

        current_blend_mode = "normal"

        for image, x, y, target_width, target_height, blend_mode in self.commands:
            if blend_mode != current_blend_mode:
                if blend_mode == "additive":
                    self.context.blend_func = self.moderngl.SRC_ALPHA, self.moderngl.ONE
                else:
                    self.context.blend_func = self.moderngl.SRC_ALPHA, self.moderngl.ONE_MINUS_SRC_ALPHA

                current_blend_mode = blend_mode

            texture, temporary = self._texture_for(image)

            try:
                self._draw(texture, x, y, target_width, target_height, width, height)
            finally:
                if temporary:
                    texture.release()

        self.framebuffer.read_into(self.read_buffer, components=3, alignment=1)
        frame[:] = np.frombuffer(self.read_buffer, dtype=np.uint8).reshape(height, width, 3)[::-1]
        self.commands.clear()

    def release(self):
        self.commands.clear()

        for texture in self.texture_cache.values():
            texture.release()

        self.texture_cache.clear()

        for resource in (
            self.framebuffer,
            self.output_texture,
            self.frame_texture,
            self.vertex_array,
            self.vertex_buffer,
            self.program,
        ):
            if resource is not None:
                resource.release()

        self.context.release()


LAST_GPU_ERROR = None


def create_gpu_compositor():
    global LAST_GPU_ERROR

    try:
        compositor = GpuCompositor()
    except Exception as error:
        LAST_GPU_ERROR = str(error)
        return None

    LAST_GPU_ERROR = None
    return compositor


def detect_gpu_renderer():
    compositor = create_gpu_compositor()

    if compositor is None:
        return None

    try:
        return compositor.renderer_name
    finally:
        compositor.release()


def gpu_unavailable_reason():
    return LAST_GPU_ERROR
