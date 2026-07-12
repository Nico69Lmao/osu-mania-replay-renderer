from pathlib import Path
import os
import shutil
import subprocess


FFMPEG_BINARY = None


def make_audio_args(speed_multiplier, nightcore_pitch):
    if speed_multiplier == 1.0:
        return ["-filter:a", "apad"]

    if nightcore_pitch and speed_multiplier == 1.5:
        return ["-filter:a", "aresample=48000,asetrate=48000*1.5,aresample=48000,apad"]

    if speed_multiplier == 1.5:
        return ["-filter:a", "atempo=1.5,apad"]

    if speed_multiplier == 0.75:
        return ["-filter:a", "atempo=0.75,apad"]

    return ["-filter:a", f"atempo={speed_multiplier},apad"]


def ffmpeg_binary():
    global FFMPEG_BINARY

    if FFMPEG_BINARY:
        return FFMPEG_BINARY

    candidates = []
    configured = os.environ.get("MANIA_RENDERER_FFMPEG")

    if configured:
        candidates.append(configured)

    system_ffmpeg = shutil.which("ffmpeg")

    if system_ffmpeg:
        candidates.append(system_ffmpeg)

    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass

    for candidate in dict.fromkeys(candidates):
        try:
            subprocess.run(
                [candidate, "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            FFMPEG_BINARY = candidate
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue

    raise RuntimeError(
        "FFmpeg was not found. Reinstall the application or set MANIA_RENDERER_FFMPEG."
    )


def ffmpeg_encoder_names():
    try:
        result = subprocess.run(
            [ffmpeg_binary(), "-hide_banner", "-encoders"],
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


def frame_input_args(frame_stream, fps, frame_stream_mode, width=None, height=None):
    if frame_stream_mode == "raw":
        if width is None or height is None:
            raise ValueError("Raw frame streams require width and height")

        return [
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s:v", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", str(frame_stream),
        ]

    return [
        "-f", "mjpeg",
        "-framerate", str(fps),
        "-i", str(frame_stream),
    ]


def video_encode_commands(fps, frame_stream, silent_video, frame_stream_mode="mjpeg", width=None, height=None, output_fps=None):
    encoders = ffmpeg_encoder_names()
    input_args = frame_input_args(frame_stream, fps, frame_stream_mode, width, height)
    ffmpeg = ffmpeg_binary()
    output_fps = int(output_fps or fps)
    vaapi_qp = os.environ.get("MANIA_RENDERER_VAAPI_QP", "24")
    x264_crf = os.environ.get("MANIA_RENDERER_X264_CRF", "20")
    x264_preset = os.environ.get("MANIA_RENDERER_X264_PRESET", "veryfast")

    device = vaapi_device()

    if device and "h264_vaapi" in encoders:
        extra_args = ["-r", str(output_fps)] if output_fps != fps else []
        yield [
            ffmpeg, "-y",
            "-vaapi_device", device,
            *input_args,
            *extra_args,
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi",
            "-qp", vaapi_qp,
            str(silent_video),
        ]

    if "h264_qsv" in encoders:
        extra_args = ["-r", str(output_fps)] if output_fps != fps else []
        yield [
            ffmpeg, "-y",
            *input_args,
            *extra_args,
            "-vf", "format=nv12",
            "-c:v", "h264_qsv",
            "-global_quality", vaapi_qp,
            str(silent_video),
        ]

    if "h264_amf" in encoders:
        extra_args = ["-r", str(output_fps)] if output_fps != fps else []
        yield [
            ffmpeg, "-y",
            *input_args,
            *extra_args,
            "-c:v", "h264_amf",
            "-quality", "quality",
            "-qp_i", "18",
            "-qp_p", "18",
            str(silent_video),
        ]

    extra_args = ["-r", str(output_fps)] if output_fps != fps else []
    yield [
        ffmpeg, "-y",
        *input_args,
        *extra_args,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", x264_preset,
        "-crf", x264_crf,
        str(silent_video),
    ]


def encode_silent_video(fps, frame_stream, silent_video, progress_callback=None, frame_stream_mode="mjpeg", width=None, height=None, output_fps=None):
    last_error = None
    attempts = []

    for cmd in video_encode_commands(fps, frame_stream, silent_video, frame_stream_mode, width, height, output_fps):
        encoder = cmd[cmd.index("-c:v") + 1]

        if progress_callback:
            progress_callback(87, f"Encoding video: trying {encoder}...")

        try:
            env = os.environ.copy()

            if encoder in ("h264_vaapi", "h264_qsv") and Path("/usr/lib/dri/iHD_drv_video.so").exists():
                env.setdefault("LIBVA_DRIVER_NAME", "iHD")

            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            attempts.append({"encoder": encoder, "status": "success"})
            return cmd, attempts
        except subprocess.CalledProcessError as error:
            last_error = error
            stderr = (error.stderr or "").strip().splitlines()
            attempts.append({
                "encoder": encoder,
                "status": "failed",
                "error": "\n".join(stderr[-8:]),
            })

    if last_error:
        raise last_error

    return None, attempts


def format_duration(seconds):
    seconds = max(0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes, remaining = divmod(int(round(seconds)), 60)
    return f"{minutes}m {remaining:02d}s"
