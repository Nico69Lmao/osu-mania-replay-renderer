import io
import json
from hashlib import md5
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np

from osu_mania_replay_renderer import updater
from osu_mania_replay_renderer.osu_finder import find_beatmap_by_hash, find_osu_folder
from osu_mania_replay_renderer.renderer import (
    RenderCancelled,
    apply_motion_blur,
    draw_difficulty_graph,
    draw_hit_lighting,
    draw_key_input_overlay,
    draw_skin_text,
    ensure_not_cancelled,
    layout_point,
)
from osu_mania_replay_renderer.layout_model import SCENE_HEIGHT, SCENE_WIDTH, layout_definitions


class JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def response(data):
    return JsonResponse(json.dumps(data).encode("utf-8"))


class OsuFinderTests(unittest.TestCase):
    def test_detects_osu_wine_and_reports_search_progress(self):
        with TemporaryDirectory() as temporary:
            home = Path(temporary)
            osu_folder = home / ".local/share/osu-wine/osu!"
            songs = osu_folder / "Songs/1 Test"
            songs.mkdir(parents=True)
            (osu_folder / "Skins").mkdir()
            (osu_folder / "osu!.db").write_bytes(b"db")
            target = songs / "target.osu"
            target.write_bytes(b"target beatmap")
            (songs / "other.osu").write_bytes(b"other beatmap")

            self.assertEqual(
                find_osu_folder(system="Linux", home=home, environ={}),
                str(osu_folder.resolve()),
            )

            progress = []
            found = find_beatmap_by_hash(
                osu_folder,
                md5(b"target beatmap").hexdigest(),
                progress_callback=lambda *values: progress.append(values),
            )
            self.assertEqual(found, str(target))
            self.assertTrue(progress)
            self.assertEqual(progress[-1][1], 2)

            fast_progress = []
            found = find_beatmap_by_hash(
                osu_folder,
                md5(b"target beatmap").hexdigest(),
                progress_callback=lambda *values: fast_progress.append(values),
                preferred_path=target,
            )
            self.assertEqual(found, str(target))
            self.assertEqual(fast_progress, [(1, 1, 0.0)])

    def test_detects_default_windows_install(self):
        with TemporaryDirectory() as temporary:
            local_app_data = Path(temporary)
            osu_folder = local_app_data / "osu!"
            (osu_folder / "Songs").mkdir(parents=True)

            self.assertEqual(
                find_osu_folder(
                    system="Windows",
                    home=local_app_data,
                    environ={"LOCALAPPDATA": str(local_app_data)},
                ),
                str(osu_folder.resolve()),
            )


class UpdaterTests(unittest.TestCase):
    def test_returns_compatible_new_release(self):
        release = {
            "tag_name": "v9.0.0",
            "html_url": "https://example.test/release",
            "body": "notes",
            "assets": [
                {
                    "name": "renderer.AppImage",
                    "browser_download_url": "https://example.test/appimage",
                }
            ],
        }

        with patch.object(updater, "github_token", return_value="token"), patch.object(
            updater, "urlopen", return_value=response(release)
        ):
            info = updater.check_for_update()

        self.assertEqual(info.version, "9.0.0")
        self.assertEqual(info.asset_name, "renderer.AppImage")

    def test_private_repository_error_is_clear(self):
        error = HTTPError(updater.LATEST_RELEASE_API, 404, "Not Found", {}, None)

        with patch.object(updater, "github_token", return_value=None), patch.object(
            updater, "urlopen", side_effect=error
        ):
            with self.assertRaisesRegex(updater.UpdateCheckError, "private"):
                updater.check_for_update()

    def test_new_release_without_platform_asset_is_not_current(self):
        release = {"tag_name": "v9.0.0", "assets": []}

        with patch.object(updater, "github_token", return_value="token"), patch.object(
            updater, "urlopen", return_value=response(release)
        ):
            with self.assertRaisesRegex(updater.UpdateCheckError, "no compatible asset"):
                updater.check_for_update()


class RendererControlTests(unittest.TestCase):
    def test_layout_positions_are_normalised_and_clamped(self):
        self.assertEqual(layout_point({"combo": [0.25, 0.75]}, "combo", 1280, 720), (320, 540))
        self.assertEqual(layout_point({"combo": [-1, 2]}, "combo", 1280, 720), (0, 720))
        self.assertIsNone(layout_point({}, "combo", 1280, 720))

    def test_cancel_callback_raises_render_cancelled(self):
        with self.assertRaises(RenderCancelled):
            ensure_not_cancelled(lambda: True)

    def test_overlay_backgrounds_are_black(self):
        frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
        draw_key_input_overlay(
            frame,
            [([], []) for _ in range(4)],
            [False] * 4,
            0,
            1280,
            180,
            (1100, 350),
        )
        self.assertTrue(np.any(np.all(frame == 0, axis=2)))

        strain = np.full((720, 1280, 3), 90, dtype=np.uint8)
        draw_difficulty_graph(
            strain,
            [0.2, 0.7, 0.4, 1.0] * 40,
            500,
            0,
            1000,
            1280,
            720,
            420,
            440,
            (900, 650),
        )
        self.assertTrue(np.any(np.all(strain == 0, axis=2)))

    def test_skin_font_preserves_punctuation_baseline(self):
        digit = np.zeros((10, 5, 4), dtype=np.uint8)
        digit[1:9, 1:4, :3] = 255
        digit[1:9, 1:4, 3] = 255
        dot = np.zeros((10, 5, 4), dtype=np.uint8)
        dot[8:10, 2:4, :3] = 255
        dot[8:10, 2:4, 3] = 255
        digit.flags.writeable = False
        dot.flags.writeable = False
        glyphs = {"1": {1.0: digit}, ".": {1.0: dot}}
        frame = np.zeros((14, 24, 3), dtype=np.uint8)

        self.assertTrue(draw_skin_text(frame, "1.", glyphs, 12, 0, 0, 1.0))
        self.assertTrue(np.any(frame[8:10, 12:17] > 0))
        self.assertFalse(np.any(frame[:5, 12:17] > 0))

    def test_gpu_compositor_matches_alpha_blending_when_available(self):
        from osu_mania_replay_renderer.gpu_compositor import create_gpu_compositor

        compositor = create_gpu_compositor()

        if compositor is None:
            self.skipTest("No headless OpenGL context is available")

        try:
            frame = np.zeros((8, 10, 3), dtype=np.uint8)
            frame[0, :, 0] = 16
            image = np.zeros((4, 4, 4), dtype=np.uint8)
            image[:, :, 2] = 200
            image[:, :, 3] = 128
            image.flags.writeable = False
            compositor.queue(image, 2, 2, 4, 4)
            compositor.flush(frame)
            self.assertTrue(np.all(frame[0, :, 0] == 16))
            self.assertIn(int(frame[3, 3, 2]), range(99, 102))

            additive = np.full((8, 10, 3), 80, dtype=np.uint8)
            compositor.queue(image, 2, 2, 4, 4, "additive")
            compositor.flush(additive)
            self.assertGreater(int(additive[3, 3, 2]), 80)
            self.assertEqual(int(additive[3, 3, 0]), 80)
        finally:
            compositor.release()

    def test_layout_preview_uses_renderer_skin_scale(self):
        self.assertEqual((SCENE_WIDTH, SCENE_HEIGHT), (1920, 1080))
        glyph = np.zeros((100, 26, 4), dtype=np.uint8)
        glyph[:, :, 3] = 255
        judgement = np.zeros((50, 50, 4), dtype=np.uint8)
        judgement[:, :, 3] = 255
        skin = {
            "cfg": {
                "column_widths": [70] * 4,
                "column_spacing": [3] * 3,
                "combo_overlap": 1,
            },
            "keys": [None] * 4,
            "combo_glyphs": {character: {2.0: glyph} for character in "128"},
            "hit_images": {"300": judgement},
            "hit_image_densities": {"300": 1.0},
        }
        definitions = layout_definitions(skin)
        self.assertEqual(definitions["playfield"]["size"], (646, 1080))
        self.assertEqual(definitions["combo"]["size"], (61, 81))
        self.assertEqual(definitions["judgement"]["size"], (70, 70))

    def test_hit_lighting_uses_additive_blending(self):
        lighting = np.zeros((10, 10, 4), dtype=np.uint8)
        lighting[:, :, 1] = 100
        lighting[:, :, 3] = 128
        lighting.flags.writeable = False
        skin = {
            "cfg": {"lighting_n_widths": None, "lighting_l_widths": None},
            "hit_lighting_normal": lighting,
            "hit_lighting_long": lighting,
            "hit_lighting_normal_density": 1.0,
            "hit_lighting_long_density": 1.0,
        }
        frame = np.full((40, 40, 3), 80, dtype=np.uint8)
        judgement = {"time": 100, "display_time": 100, "value": 300, "lane": 0, "kind": "tap"}
        draw_hit_lighting(frame, skin, [judgement], [100], 150, [10], [20], 20, 1.0, [])
        self.assertGreater(int(frame[20, 20, 1]), 80)
        self.assertEqual(int(frame[20, 20, 0]), 80)

    def test_motion_blur_covers_the_entire_frame(self):
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        frame[5, 1] = 255
        frame[18, 22] = 255
        apply_motion_blur(frame, 0, 0, 24, 24, 3)
        self.assertTrue(np.any(frame[2:9, 1] > 0))
        self.assertTrue(np.any(frame[15:22, 22] > 0))

if __name__ == "__main__":
    unittest.main()
