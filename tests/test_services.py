import io
import json
from hashlib import md5
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from osu_mania_replay_renderer import updater
from osu_mania_replay_renderer.osu_finder import find_beatmap_by_hash, find_osu_folder


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


if __name__ == "__main__":
    unittest.main()
