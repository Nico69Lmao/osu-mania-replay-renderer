from dataclasses import dataclass
import json
import os
import platform
import re
from urllib.request import Request, urlopen

from osu_mania_replay_renderer.version import __version__


GITHUB_REPOSITORY = "Nico69Lmao/osu-mania-replay-renderer"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest"


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    release_url: str
    download_url: str
    asset_name: str
    notes: str


def version_tuple(value):
    numbers = [int(part) for part in re.findall(r"\d+", str(value))[:3]]
    return tuple((numbers + [0, 0, 0])[:3])


def release_asset(assets):
    system = platform.system().lower()

    if system == "windows":
        suffix = ".exe"
    elif system == "linux":
        suffix = ".appimage"
    else:
        return None

    for asset in assets:
        name = str(asset.get("name", ""))

        if name.lower().endswith(suffix):
            return asset

    return None


def check_for_update(timeout=8):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": f"osu-mania-replay-renderer/{__version__}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("MANIA_RENDERER_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")

    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(
        LATEST_RELEASE_API,
        headers=headers,
    )

    with urlopen(request, timeout=timeout) as response:
        release = json.load(response)

    remote_version = str(release.get("tag_name", "")).lstrip("vV")

    if not remote_version or version_tuple(remote_version) <= version_tuple(__version__):
        return None

    asset = release_asset(release.get("assets", []))

    if asset is None:
        return None

    return UpdateInfo(
        version=remote_version,
        release_url=str(release.get("html_url", "")),
        download_url=str(asset.get("browser_download_url", "")),
        asset_name=str(asset.get("name", "")),
        notes=str(release.get("body", "")),
    )
