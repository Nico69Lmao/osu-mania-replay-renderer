from dataclasses import dataclass
import json
import os
import platform
import re
import shutil
import subprocess
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from osu_mania_replay_renderer.version import __version__


GITHUB_REPOSITORY = "Nico69Lmao/osu-mania-replay-renderer"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest"
RELEASES_URL = f"https://github.com/{GITHUB_REPOSITORY}/releases"


class UpdateCheckError(RuntimeError):
    pass


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


def github_token():
    token = os.environ.get("MANIA_RENDERER_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")

    if token:
        return token.strip()

    gh = shutil.which("gh")

    if not gh:
        return None

    try:
        result = subprocess.run(
            [gh, "auth", "token"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def check_for_update(timeout=8):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": f"osu-mania-replay-renderer/{__version__}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = github_token()

    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(
        LATEST_RELEASE_API,
        headers=headers,
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            release = json.load(response)
    except HTTPError as error:
        if error.code == 404 and not token:
            raise UpdateCheckError(
                "GitHub could not find the release feed. The repository may be unavailable or private."
            ) from error

        if error.code in (403, 429):
            remaining = error.headers.get("X-RateLimit-Remaining")
            message = "GitHub API rate limit reached." if remaining == "0" else "GitHub refused the update request."
            raise UpdateCheckError(message) from error

        raise UpdateCheckError(f"GitHub update request failed with HTTP {error.code}.") from error
    except URLError as error:
        reason = getattr(error, "reason", error)
        raise UpdateCheckError(f"Could not connect to GitHub: {reason}") from error
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise UpdateCheckError(f"GitHub returned an invalid update response: {error}") from error

    if not isinstance(release, dict):
        raise UpdateCheckError("GitHub returned an invalid release response.")

    remote_version = str(release.get("tag_name", "")).lstrip("vV")

    if not remote_version:
        raise UpdateCheckError("The latest GitHub release does not contain a valid version tag.")

    if version_tuple(remote_version) <= version_tuple(__version__):
        return None

    asset = release_asset(release.get("assets", []))

    if asset is None:
        raise UpdateCheckError(
            f"Version {remote_version} exists, but it has no compatible asset for {platform.system()}."
        )

    return UpdateInfo(
        version=remote_version,
        release_url=str(release.get("html_url", "")),
        download_url=str(asset.get("browser_download_url", "")),
        asset_name=str(asset.get("name", "")),
        notes=str(release.get("body", "")),
    )
