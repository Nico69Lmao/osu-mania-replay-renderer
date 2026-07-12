#!/usr/bin/env python3
"""Create and push a release tag; GitHub Actions builds and publishes binaries."""

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
VERSION_FILE = ROOT / "src/osu_mania_replay_renderer/version.py"


def run(*args):
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    if len(sys.argv) != 2 or not re.fullmatch(r"v?\d+\.\d+\.\d+", sys.argv[1]):
        raise SystemExit("Usage: python scripts/publish_release.py X.Y.Z or vX.Y.Z")

    version = sys.argv[1].lstrip("vV")
    tag = f"v{version}"
    package_version = version
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    if status.strip():
        raise SystemExit("The working tree must be clean before publishing.")

    pyproject = PYPROJECT.read_text(encoding="utf-8")
    pyproject = re.sub(
        r'(?m)^(version\s*=\s*)"[^"]+"',
        rf'\g<1>"{package_version}"',
        pyproject,
        count=1,
    )
    PYPROJECT.write_text(pyproject, encoding="utf-8")
    VERSION_FILE.write_text(f'__version__ = "{package_version}"\n', encoding="utf-8")

    run("uv", "lock")
    run("git", "add", "pyproject.toml", "uv.lock", str(VERSION_FILE.relative_to(ROOT)))
    run("git", "commit", "-m", f"Release {tag}")
    run("git", "tag", "-a", tag, "-m", f"Release {tag}")
    run("git", "push", "origin", "HEAD", tag)
    print(f"{tag} pushed. GitHub Actions is building the .exe and AppImage.")


if __name__ == "__main__":
    main()
