<div align="center">

# osu!mania Fast Renderer

**A GPU-first replay renderer for osu!mania `.osr` files.**

![Version](https://img.shields.io/badge/version-0.5.0v-7bd88f)
![Status](https://img.shields.io/badge/status-public%20alpha-ffb347)
![Windows](https://img.shields.io/badge/Windows-.exe-3572A5?logo=windows)
![Linux](https://img.shields.io/badge/Linux-AppImage-FCC624?logo=linux&logoColor=black)

[Download latest release](../../releases/latest) · [Usage](#usage) · [Skin support](#skin-support) · [Development](#development)

![Nico69 v4 showcase](docs/assets/showcase-nico69-v4.gif)

</div>

osu!mania Fast Renderer turns local osu!mania replays into skinned MP4 videos. It reads the replay, finds the matching beatmap, loads mania skin elements, renders gameplay with overlays, and adds the map audio.

This project is a public alpha. The renderer is usable, but skin compatibility and packaging will keep improving.

## Highlights

- Fast OpenGL-based frame generation.
- Hardware video encoding when available: VAAPI on Linux, AMF/NVENC/QSV on Windows when FFmpeg exposes them.
- 4K and 7K osu!mania rendering support.
- Dynamic legacy `skin.ini` parsing for receptors, notes, long notes, hit lighting, judgements, stage assets, combo/score fonts, and animations.
- Bundled verified skins:
  - ★ Nico69_ v4 — Verified [4k]
  - ★ Cawolo skin new Max — Verified [7k]
- Supported mods:
  - NoMod
  - Mirror
  - ScoreV2
  - DoubleTime / Nightcore
  - Nightcore pitch change
- DT/NC audio speed-up is applied during render.
- Automatic osu! folder detection on Windows, Linux, Wine, osu-wine, and Lutris setups.
- Fast replay list cache and beatmap lookup from replay hash.
- Optional side stats, key/BPM overlay, strain graph, timeline, results screen, and vignette.
- Update checks on startup.

## Usage

1. Open the app.
2. Select your osu! folder if it is not detected automatically.
3. Select an `.osr` replay.
4. Let the app find the beatmap, or select the `.osu` file manually.
5. Pick a skin. The verified bundled skins are always shown first.
6. Choose where to save the `.mp4`.
7. Render.

The app writes a `.debug.json` next to each render. It lists missing skin elements, encoder information, replay matching details, and other useful diagnostics.

## Skin support

The renderer targets legacy osu!mania skins and reads the `[Mania]` section from `skin.ini`.

Currently handled:

- `Keys`, `ColumnStart`, `HitPosition`, column widths, spacing, line widths, and lane colours;
- `KeyImage#`, `KeyImage#D`, `NoteImage#`, `NoteImage#H`, `NoteImage#L`, `NoteImage#T`;
- long-note body styles and very short LN edge cases;
- animated skin elements where the skin provides numbered frames;
- `Hit0`, `Hit50`, `Hit100`, `Hit200`, `Hit300`, `Hit300g`;
- stage sides, stage bottom, stage light, hit lighting, judgement line and stage hint;
- combo/score font digits and ranking assets.

The code has been tested with multiple 4K and 7K skins. Some of the compatibility work and optimization was done with AI-assisted testing and code iteration.

## Accuracy and combo

The renderer reconstructs osu!mania judgements from the replay input stream, then reconciles the final counters with the values stored inside the `.osr`.

Handled cases include:

- normal notes;
- long-note head/release logic;
- ScoreV1 and ScoreV2 combo differences;
- Mirror lane mapping;
- DT/NC timing and audio speed.

## Installation

Download the latest build from [Releases](../../releases/latest):

- Windows: `osu-mania-replay-renderer-*-Windows-x86_64.exe`
- Linux: `osu-mania-replay-renderer-*-Linux-x86_64.AppImage`

Linux:

```bash
chmod +x osu-mania-replay-renderer-*-Linux-x86_64.AppImage
./osu-mania-replay-renderer-*-Linux-x86_64.AppImage
```

## Development

Requirements:

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv)
- FFmpeg for local development builds

Run locally:

```bash
uv sync
uv run mania-renderer
```

Quick checks:

```bash
uv run python -m py_compile src/osu_mania_replay_renderer/*.py
uv run python -m osu_mania_replay_renderer --multiprocessing-smoke-test
```

Renderer structure:

- `renderer.py` — high-level render orchestration.
- `fast_gpu_renderer.py` — OpenGL frame generation.
- `renderer_media.py` — FFmpeg/audio/hardware encoder handling.
- `skin_loader.py` — dynamic skin parsing.
- `scoring.py` — replay judgement, combo, accuracy and ScoreV2 logic.

More details are in [docs/renderer_architecture.md](docs/renderer_architecture.md).

## Releases

GitHub Actions builds:

- Windows `.exe`
- Linux AppImage

Release `0.5.0v` is scheduled by workflow for July 12, 2026 at 13:00 UTC.

Manual publish from a clean tree:

```bash
python scripts/publish_release.py 0.5.0v
```

## Support

If this renderer helps you make osu!mania videos, you can support development on [Ko-fi](https://ko-fi.com/nico69yaza).

This tool is made by one person after many hours of testing and coding. A small donation really helps the project grow and helps me keep going.
