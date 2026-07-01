<div align="center">

# osu!mania Replay Renderer

**Render osu!mania `.osr` replays into skinned MP4 videos.**

![Public alpha](https://img.shields.io/badge/status-public%20alpha-ffb347)
![Windows](https://img.shields.io/badge/Windows-.exe-3572A5?logo=windows)
![Linux](https://img.shields.io/badge/Linux-AppImage-FCC624?logo=linux&logoColor=black)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)

[Download latest release](../../releases/latest) · [Features](#features) · [Usage](#usage) · [Development](#development)

</div>

osu!mania Replay Renderer is a desktop app that takes a replay, finds the matching beatmap, loads your osu! skin, and renders a video that looks close to the in-game replay/results screen.

> Public alpha: the renderer is already usable, but visuals, packaging, settings, and skin support may still change quickly.

## Features

- Renders osu!mania replays to MP4 with audio sync.
- Uses legacy osu! skins: notes, receptors, long notes, hit lighting, combo digits, judgements, ranking assets, and `@2x` files.
- Supports long notes, release judgements, DT, NC, HT, Mirror, multiple key counts, scroll speed, and common render resolutions up to 4K.
- Matches replay judgement totals from the `.osr`, so the final results screen stays consistent with the recorded play.
- Includes a skinned results screen with rank, accuracy, max combo, judgement counts, mods, map title, player name, and estimated pp when star rating data is available.
- Shows optional overlays such as side stats, key/input BPM graph, strain graph, circular timer, vignette, and motion blur.
- Finds osu! installations automatically on Windows, Linux, Wine, osu-wine, and Lutris.
- Finds the matching beatmap automatically by replay hash.
- Lets you edit overlay positions with a persistent drag-and-drop layout editor.
- Generates frames in parallel and can use OpenGL/EGL GPU compositing when available.
- Tries hardware H.264 encoding first, then falls back to CPU encoding.
- Ships as a Windows `.exe` and Linux AppImage.
- Checks for updates on startup and asks before opening the new release download page.

## Installation

Download the latest package from [Releases](../../releases/latest):

- Windows: `osu-mania-replay-renderer-*-Windows-x86_64.exe`
- Linux: `osu-mania-replay-renderer-*-Linux-x86_64.AppImage`

On Linux:

```bash
chmod +x osu-mania-replay-renderer-*-Linux-x86_64.AppImage
./osu-mania-replay-renderer-*-Linux-x86_64.AppImage
```

FFmpeg is bundled with the release builds. If a better system FFmpeg is installed, the app can use it automatically.

## Usage

1. Open the app.
2. Confirm the detected osu! folder, or select the folder containing `Songs`, `Skins`, and `osu!.db`.
3. Select an `.osr` replay.
4. Pick a skin and render options.
5. Choose the output `.mp4` path.
6. Start rendering.

The app writes a debug JSON next to the rendered video. It is useful when a skin, replay, encoder, or beatmap lookup behaves strangely.

## Skin support

The renderer focuses on legacy osu!mania skins and reads the matching `[Mania]` block from `skin.ini`, including:

- column widths, spacing, line widths, and hit position;
- key images, notes, receptors, long-note body/head/tail pieces;
- `Hit0`, `Hit50`, `Hit100`, `Hit200`, `Hit300`, and `Hit300g`;
- stage assets, hit lighting, combo font, score font, ranking panel, and rank images.

Some osu! UI-only elements are intentionally not drawn yet, such as replay navigation buttons and online ranking controls.

## Accuracy, combo, and pp

Judgements use osu!mania OD-based hit windows and replay timestamps. The final judgement counters are reconciled with the values stored inside the `.osr`, which keeps the rendered result aligned with the actual replay.

Displayed pp is estimated from local osu! star rating data when the matching value exists in `osu!.db`. If the renderer cannot find a matching star rating, it shows `pp: N/A`.

## Development

Requirements:

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv)

Install dependencies:

```bash
uv sync
```

Run the app:

```bash
uv run mania-renderer
```

Run quick checks:

```bash
uv run python -m compileall -q src scripts
uv run python -m osu_mania_replay_renderer --multiprocessing-smoke-test
```

## Releases

Release builds are made by GitHub Actions. A version tag builds:

- a Windows one-file `.exe`;
- a Linux AppImage;
- a GitHub release containing both files.

From a clean working tree:

```bash
python scripts/publish_release.py 0.4.1
```

The app checks GitHub Releases on startup. If a newer compatible build exists, it asks the user whether to open the release page.

## References

- [Legacy mania skin configuration](https://github.com/ppy/osu/blob/master/osu.Game/Skinning/LegacyManiaSkinConfiguration.cs)
- [osu!mania performance calculator](https://github.com/ppy/osu/blob/master/osu.Game.Rulesets.Mania/Difficulty/ManiaPerformanceCalculator.cs)
- [osu!mania grade requirements](https://osu.ppy.sh/wiki/en/Gameplay/Grade#osu!mania)
- [Ranking screen skin elements](https://osu.ppy.sh/wiki/en/Skinning/Interface#ranking-screen)
