<div align="center">

# osu!mania Replay Renderer

**Turn local `.osr` replays into synchronized, skin-accurate MP4 videos.**

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PySide6](https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white)
![GPU compositing](https://img.shields.io/badge/Compositing-OpenGL%20%2F%20EGL-5586A4)
![Platforms](https://img.shields.io/badge/Platforms-Windows%20%7C%20Linux-6C7A89)

[Download](../../releases/latest) · [Features](#features) · [Installation](#installation) · [Rendering Pipeline](#rendering-pipeline) · [Skin Compatibility](#skin-compatibility) · [Releases](#releases-and-remote-updates)

</div>

The renderer reads an osu! beatmap, an OSR replay, and a legacy osu! skin, then produces a synchronized video with skinned notes, long notes, receptors, hit lighting, judgements, statistics, audio, and a results screen.

> Public alpha: the renderer is usable, but the UI, defaults, skin compatibility, and release packages may still change quickly.

## At a Glance

| Skin Fidelity | Rendering Pipeline |
| --- | --- |
| Legacy `skin.ini`, `@2x` assets, LN parts, native fonts, hit lighting, stage and ranking elements | Parallel frame generation, OpenGL/EGL texture compositing, hardware encoding and CPU fallback |
| Replay Accuracy | Desktop Workflow |
| OD-dependent windows, OSR counter reconciliation, DT/NC/HT timing, star rating and PP display | Automatic osu! discovery, asynchronous beatmap lookup, ETA, cancellation and persistent Full HD layout editing |

## Features

- Supports native osu!mania beatmaps with key counts read from the `.osu` file.
- Parses legacy mania sections from `skin.ini`, including column geometry, hit position, notes, receptors, long-note parts, stage assets, and hit images.
- Renders long-note heads, bodies, tails, held states, and release judgements.
- Uses OD-dependent osu!mania hit windows and reconciles final judgement totals with the authoritative counts stored in the OSR replay.
- Displays combo, accuracy, judgement totals, star rating, estimated pp, a four-lane input/BPM visualizer, and a circular song timer.
- Draws the gameplay combo with the selected skin's `ComboPrefix`, `ComboOverlap`, and mania `ComboPosition`, while keeping the optional side overlay.
- Uses double-density combo glyphs correctly, reproduces the native increment animation, and can tint the combo from the configured mania `Hit300` artwork while a long note is held.
- Keeps only the latest gameplay judgement visible without animation; a new result replaces it immediately and inactivity clears it after one second.
- Shows a compact results header with the map title, mapper, player, and active mods.
- Draws a compact strain profile with completed sections in green and upcoming sections in grey.
- Supports DT, NC, and HT timing. NC also applies raised audio pitch.
- Includes optional temporal motion blur that affects changing gameplay pixels while leaving static text and overlays sharp.
- Draws `lightingN` and `lightingL` hit lighting from the selected mania skin with additive blending.
- Includes selectable side statistics, strain graph, vignette strength, results background opacity, results duration, and results-screen visibility.
- Produces a legacy-style results screen using the beatmap background and the selected skin's ranking and hit-result assets.
- Tries VAAPI, Intel QSV, and AMD AMF hardware encoding before falling back to `libx264`.
- Generates frames in parallel into batched MJPEG streams, avoiding thousands of temporary files and reducing disk overhead.
- Ships as a Windows `.exe` and Linux AppImage, with an automatic startup update check and a manual release check button.
- Detects common stable osu! installations automatically on Windows, Linux, osu-wine, Wine, and Lutris.
- Reads replay metadata and searches beatmaps in background threads with progress, cancellation, and ETA.
- Supports cancelling frame generation mid-render and removes partial frame streams and output files.
- Provides a persistent drag-and-drop layout editor with renderer-scale skin PNGs and representative gameplay overlay frames.
- Prefers the desktop file picker: GTK/Zenity or the XDG portal on Linux, KDE's KDialog, and Explorer on Windows.
- Uses batched OpenGL/EGL skin compositing when a GPU context is available, with automatic CPU fallback and VA-API, QSV, or AMF video encoding.

## Project Files

| File | Purpose |
| --- | --- |
| `src/osu_mania_replay_renderer/__main__.py` | Package entry point. Initializes multiprocessing support, creates the PySide6 application, and opens the main window. |
| `src/osu_mania_replay_renderer/gui.py` | Defines the desktop interface and background render thread. Handles osu! folder, replay, beatmap, skin, resolution, scroll speed, motion blur, output selection, and progress updates. |
| `src/osu_mania_replay_renderer/layout_editor.py` | Provides the drag-and-drop Full HD layout editor, normalized element coordinates, and the reset-to-skin-default action. |
| `src/osu_mania_replay_renderer/layout_model.py` | Calculates renderer-scale element dimensions in a logical `1920x1080` coordinate space without depending on Qt. |
| `src/osu_mania_replay_renderer/gpu_compositor.py` | Caches skin PNGs as GPU textures and performs batched scaling and alpha compositing through a headless OpenGL/EGL context. |
| `src/osu_mania_replay_renderer/renderer.py` | Core rendering engine. Calculates judgements, reconciles replay totals, draws gameplay and results frames, builds the strain graph, computes displayed pp, runs multiprocessing workers, invokes FFmpeg, and writes debug reports. |
| `src/osu_mania_replay_renderer/beatmap_parser.py` | Parses `.osu` metadata, audio and background paths, mode, key count, OD, normal notes, and long notes. Also computes the beatmap MD5 used for database lookup. |
| `src/osu_mania_replay_renderer/replay_parser.py` | Converts OSR replay frames into timestamped press and release events for each mania lane. |
| `src/osu_mania_replay_renderer/osu_finder.py` | Loads OSR files, decodes enabled mods, finds matching beatmaps by MD5, lists installed skins, and calculates the official aggregate mania accuracy stored by the replay. |
| `src/osu_mania_replay_renderer/skin_loader.py` | Parses the matching `[Mania]` block from `skin.ini`, resolves case-insensitive Windows-style asset paths on Linux, and loads gameplay, stage, judgement, and ranking assets with `@2x` density information. |
| `src/osu_mania_replay_renderer/osu_db_reader.py` | Reads cached osu!mania star ratings from the local `osu!.db`, matching the beatmap hash and replay mod combination. |
| `src/osu_mania_replay_renderer/settings.py` | Stores and loads persistent GUI preferences from `~/.config/mania-renderer/settings.json`. |
| `pyproject.toml` | Project metadata, dependencies, and `uv` configuration. |
| `.gitignore` | Excludes virtual environments, caches, rendered videos, replays, beatmaps, debug output, and temporary render data from Git. |
| `README.md` | Project documentation. |

## Rendering Pipeline

1. The GUI identifies the beatmap belonging to the selected replay by comparing MD5 hashes.
2. `beatmap_parser.py` loads metadata, OD, lane count, hit objects, audio, and background artwork.
3. `replay_parser.py` turns replay key states into per-lane input events.
4. `skin_loader.py` loads the selected legacy skin and its matching mania configuration.
5. `renderer.py` matches input events to notes using OD-dependent hit windows.
6. The simulated final judgement totals are reconciled with `count_geki`, `count_300`, `count_katu`, `count_100`, `count_50`, and `count_miss` from the OSR file.
7. Worker processes render frames in parallel. Visible notes and lane events are located with binary searches to avoid scanning the full map on every frame.
8. FFmpeg encodes the frame sequence, applies the correct DT, NC, or HT audio filter, and muxes the final MP4.
9. A `.debug.json` report is written beside the output video.

## Installation

Download the `.exe` or AppImage from the repository Releases page. FFmpeg is included in both packages; an installed system FFmpeg is preferred automatically when it provides hardware encoders.

To run the AppImage:

```bash
chmod +x osu-mania-replay-renderer-*-Linux-x86_64.AppImage
./osu-mania-replay-renderer-*-Linux-x86_64.AppImage
```

For development, Python 3.11 or newer and `uv` are required:

```bash
uv sync
```

Run the application with:

```bash
uv run mania-renderer
```

You can also run the package entry module directly:

```bash
uv run python -m osu_mania_replay_renderer
```

## Basic Usage

1. Confirm the automatically detected osu! installation, or select the folder containing `Songs`, `Skins`, and `osu!.db` manually.
2. Select an `.osr` replay. The matching beatmap is searched automatically.
3. Select the skin and rendering options.
4. Choose an MP4 output path and start the render.

The renderer supports common output resolutions from `640x360` through `3840x2160`.

## Skin Compatibility

The renderer reads the selected key-count block from `skin.ini`, including:

- `ColumnWidth`, `ColumnSpacing`, `ColumnLineWidth`, and `HitPosition`
- `KeyImage`, `KeyImageD`
- `NoteImage`, `NoteImageH`, `NoteImageL`, `NoteImageT`
- `Hit0`, `Hit50`, `Hit100`, `Hit200`, `Hit300`, `Hit300g`
- `StageLeft`, `StageRight`, `StageBottom`, `StageLight`, and `StageHint`
- `ComboPrefix`, `ComboOverlap`, `ScorePrefix`, and `ScoreOverlap`

The results screen follows the documented legacy v2 layout in a logical `1024x768` coordinate space. It uses `ranking-panel`, `ranking-{grade}`, `ranking-maxcombo`, `ranking-accuracy`, `ranking-graph`, `ranking-perfect`, and `ranking-title` when provided. Judgement counters use the selected mania block's `Hit300g`, `Hit300`, `Hit200`, `Hit100`, `Hit50`, and `Hit0` artwork so they match gameplay rather than the standard-mode hit bursts.

Navigation controls such as Back, retry, replay, and Online Ranking are intentionally not rendered.

## Timing and Mods

The visual scroll time follows osu!'s mania implementation:

```text
scroll_time_ms = max(290, 11485 / scroll_speed)
```

- `DT`: video and audio at `1.5x`
- `NC`: video at `1.5x`, audio accelerated with raised pitch
- `HT`: video and audio at `0.75x`

## Accuracy and PP

Native mania hit windows depend on beatmap OD. Replay timestamps are used to assign the most plausible result to each object, while the OSR judgement counters guarantee that final `300g`, `300`, `200`, `100`, `50`, and miss totals match the recorded play.

The results grade follows the official osu!mania accuracy thresholds: SS at 100%, S above 95%, A above 90%, B above 80%, C above 70%, and D otherwise. SS and S use their silver variants when Hidden, Flashlight, or Fade In is enabled.

Displayed pp uses the official osu!mania performance formula when a matching star rating is available in `osu!.db`:

```text
pp = 8 * max(star_rating - 0.15, 0.05)^2.2
     * max(0, 5 * custom_accuracy - 4)
     * (1 + 0.1 * min(1, total_hits / 1500))
```

If no matching star rating is found, the renderer displays `pp: N/A` rather than inventing a value.

## Hardware Encoding

The encoder order is:

1. H.264 VAAPI
2. Intel H.264 QSV
3. AMD H.264 AMF
4. CPU `libx264`

The selected encoder and every failed attempt are recorded in the debug report.

## Releases and Remote Updates

Pushing a version tag such as `v0.2.0` starts the GitHub Actions release workflow. Windows builds the native one-file `.exe`, Linux builds the AppImage, and both files are attached to the GitHub release automatically.

For later versions, start from a clean working tree and run:

```bash
python scripts/publish_release.py 0.3.1
```

This updates the package version, refreshes `uv.lock`, commits, creates the tag, and pushes it. On startup, the desktop app checks GitHub Releases in the background. If a newer compatible `.exe` or AppImage is available, it asks the user whether they want to open the release page and download it.

## Debug Report

Each render creates `<video-name>.debug.json` containing:

- Replay and output paths
- Enabled mods and speed multiplier
- Automatic timing offset
- Recorded and simulated accuracy
- Reconciled judgement totals
- OD and hit windows
- Scroll speed and note travel time
- Star rating
- Selected FFmpeg encoder
- Errors from failed hardware encoder attempts
- First non-perfect simulated judgements

## Technical References

- [osu!mania scroll-time implementation](https://github.com/ppy/osu/blob/master/osu.Game.Rulesets.Mania/UI/DrawableManiaRuleset.cs)
- [Legacy mania skin configuration](https://github.com/ppy/osu/blob/master/osu.Game/Skinning/LegacyManiaSkinConfiguration.cs)
- [osu!mania performance calculator](https://github.com/ppy/osu/blob/master/osu.Game.Rulesets.Mania/Difficulty/ManiaPerformanceCalculator.cs)
- [Official grade requirements](https://osu.ppy.sh/wiki/en/Gameplay/Grade#osu!mania)
- [Legacy ranking-screen skin elements](https://osu.ppy.sh/wiki/en/Skinning/Interface#ranking-screen)
- [osu!mania ranking-screen hit hierarchy](https://osu.ppy.sh/wiki/en/Skinning/FAQ#ranking-screen-hit-score-hierarchy)
