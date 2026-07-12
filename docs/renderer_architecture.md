# Renderer architecture

The renderer is intentionally split by responsibility:

- `gui.py` — desktop UI, settings, replay/beatmap selection, update checks, support popup.
- `renderer.py` — high-level render orchestration: parse inputs, build timing caches, prepare results screen, call the fast renderer, mux audio, write debug reports.
- `fast_gpu_renderer.py` — OpenGL frame generation path used by the app.
- `renderer_media.py` — FFmpeg discovery, audio filters, hardware encoder selection, silent-video encoding.
- `skin_loader.py` — legacy osu!mania `skin.ini` parsing and dynamic asset lookup, including animations and bundled verified skins.
- `scoring.py` — osu!mania replay judgement reconstruction, combo, accuracy, ScoreV1/ScoreV2 handling and replay count reconciliation.
- `osu_finder.py` and `osu_db_reader.py` — osu! folder discovery, fast replay/beatmap lookup, local `osu!.db` access.
- `replay_parser.py` and `beatmap_parser.py` — `.osr` input events and `.osu` beatmap parsing.

The public entry point remains `render_video(...)` in `renderer.py`.
Keep that function stable for GUI and packaging compatibility.
