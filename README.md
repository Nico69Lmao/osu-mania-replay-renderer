# osu!mania Local Renderer

Renderer locale per replay osu!mania. Legge una beatmap `.osu`, un replay `.osr` e una skin mania, poi genera un video MP4 con note, long notes, receptor, judgement, counter e audio sincronizzato.

## Funzioni principali

- Rendering osu!mania 4K e altre key count lette dal file `.osu`.
- Skin mania da `skin.ini`:
  - `ColumnWidth`, `ColumnSpacing`, `HitPosition`, `ScorePosition`, `ComboPosition`
  - `KeyImage`, `KeyImageD`
  - `NoteImage`, `NoteImageH`, `NoteImageL`, `NoteImageT`
  - `Hit0`, `Hit50`, `Hit100`, `Hit200`, `Hit300`, `Hit300g`
  - stage/lane cover se l'asset esiste e ha alpha visibile
- Long notes con body, head tenuta durante l'hold e release judgement.
- Accuracy dinamica basata su OD e riconciliata con i conteggi ufficiali salvati nel replay OSR.
- Counter per judgement: `300g`, `300`, `200`, `100`, `50`, `Miss`.
- BPM dinamici per ciascun tasto, calcolati sulle pressioni degli ultimi due secondi.
- PP counter con formula ufficiale osu!mania quando e disponibile una star rating; altrimenti mostra `pp: N/A`.
- Star rating letta dalla cache `osu!.db` quando disponibile, scegliendo la voce mania per i mod del replay.
- Indicatore temporale circolare grigio sotto le statistiche a destra.
- Supporto mod speed:
  - DT: video e audio a `1.5x`
  - NC: video a `1.5x`, audio con pitch alto tramite `asetrate`
  - HT: video e audio a `0.75x`
- Scroll speed basata sul sorgente ufficiale ppy/osu:
  - `scroll_time_ms = 11485 / scroll_speed`
  - clamp minimo `290ms`, range GUI `1..40`
- Motion blur opzionale sul playfield.
- Encoding video con accelerazione hardware automatica:
  - VAAPI, utile per Intel/AMD su Linux
  - Intel QSV
  - AMD AMF
  - fallback CPU `libx264`

## Come funziona il rendering

1. `beatmap_parser.py` legge metadata, audio, key count, OD e hitobjects dalla beatmap.
2. `replay_parser.py` legge gli input del replay e li converte in eventi press/release per lane.
3. `skin_loader.py` legge il blocco `[Mania]` corretto dal `skin.ini` selezionato e carica le immagini della skin.
4. `renderer.py` calcola i judgement:
   - le finestre mania native usano OD: `300 = 64 - 3*OD`, `200 = 97 - 3*OD`, ecc.
   - le LN usano head/release per visual, ma il risultato che conta viene risolto al release.
   - i timing ordinano i risultati per nota, mentre `count_geki`, `count_300`, `count_katu`, `count_100`, `count_50` e `count_miss` dell'OSR garantiscono conteggi finali identici a osu!stable.
5. Ogni frame viene generato in parallelo:
   - calcolo `map_time`
   - note visibili tramite binary search
   - eventi press/release indicizzati per lane
   - rendering playfield, note, LN, cover, receptor, UI
6. FFmpeg crea un video temporaneo e poi aggiunge l'audio con il filtro corretto per DT/NC/HT.

## Avvio

```bash
python main.py
```

Se usi il virtualenv incluso:

```bash
./venv/bin/python main.py
```

## Output debug

Per ogni render viene scritto anche un file `.debug.json` accanto al video. Contiene:

- mods e speed multiplier
- offset scelto
- accuracy replay e simulata
- OD e hit windows usate
- scroll speed e travel time
- encoder ffmpeg usato
- esito ed errore di ogni tentativo VAAPI/QSV/AMF/libx264
- conteggi judgement simulati e conferma della riconciliazione OSR
- star rating letta da `osu!.db`
- primi judgement non perfetti

## Accelerazione Intel su Arch/EndeavourOS

Per una Intel Iris Xe servono il driver VAAPI Intel e gli strumenti di verifica:

```bash
sudo pacman -S --needed intel-media-driver libva-utils
```

Verifica il driver con `vainfo` e monitora l'encoding con `sudo intel_gpu_top`. Nel file `.debug.json`, `video_encoder` deve contenere `h264_vaapi`; se VAAPI fallisce, `video_encoder_attempts` conserva l'errore e il renderer passa automaticamente all'encoder successivo.

## Note sui PP

La formula pp mania ufficiale richiede la star rating della mappa. La star rating non e contenuta nel file `.osu`; il renderer prova a leggerla da `osu!.db`, dove osu!stable cache-a le star ratings per ruleset e combinazione mod. Se non viene trovata, il renderer non inventa pp e mostra `pp: N/A`.

Formula usata quando la SR e disponibile:

```text
pp = 8 * max(star_rating - 0.15, 0.05)^2.2
     * max(0, 5 * custom_accuracy - 4)
     * (1 + 0.1 * min(1, total_hits / 1500))
```

`custom_accuracy`:

```text
(perfect*320 + great*300 + good*200 + ok*100 + meh*50) / (total_hits*320)
```

## Fonti

- osu!mania scroll speed, `DrawableManiaRuleset.ComputeScrollTime()`
  https://github.com/ppy/osu/blob/master/osu.Game.Rulesets.Mania/UI/DrawableManiaRuleset.cs
- legacy mania hit position conversion
  https://github.com/ppy/osu/blob/master/osu.Game/Skinning/LegacyManiaSkinConfiguration.cs
- osu!mania performance calculator
  https://github.com/ppy/osu/blob/master/osu.Game.Rulesets.Mania/Difficulty/ManiaPerformanceCalculator.cs
