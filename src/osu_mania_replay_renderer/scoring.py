from bisect import bisect_left, bisect_right


def mania_hit_windows(overall_difficulty, is_convert=False):
    if is_convert:
        return {
            "perfect": 16,
            "great": 34 if overall_difficulty > 4 else 47,
            "good": 67 if overall_difficulty > 4 else 77,
            "ok": 97,
            "meh": 121,
            "miss": 158,
        }

    return {
        "perfect": 16,
        "great": int(64 - 3 * overall_difficulty),
        "good": int(97 - 3 * overall_difficulty),
        "ok": int(127 - 3 * overall_difficulty),
        "meh": int(151 - 3 * overall_difficulty),
        "miss": int(188 - 3 * overall_difficulty),
    }


def mania_score_value(diff, windows):
    if diff <= windows["great"]:
        return 300
    if diff <= windows["good"]:
        return 200
    if diff <= windows["ok"]:
        return 100
    if diff <= windows["meh"]:
        return 50
    return 0


def hit_image_key(value, diff=None):
    if value == 300 and diff is not None and diff <= 16:
        return "300g"
    if value >= 300:
        return "300"
    return str(value)


def replay_judgement_counts(replay):
    return {
        "300g": int(getattr(replay, "count_geki", 0)),
        "300": int(getattr(replay, "count_300", 0)),
        "200": int(getattr(replay, "count_katu", 0)),
        "100": int(getattr(replay, "count_100", 0)),
        "50": int(getattr(replay, "count_50", 0)),
        "0": int(getattr(replay, "count_miss", 0)),
    }


def mania_accuracy_from_counts(counts, score_v2=False):
    total = sum(counts.values())

    if total <= 0:
        return 100.0

    max_value = 305 if score_v2 else 300
    perfect_value = 305 if score_v2 else 300
    score_sum = (
        counts.get("50", 0) * 50
        + counts.get("100", 0) * 100
        + counts.get("200", 0) * 200
        + counts.get("300", 0) * 300
        + counts.get("300g", 0) * perfect_value
    )
    return score_sum / (max_value * total) * 100


def stable_mania_accuracy_from_replay(replay, score_v2=False):
    return mania_accuracy_from_counts(replay_judgement_counts(replay), score_v2)


def judgement_counts(judgements):
    counts = {key: 0 for key in ("300g", "300", "200", "100", "50", "0")}

    for judgement in judgements:
        if judgement.get("counts_accuracy", True):
            key = judgement.get("image_key") or hit_image_key(judgement["value"], judgement.get("diff"))
            counts[key] = counts.get(key, 0) + 1

    return counts


def recompute_judgement_state(judgements):
    """Rebuild combo and accuracy in display order, including stable LN ticks."""
    combo = 0
    score_sum = 0
    max_score_sum = 0
    score_v2 = any(j.get("score_v2") for j in judgements)

    judgements.sort(key=lambda j: (j.get("display_time", j["time"]), j.get("time", 0)))

    for judgement in judgements:
        if judgement.get("counts_accuracy", True):
            value = judgement["value"]
            key = judgement.get("image_key") or hit_image_key(value, judgement.get("diff"))
            score_sum += 305 if score_v2 and key == "300g" else value
            max_score_sum += 305 if score_v2 else 300

        if judgement.get("combo_break", False):
            combo = 0
        else:
            combo += int(judgement.get("combo_delta", 0))

        judgement["combo"] = combo
        judgement["accuracy"] = (score_sum / max_score_sum) * 100 if max_score_sum else 100.0

    return combo


def build_event_lanes(events, keys):
    """Index physical replay key states without re-applying Mirror."""
    indexed = [[] for _ in range(keys)]

    for event in events:
        lane = event["lane"]

        if 0 <= lane < keys:
            indexed[lane].append((event["time"], event["pressed"]))

    return [
        ([event_time for event_time, _ in lane], [pressed for _, pressed in lane])
        for lane in indexed
    ]


def add_stable_ln_ticks(judgements, notes, events, keys, mirror):
    """Add stable's 100 ms hold-note combo ticks when the lane is held."""
    judgements[:] = [j for j in judgements if j.get("kind") != "ln_tick"]
    heads = {
        j.get("note_id"): j
        for j in judgements
        if j.get("kind") == "ln_head"
    }
    lane_events = build_event_lanes(events, keys)

    for note_id, note in enumerate(notes):
        if note.get("end_time") is None:
            continue

        lane = keys - 1 - note["lane"] if mirror else note["lane"]
        head = heads.get(note_id)

        if head is None or head.get("value", 0) <= 0 or not (0 <= lane < keys):
            continue

        times, states = lane_events[lane]
        tick_time = note["time"] + 100

        while tick_time < note["end_time"]:
            event_i = bisect_right(times, tick_time) - 1

            if event_i >= 0 and states[event_i]:
                judgements.append({
                    "time": tick_time,
                    "display_time": tick_time,
                    "lane": lane,
                    "kind": "ln_tick",
                    "note_id": note_id,
                    "combo": 0,
                    "accuracy": 100.0,
                    "value": 0,
                    "diff": None,
                    "image_key": None,
                    "counts_accuracy": False,
                    "combo_delta": 1,
                    "combo_break": False,
                })

            tick_time += 100

    recompute_judgement_state(judgements)


def reconcile_judgements_with_replay(judgements, target_counts):
    """Match stable's aggregate OSR results while retaining timing-based per-note ordering."""
    scoring = [j for j in judgements if j.get("counts_accuracy", True)]

    if len(scoring) != sum(target_counts.values()):
        return False

    ranked = sorted(
        scoring,
        key=lambda j: (
            j.get("diff") is None,
            j.get("diff") if j.get("diff") is not None else float("inf"),
            j["time"],
        ),
    )
    cursor = 0

    for key in ("300g", "300", "200", "100", "50", "0"):
        value = 300 if key in ("300g", "300") else int(key)

        for judgement in ranked[cursor:cursor + target_counts[key]]:
            judgement["value"] = value
            judgement["image_key"] = key

        cursor += target_counts[key]

    for judgement in judgements:
        if judgement.get("counts_accuracy", True):
            value = judgement["value"]
            judgement["combo_break"] = value <= 0
            if judgement.get("kind") in ("tap", "ln_head") or (judgement.get("kind") == "ln_tail" and judgement.get("score_v2")):
                judgement["combo_delta"] = 1 if value > 0 else 0
            elif judgement.get("kind") == "ln_tail":
                judgement["combo_delta"] = 0

    recompute_judgement_state(judgements)

    return True


def build_judgements(notes, events, keys, mirror, overall_difficulty=5.0, is_convert=False, score_v2=False):
    windows = mania_hit_windows(overall_difficulty, is_convert)
    press_by_lane = [[] for _ in range(keys)]
    release_by_lane = [[] for _ in range(keys)]

    for event in events:
        lane = event["lane"]

        if 0 <= lane < keys:
            (press_by_lane if event["pressed"] else release_by_lane)[lane].append(event)

    press_times_by_lane = []
    release_times_by_lane = []

    for lane in range(keys):
        press_by_lane[lane].sort(key=lambda event: event["time"])
        release_by_lane[lane].sort(key=lambda event: event["time"])
        press_times_by_lane.append([event["time"] for event in press_by_lane[lane]])
        release_times_by_lane.append([event["time"] for event in release_by_lane[lane]])

    used_press = [set() for _ in range(keys)]
    used_release = [set() for _ in range(keys)]

    objects = []

    for note_id, note in enumerate(notes):
        lane = keys - 1 - note["lane"] if mirror else note["lane"]
        is_ln = note["end_time"] is not None

        objects.append({
            "time": note["time"],
            "lane": lane,
            "kind": "tap" if not is_ln else "ln_head",
            "note_id": note_id,
            "end_time": note["end_time"],
        })

        if is_ln:
            objects.append({
                "time": note["end_time"],
                "lane": lane,
                "kind": "ln_tail",
                "note_id": note_id,
                "start_time": note["time"],
            })

    objects.sort(key=lambda x: x["time"])

    results = []
    debug_bad = []

    score_sum = 0
    max_score_sum = 0
    lane_holding = [False] * keys
    ln_heads = {}

    for obj in objects:
        value = 0
        diff_used = None
        event_time = None

        counts_accuracy = score_v2 or obj["kind"] != "ln_head"

        if obj["kind"] in ("tap", "ln_head"):
            best = None
            best_i = None
            lane_presses = press_by_lane[obj["lane"]]
            lane_press_times = press_times_by_lane[obj["lane"]]
            left = bisect_left(lane_press_times, obj["time"] - windows["miss"])
            right = bisect_right(lane_press_times, obj["time"] + windows["miss"])

            for i in range(left, right):
                if i in used_press[obj["lane"]]:
                    continue

                event = lane_presses[i]
                diff = abs(event["time"] - obj["time"])

                if diff <= windows["miss"] and (best is None or diff < best):
                    best = diff
                    best_i = i

            if best_i is not None:
                used_press[obj["lane"]].add(best_i)
                lane_holding[obj["lane"]] = True
                diff_used = best
                event_time = lane_presses[best_i]["time"]
                value = mania_score_value(best, windows)

                if obj["kind"] == "ln_head":
                    ln_heads[obj["note_id"]] = {
                        "diff": best,
                        "value": value,
                    }

        elif obj["kind"] == "ln_tail":
            best = None
            best_i = None
            lane_releases = release_by_lane[obj["lane"]]
            lane_release_times = release_times_by_lane[obj["lane"]]
            left = bisect_left(lane_release_times, obj["time"] - windows["miss"])
            right = bisect_right(lane_release_times, obj["time"] + windows["miss"])

            for i in range(left, right):
                if i in used_release[obj["lane"]]:
                    continue

                event = lane_releases[i]
                diff = abs(event["time"] - obj["time"])

                if diff <= windows["miss"] and (best is None or diff < best):
                    best = diff
                    best_i = i

            head = ln_heads.get(obj["note_id"])

            if best_i is not None and head is not None and head["value"] > 0:
                used_release[obj["lane"]].add(best_i)
                lane_holding[obj["lane"]] = False
                diff_used = best
                event_time = lane_releases[best_i]["time"]

                combined = head["diff"] + best

                if head["diff"] <= windows["perfect"] * 1.2 and combined <= windows["perfect"] * 2.4:
                    value = 300
                    diff_used = max(head["diff"], best)
                elif head["diff"] <= windows["great"] * 1.1 and combined <= windows["great"] * 2.2:
                    value = 300
                    diff_used = max(head["diff"], best, windows["perfect"] + 1)
                elif head["diff"] <= windows["good"] and combined <= windows["good"] * 2:
                    value = 200
                elif head["diff"] <= windows["ok"] and combined <= windows["ok"] * 2:
                    value = 100
                else:
                    value = 50

            lane_holding[obj["lane"]] = False

        if counts_accuracy:
            key = hit_image_key(value, diff_used)
            score_sum += 305 if score_v2 and key == "300g" else value
            max_score_sum += 305 if score_v2 else 300

        accuracy = (score_sum / max_score_sum) * 100 if max_score_sum else 100.0
        combo_counts = obj["kind"] in ("tap", "ln_head") or (score_v2 and obj["kind"] == "ln_tail")

        result = {
            "time": obj["time"],
            "lane": obj["lane"],
            "kind": obj["kind"],
            "note_id": obj["note_id"],
            "combo": 0,
            "accuracy": accuracy,
            "value": value,
            "diff": diff_used,
            "image_key": hit_image_key(value, diff_used),
            "display_time": event_time if event_time is not None else obj["time"],
            "hit_time": event_time,
            "counts_accuracy": counts_accuracy,
            "score_v2": score_v2,
            "combo_delta": 1 if combo_counts and value > 0 else 0,
            "combo_break": value <= 0,
        }

        results.append(result)

        if counts_accuracy and value < 300 and len(debug_bad) < 100:
            debug_bad.append(result)

    if not score_v2:
        add_stable_ln_ticks(results, notes, events, keys, mirror)
    else:
        recompute_judgement_state(results)
    return results, debug_bad


def find_best_offset(notes, events, keys, mirror, overall_difficulty=5.0, is_convert=False, target_counts=None, score_v2=False):
    best_offset = 0
    best_score = None
    best_judgements = []
    best_bad = []

    def score_offset(offset):
        shifted_events = [
            {
                "time": e["time"] + offset,
                "lane": e["lane"],
                "pressed": e["pressed"],
            }
            for e in events
        ]

        test_judgements, test_bad = build_judgements(notes, shifted_events, keys, mirror, overall_difficulty, is_convert, score_v2)
        test_accuracy = test_judgements[-1]["accuracy"] if test_judgements else 100.0
        scoring = [j for j in test_judgements if j.get("counts_accuracy", True)]
        misses = sum(1 for j in scoring if j["value"] == 0)
        timing_error = sum((j["diff"] or 180) for j in scoring)
        count_error = 0

        if target_counts:
            test_counts = judgement_counts(test_judgements)
            count_error = sum(abs(test_counts[key] - target_counts[key]) for key in target_counts)

        score = (count_error, -test_accuracy, misses, timing_error)

        return score, test_judgements, test_bad

    coarse_offsets = range(-100, 101, 5)
    for offset in coarse_offsets:
        score, test_judgements, test_bad = score_offset(offset)
        if best_score is None or score < best_score:
            best_score = score
            best_offset = offset
            best_judgements = test_judgements
            best_bad = test_bad

    for offset in range(max(-100, best_offset - 4), min(100, best_offset + 4) + 1):
        score, test_judgements, test_bad = score_offset(offset)
        if score < best_score:
            best_score = score
            best_offset = offset
            best_judgements = test_judgements
            best_bad = test_bad

    shifted_events = [
        {
            "time": e["time"] + best_offset,
            "lane": e["lane"],
            "pressed": e["pressed"],
        }
        for e in events
    ]

    return best_offset, shifted_events, best_judgements, best_bad
