from osu_finder import get_replay


def get_key_bits(keys_obj):
    try:
        return int(keys_obj.value)
    except Exception:
        try:
            return int(keys_obj)
        except Exception:
            return 0


def get_replay_events(replay_path: str, keys: int):
    replay = get_replay(replay_path)

    events = []
    current_time = 0.0
    last_state = [False] * keys

    for frame in replay.replay_data:
        current_time += float(frame.time_delta)
        key_bits = get_key_bits(frame.keys)

        state = []
        for lane in range(keys):
            state.append(bool(key_bits & (1 << lane)))

        for lane in range(keys):
            if state[lane] != last_state[lane]:
                events.append({
                    "time": int(round(current_time)),
                    "lane": lane,
                    "pressed": state[lane],
                })

        last_state = state

    return events
