SCENE_WIDTH = 1920
SCENE_HEIGHT = 1080
SKIN_SCALE = SCENE_HEIGHT / 480.0
LAYOUT_POSITIONS = {
    "playfield": (0.50, 0.50),
    "combo": (0.50, 0.25),
    "judgement": (0.50, 0.37),
    "side_stats": (0.95, 0.16),
    "key_input": (0.95, 0.46),
    "timeline": (0.95, 0.64),
    "strain_graph": (0.87, 0.97),
    "star_rating": (0.04, 0.82),
}
DEFAULT_SIZES = {
    "playfield": (450, 1080),
    "combo": (102, 102),
    "judgement": (36, 36),
    "side_stats": (180, 300),
    "key_input": (174, 300),
    "timeline": (174, 60),
    "strain_graph": (500, 66),
    "star_rating": (120, 36),
}


def meaningful_image(image):
    if image is None:
        return False

    if image.ndim < 3 or image.shape[2] < 4:
        return True

    return int((image[:, :, 3] > 8).sum()) >= 16


def logical_glyph_metrics(glyphs, text, scale, overlap):
    metrics = []

    for character in text:
        variants = glyphs.get(character, {})

        if not variants:
            return None

        density = max(variants)
        image = variants[density]
        metrics.append((
            image,
            max(1, int(image.shape[1] * scale / density)),
            max(1, int(image.shape[0] * scale / density)),
        ))

    overlap_px = int(overlap * scale)
    width = sum(item[1] for item in metrics) - overlap_px * max(0, len(metrics) - 1)
    height = max(item[2] for item in metrics)
    return metrics, max(1, width), max(1, height), overlap_px


def visible_glyph_metrics(glyphs, text, scale, overlap):
    metrics = []

    for character in text:
        variants = glyphs.get(character, {})

        if not variants:
            return None

        density = max(variants)
        image = variants[density]

        if image.ndim == 3 and image.shape[2] == 4:
            ys, xs = (image[:, :, 3] > 8).nonzero()

            if len(xs):
                image = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        metrics.append((
            image,
            max(1, int(image.shape[1] * scale / density)),
            max(1, int(image.shape[0] * scale / density)),
        ))

    overlap_px = int(overlap * scale)
    width = sum(item[1] for item in metrics) - overlap_px * max(0, len(metrics) - 1)
    height = max(item[2] for item in metrics)
    return metrics, max(1, width), max(1, height), overlap_px


def layout_definitions(skin):
    cfg = skin.get("cfg", {})
    keys = max(1, len(skin.get("keys", [])))
    column_widths = cfg.get("column_widths") or [70] * keys
    column_spacing = cfg.get("column_spacing") or [0] * (keys - 1)
    scaled_widths = [int(width * SKIN_SCALE) for width in column_widths]
    scaled_spacing = [int(spacing * SKIN_SCALE) for spacing in column_spacing]
    playfield_width = max(1, sum(scaled_widths) + sum(scaled_spacing))

    combo_metrics = visible_glyph_metrics(
        skin.get("combo_glyphs", {}),
        "39",
        SKIN_SCALE * 0.72,
        cfg.get("combo_overlap", 0),
    )
    combo_size = (combo_metrics[1], combo_metrics[2]) if combo_metrics else DEFAULT_SIZES["combo"]

    sizes = dict(DEFAULT_SIZES)
    sizes["playfield"] = (playfield_width, SCENE_HEIGHT)
    sizes["combo"] = combo_size
    return {
        key: {"position": LAYOUT_POSITIONS[key], "size": sizes[key]}
        for key in LAYOUT_POSITIONS
    }
