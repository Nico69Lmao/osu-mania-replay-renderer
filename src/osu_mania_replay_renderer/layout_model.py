SCENE_WIDTH = 640
SCENE_HEIGHT = 360
SKIN_SCALE = SCENE_HEIGHT / 480.0
LAYOUT_POSITIONS = {
    "playfield": (0.50, 0.50),
    "combo": (0.50, 0.25),
    "judgement": (0.50, 0.38),
    "side_stats": (0.94, 0.22),
    "key_input": (0.94, 0.49),
    "timeline": (0.94, 0.64),
    "strain_graph": (0.87, 0.95),
    "star_rating": (0.08, 0.82),
}
DEFAULT_SIZES = {
    "playfield": (150, 360),
    "combo": (34, 34),
    "judgement": (30, 30),
    "side_stats": (74, 104),
    "key_input": (50, 90),
    "timeline": (52, 18),
    "strain_graph": (166, 33),
    "star_rating": (50, 14),
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


def layout_definitions(skin):
    cfg = skin.get("cfg", {})
    keys = max(1, len(skin.get("keys", [])))
    column_widths = cfg.get("column_widths") or [70] * keys
    column_spacing = cfg.get("column_spacing") or [0] * (keys - 1)
    scaled_widths = [int(width * SKIN_SCALE) for width in column_widths]
    scaled_spacing = [int(spacing * SKIN_SCALE) for spacing in column_spacing]
    playfield_width = max(1, sum(scaled_widths) + sum(scaled_spacing))

    combo_metrics = logical_glyph_metrics(
        skin.get("combo_glyphs", {}),
        "128",
        SKIN_SCALE * 0.72,
        cfg.get("combo_overlap", 0),
    )
    combo_size = (combo_metrics[1], combo_metrics[2]) if combo_metrics else DEFAULT_SIZES["combo"]

    judgement = skin.get("hit_images", {}).get("300")
    judgement_density = max(1.0, float(skin.get("hit_image_densities", {}).get("300", 1.0)))

    if meaningful_image(judgement):
        judgement_scale = SCENE_HEIGHT / 768.0 / judgement_density
        judgement_size = (
            max(1, int(judgement.shape[1] * judgement_scale)),
            max(1, int(judgement.shape[0] * judgement_scale)),
        )
    else:
        judgement_size = DEFAULT_SIZES["judgement"]

    sizes = dict(DEFAULT_SIZES)
    sizes["playfield"] = (playfield_width, SCENE_HEIGHT)
    sizes["combo"] = combo_size
    sizes["judgement"] = judgement_size
    return {
        key: {"position": LAYOUT_POSITIONS[key], "size": sizes[key]}
        for key in LAYOUT_POSITIONS
    }
