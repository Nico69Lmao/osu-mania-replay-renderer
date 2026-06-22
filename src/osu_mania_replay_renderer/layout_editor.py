from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QBrush, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from osu_mania_replay_renderer.skin_loader import load_mania_skin


SCENE_WIDTH = 640
SCENE_HEIGHT = 360
EDGE_WARNING_DISTANCE = 10
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


def visible_crop(image):
    if image is None or image.ndim < 3 or image.shape[2] < 4:
        return image

    ys, xs = (image[:, :, 3] > 8).nonzero()

    if len(xs) == 0:
        return image

    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


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


def skin_pixmap(image):
    if image is None:
        return None

    height, width = image.shape[:2]

    if image.ndim == 2:
        fmt = QImage.Format_Grayscale8
    elif image.shape[2] == 4:
        fmt = QImage.Format_ARGB32
    else:
        fmt = QImage.Format_BGR888

    qimage = QImage(image.data, width, height, image.strides[0], fmt).copy()
    return QPixmap.fromImage(qimage)


def draw_fitted(painter, image, bounds, margin=2):
    pixmap = skin_pixmap(image)

    if pixmap is None or pixmap.isNull():
        return False

    target = bounds.adjusted(margin, margin, -margin, -margin)
    scaled = pixmap.scaled(target.size().toSize(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    x = target.x() + (target.width() - scaled.width()) / 2
    y = target.y() + (target.height() - scaled.height()) / 2
    painter.drawPixmap(int(x), int(y), scaled)
    return True


def draw_exact(painter, image, x, y, width, height):
    pixmap = skin_pixmap(image)

    if pixmap is None or pixmap.isNull():
        return False

    scaled = pixmap.scaled(max(1, int(width)), max(1, int(height)), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    painter.drawPixmap(int(x), int(y), scaled)
    return True


def draw_combo_preview(painter, skin, width, height):
    cfg = skin.get("cfg", {})
    metrics = logical_glyph_metrics(
        skin.get("combo_glyphs", {}),
        "128",
        SKIN_SCALE * 0.72,
        cfg.get("combo_overlap", 0),
    )

    if metrics is None:
        return False

    glyphs, total_width, max_height, overlap = metrics
    x = int((width - total_width) / 2)
    top = int((height - max_height) / 2)

    for image, glyph_width, glyph_height in glyphs:
        y = top + (max_height - glyph_height) // 2
        draw_exact(painter, image, x, y, glyph_width, glyph_height)
        x += glyph_width - overlap

    return True


def preview_pixmap(key, definition, skin):
    width, height = definition["size"]
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    painter.setPen(QPen(QColor("#eef2f4"), 1))
    font = QFont("Sans Serif", 9)
    font.setWeight(QFont.DemiBold)
    painter.setFont(font)

    if key == "playfield":
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 225))
        keys = max(1, len(skin.get("keys", [])))
        cfg = skin.get("cfg", {})
        source_widths = cfg.get("column_widths") or [70] * keys
        source_spacing = cfg.get("column_spacing") or [0] * (keys - 1)
        lane_widths = [int(value * SKIN_SCALE) for value in source_widths]
        lane_spacing = [int(value * SKIN_SCALE) for value in source_spacing]
        judge_y = int((cfg.get("hit_position") or 402) * SKIN_SCALE)
        lane_x = 0

        for lane in range(keys):
            lane_width = lane_widths[lane]
            x = lane_x
            painter.fillRect(QRectF(x, 0, lane_width, height), QColor(9, 11, 12, 245))
            painter.setPen(QPen(QColor(92, 98, 102), 0.8))
            painter.drawLine(x, 0, x, height)
            note = skin.get("notes", [None] * keys)[lane]
            receptor = visible_crop(skin.get("keys", [None] * keys)[lane])
            note_width = max(1, int(lane_width * 0.94))
            note_height = max(1, int(note.shape[0] * note_width / note.shape[1])) if note is not None else note_width
            note_y = 40 + lane * 35
            draw_exact(painter, note, x + (lane_width - note_width) / 2, note_y, note_width, note_height)
            receptor_center_y = judge_y - note_width // 2
            draw_exact(
                painter,
                receptor,
                x + (lane_width - note_width) / 2,
                receptor_center_y - note_width / 2,
                note_width,
                note_width,
            )
            lane_x += lane_width

            if lane < len(lane_spacing):
                lane_x += lane_spacing[lane]

        stage_bottom = skin.get("stage_bottom")

        if meaningful_image(stage_bottom):
            cover_width = int(stage_bottom.shape[1] * SKIN_SCALE)
            cover_height = int(stage_bottom.shape[0] * SKIN_SCALE)
            draw_exact(
                painter,
                stage_bottom,
                (width - cover_width) / 2,
                height - cover_height,
                cover_width,
                cover_height,
            )

        painter.setPen(QPen(QColor(92, 98, 102), 0.8))
        painter.drawLine(width - 1, 0, width - 1, height)
    elif key == "combo":
        if not draw_combo_preview(painter, skin, width, height):
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "123")
    elif key == "judgement":
        image = skin.get("hit_images", {}).get("300g")

        if image is None or (image.ndim == 3 and image.shape[2] == 4 and (image[:, :, 3] > 8).sum() < 64):
            image = skin.get("hit_images", {}).get("300")

        if not draw_fitted(painter, image, QRectF(pixmap.rect()), 0):
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "300")
    elif key == "side_stats":
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 242))
        large_font = QFont("Sans Serif", 8)
        large_font.setWeight(QFont.DemiBold)
        painter.setFont(large_font)
        painter.drawText(QRectF(3, 1, width - 6, 17), Qt.AlignRight, "128x")
        painter.setFont(QFont("Sans Serif", 6))
        painter.drawText(QRectF(3, 18, width - 6, 14), Qt.AlignRight, "99.82%")
        painter.setFont(QFont("Sans Serif", 5))
        painter.drawText(
            QRectF(3, 34, width - 6, height - 36),
            Qt.AlignRight,
            "300g  82\n300   46\n200    3\n100    1\n50     0\nMiss   0",
        )
    elif key == "key_input":
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 238))
        lane_width = 7
        gap = 4
        start_x = int((width - (lane_width * 4 + gap * 3)) / 2)

        for lane in range(4):
            x = start_x + lane * (lane_width + gap)
            for row in range(8):
                if (row + lane) % 3 != 1:
                    painter.fillRect(x, 5 + row * 8, lane_width, 4, QColor(205, 211, 185))
            painter.setPen(QPen(QColor(211, 171, 63), 1))
            painter.drawRect(x, height - 13, lane_width, 8)
    elif key == "timeline":
        painter.setPen(QPen(QColor(220, 223, 226), 1.5))
        painter.drawEllipse(2, 2, 14, 14)
        painter.setPen(QPen(QColor(145, 148, 152), 2.5))
        painter.drawArc(4, 4, 10, 10, 90 * 16, -210 * 16)
        painter.setPen(QPen(QColor("#eef2f4"), 1))
        painter.setFont(QFont("Sans Serif", 5))
        painter.drawText(QRectF(19, 0, width - 20, height), Qt.AlignVCenter | Qt.AlignLeft, "1:42")
    elif key == "strain_graph":
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 238))
        points = QPolygonF([
            QPointF(2, height - 4), QPointF(width * 0.18, height * 0.35), QPointF(width * 0.34, height * 0.72),
            QPointF(width * 0.49, height * 0.16), QPointF(width * 0.66, height * 0.58),
            QPointF(width * 0.82, height * 0.28), QPointF(width - 2, height - 5),
        ])
        painter.setPen(QPen(QColor(75, 224, 104), 2))
        painter.drawPolyline(points)
    else:
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 210))
        painter.setFont(QFont("Sans Serif", 6))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "SR 5.82*")

    painter.end()
    return pixmap


class MovableLayoutItem(QGraphicsPixmapItem):
    def __init__(self, key, definition, skin, moved_callback):
        super().__init__(preview_pixmap(key, definition, skin))
        self.key = key
        self.definition = definition
        self.moved_callback = moved_callback
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setCursor(Qt.OpenHandCursor)

    def set_skin(self, skin, definition=None):
        if definition is not None:
            self.definition = definition

        self.setPixmap(preview_pixmap(self.key, self.definition, skin))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            bounds = self.boundingRect()
            x = min(max(0.0, value.x()), SCENE_WIDTH - bounds.width())
            y = min(max(0.0, value.y()), SCENE_HEIGHT - bounds.height())
            return QPointF(x, y)

        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        scene_pos = self.scenePos()
        bounds = self.boundingRect()
        near_edge = (
            scene_pos.x() <= EDGE_WARNING_DISTANCE
            or scene_pos.y() <= EDGE_WARNING_DISTANCE
            or scene_pos.x() + bounds.width() >= SCENE_WIDTH - EDGE_WARNING_DISTANCE
            or scene_pos.y() + bounds.height() >= SCENE_HEIGHT - EDGE_WARNING_DISTANCE
        )
        colour = QColor("#ffb454") if near_edge else QColor("#dce4e8")

        if self.isSelected():
            colour = QColor("#63d6e6")

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(colour, 2.4 if self.isSelected() else 1.4))
        painter.drawRect(bounds.adjusted(1, 1, -1, -1))

    def mousePressEvent(self, event):
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)
        self.moved_callback(self)


class LayoutView(QGraphicsView):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)


class LayoutEditor(QWidget):
    positions_changed = Signal(dict)

    def __init__(self, positions=None, parent=None):
        super().__init__(parent)
        self.custom_positions = dict(positions or {})
        self.items = {}
        self.skin = load_mania_skin(None, 4)
        self.definitions = layout_definitions(self.skin)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        toolbar = QHBoxLayout()
        status = QLabel("Canvas 16:9")
        status.setObjectName("layoutStatus")
        toolbar.addWidget(status)
        toolbar.addStretch()
        reset_button = QPushButton("Reset positions")
        reset_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        reset_button.clicked.connect(self.reset_positions)
        toolbar.addWidget(reset_button)
        layout.addLayout(toolbar)

        self.scene = QGraphicsScene(0, 0, SCENE_WIDTH, SCENE_HEIGHT, self)
        self.scene.setBackgroundBrush(QBrush(QColor("#050607")))
        self._draw_canvas_guides()
        self.view = LayoutView(self.scene)
        self.view.setObjectName("layoutPreview")
        self.view.setRenderHints(self.view.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setMinimumHeight(430)
        layout.addWidget(self.view, 1)

        for key, definition in self.definitions.items():
            item = MovableLayoutItem(key, definition, self.skin, self._item_moved)
            self.scene.addItem(item)
            self.items[key] = item
            self._position_item(item, self.custom_positions.get(key, definition["position"]))

    def _draw_canvas_guides(self):
        border = self.scene.addRect(1, 1, SCENE_WIDTH - 2, SCENE_HEIGHT - 2, QPen(QColor("#dce4e8"), 2.2))
        border.setZValue(-10)
        safe_pen = QPen(QColor("#69747a"), 1, Qt.DashLine)
        safe = self.scene.addRect(12, 12, SCENE_WIDTH - 24, SCENE_HEIGHT - 24, safe_pen)
        safe.setZValue(-10)
        guide_pen = QPen(QColor(74, 82, 87, 150), 1, Qt.DashLine)

        for x in (SCENE_WIDTH / 2,):
            guide = self.scene.addLine(x, 12, x, SCENE_HEIGHT - 12, guide_pen)
            guide.setZValue(-10)

        for y in (SCENE_HEIGHT / 2,):
            guide = self.scene.addLine(12, y, SCENE_WIDTH - 12, y, guide_pen)
            guide.setZValue(-10)

    def set_skin(self, skin_folder, keys=4):
        self.skin = load_mania_skin(skin_folder, max(1, int(keys)))
        self.definitions = layout_definitions(self.skin)

        for key, item in self.items.items():
            position = self.custom_positions.get(key, self.definitions[key]["position"])
            item.set_skin(self.skin, self.definitions[key])
            self._position_item(item, position)

        self.scene.update()

    @staticmethod
    def _normalised_position(item):
        bounds = item.boundingRect()
        return [
            round((item.x() + bounds.width() / 2) / SCENE_WIDTH, 4),
            round((item.y() + bounds.height() / 2) / SCENE_HEIGHT, 4),
        ]

    @staticmethod
    def _position_item(item, position):
        x, y = position
        bounds = item.boundingRect()
        item.setPos(float(x) * SCENE_WIDTH - bounds.width() / 2, float(y) * SCENE_HEIGHT - bounds.height() / 2)

    def _item_moved(self, item):
        self.custom_positions[item.key] = self._normalised_position(item)
        self.positions_changed.emit(dict(self.custom_positions))

    def reset_positions(self):
        self.custom_positions.clear()

        for key, definition in self.definitions.items():
            self._position_item(self.items[key], definition["position"])

        self.positions_changed.emit({})

    def positions(self):
        return dict(self.custom_positions)
