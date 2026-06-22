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
from osu_mania_replay_renderer.layout_model import (
    SCENE_HEIGHT,
    SCENE_WIDTH,
    SKIN_SCALE,
    layout_definitions,
    meaningful_image,
    visible_glyph_metrics,
)


DISPLAY_UNIT = SCENE_HEIGHT / 360.0
EDGE_WARNING_DISTANCE = int(10 * DISPLAY_UNIT)
SCALED_OVERLAY_KEYS = {"side_stats", "key_input", "timeline", "strain_graph", "star_rating"}


def visible_crop(image):
    if image is None or image.ndim < 3 or image.shape[2] < 4:
        return image

    ys, xs = (image[:, :, 3] > 8).nonzero()

    if len(xs) == 0:
        return image

    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def skin_pixmap(image):
    if image is None:
        return None

    if not image.flags.c_contiguous:
        image = image.copy()

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
    metrics = visible_glyph_metrics(
        skin.get("combo_glyphs", {}),
        "39",
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

    if key in SCALED_OVERLAY_KEYS:
        painter.scale(DISPLAY_UNIT, DISPLAY_UNIT)
        width = int(width / DISPLAY_UNIT)
        height = int(height / DISPLAY_UNIT)

    if key == "playfield":
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 225))
        keys = max(1, len(skin.get("keys", [])))
        cfg = skin.get("cfg", {})
        source_widths = cfg.get("column_widths") or [70] * keys
        source_spacing = cfg.get("column_spacing") or [0] * (keys - 1)
        lane_widths = [int(value * SKIN_SCALE) for value in source_widths]
        lane_spacing = [int(value * SKIN_SCALE) for value in source_spacing]
        reference_note_centres = {0: 320, 2: 790, 3: 320}
        receptor_center_y = int(SCENE_HEIGHT * 0.915)
        lane_x = 0

        for lane in range(keys):
            lane_width = lane_widths[lane]
            x = lane_x
            painter.fillRect(QRectF(x, 0, lane_width, height), QColor(9, 11, 12, 245))
            painter.setPen(QPen(QColor(92, 98, 102), 0.8 * DISPLAY_UNIT))
            painter.drawLine(x, 0, x, height)
            note = skin.get("notes", [None] * keys)[lane]
            receptor = visible_crop(skin.get("keys", [None] * keys)[lane])
            note_width = max(1, int(lane_width * 0.94))
            note_height = max(1, int(note.shape[0] * note_width / note.shape[1])) if note is not None else note_width

            if lane in reference_note_centres:
                note_y = reference_note_centres[lane] - note_height / 2
                draw_exact(painter, note, x + (lane_width - note_width) / 2, note_y, note_width, note_height)

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

        painter.setPen(QPen(QColor(92, 98, 102), 0.8 * DISPLAY_UNIT))
        painter.drawLine(width - 1, 0, width - 1, height)
    elif key == "combo":
        if not draw_combo_preview(painter, skin, width, height):
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "39")
    elif key == "judgement":
        image = skin.get("hit_images", {}).get("300g")
        image = visible_crop(image)

        if meaningful_image(image):
            density = max(1.0, float(skin.get("hit_image_densities", {}).get("300g", 1.0)))
            scale = SCENE_HEIGHT / 768.0 / density
            dot_width = max(1, int(image.shape[1] * scale))
            dot_height = max(1, int(image.shape[0] * scale))
            draw_exact(
                painter,
                image,
                (width - dot_width) / 2,
                (height - dot_height) / 2,
                dot_width,
                dot_height,
            )
        else:
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(QRectF(width / 2 - 3, height / 2 - 3, 6, 6))
    elif key == "side_stats":
        large_font = QFont("Sans Serif", 8)
        large_font.setWeight(QFont.DemiBold)
        painter.setFont(large_font)
        painter.drawText(QRectF(3, 1, width - 6, 17), Qt.AlignRight, "39x")
        painter.setFont(QFont("Sans Serif", 6))
        painter.drawText(QRectF(3, 18, width - 6, 14), Qt.AlignRight, "100.00%")
        painter.setFont(QFont("Sans Serif", 5))
        rows = [
            ("300g: 27", "#e8d66b"), ("300: 5", "#ececec"),
            ("200: 0", "#66c878"), ("100: 0", "#67b7d8"),
            ("50: 0", "#647cba"), ("Miss: 0", "#7655b8"),
        ]

        for index, (text, colour) in enumerate(rows):
            painter.setPen(QColor(colour))
            painter.drawText(QRectF(3, 35 + index * 9, width - 6, 9), Qt.AlignRight, text)

        painter.setPen(QColor("#eeeeee"))
        painter.setFont(QFont("Sans Serif", 6))
        painter.drawText(QRectF(3, 90, width - 6, 10), Qt.AlignRight, "pp: 42.67")
    elif key == "key_input":
        painter.setPen(QColor("#e8e8e8"))
        painter.setFont(QFont("Sans Serif", 5))
        painter.drawText(QRectF(0, 0, width, 10), Qt.AlignRight, "INPUT / BPM")
        painter.fillRect(QRectF(0, 12, width, height - 12), QColor(0, 0, 0, 238))
        lane_width = 7
        gap = 6
        start_x = int((width - (lane_width * 4 + gap * 3)) / 2)

        for lane in range(4):
            x = start_x + lane * (lane_width + gap)
            bar_rows = ([18, 43, 69] if lane == 0 else [18, 67] if lane == 1 else [30, 69] if lane == 2 else [42, 69])

            for row in bar_rows:
                painter.fillRect(x, row, lane_width, 1, QColor(225, 225, 225))

            if lane in (1, 2, 3):
                painter.fillRect(x, 48 + lane * 5, lane_width, 7, QColor(105, 105, 105))

            painter.setPen(QPen(QColor(95, 145, 170), 0.8))
            painter.drawRect(x, 77, lane_width, 8)
            painter.setPen(QColor("#e8e8e8"))
            painter.drawText(QRectF(x, 77, lane_width, 8), Qt.AlignCenter, str(lane + 1))

        painter.setFont(QFont("Sans Serif", 4))
        for lane, bpm in enumerate(("101", "102", "086", "090")):
            x = start_x + lane * (lane_width + gap)
            painter.drawText(QRectF(x - 2, 88, lane_width + 4, 8), Qt.AlignCenter, bpm)
    elif key == "timeline":
        painter.setPen(QPen(QColor(220, 223, 226), 1.5))
        painter.drawEllipse(width - 18, 2, 16, 16)
        painter.setPen(QPen(QColor(145, 148, 152), 2.5))
        painter.drawArc(width - 16, 4, 12, 12, 90 * 16, -300 * 16)
        painter.setPen(QPen(QColor("#eef2f4"), 1))
        painter.setFont(QFont("Sans Serif", 5))
        painter.drawText(QRectF(0, 0, width - 21, height), Qt.AlignVCenter | Qt.AlignRight, "0:05 / 0:30")
    elif key == "strain_graph":
        point_list = [
            QPointF(1, height - 2), QPointF(width * 0.03, height * 0.52), QPointF(width * 0.05, height - 2),
            QPointF(width * 0.17, height - 2), QPointF(width * 0.19, height * 0.15),
            QPointF(width * 0.21, height - 2), QPointF(width * 0.34, height - 2),
            QPointF(width * 0.36, height * 0.30), QPointF(width * 0.38, height - 2),
            QPointF(width * 0.51, height * 0.18), QPointF(width * 0.54, height - 2),
            QPointF(width * 0.65, height * 0.12), QPointF(width * 0.68, height * 0.40),
            QPointF(width * 0.72, height - 2), QPointF(width * 0.82, height * 0.20),
            QPointF(width * 0.85, height - 2), QPointF(width * 0.96, height * 0.10),
            QPointF(width - 1, height * 0.62),
        ]
        split = 6
        painter.setPen(QPen(QColor(63, 221, 85), 1.5))
        painter.drawPolyline(QPolygonF(point_list[:split]))
        painter.setPen(QPen(QColor(145, 145, 145), 1.5))
        painter.drawPolyline(QPolygonF(point_list[split - 1:]))
    else:
        painter.setFont(QFont("Sans Serif", 6))
        painter.drawText(QRectF(0, 0, width, height), Qt.AlignCenter, "SR: 2.34*")

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
        painter.setPen(QPen(colour, (2.4 if self.isSelected() else 1.4) * DISPLAY_UNIT))
        inset = DISPLAY_UNIT
        painter.drawRect(bounds.adjusted(inset, inset, -inset, -inset))

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
        status = QLabel("Canvas 1920 x 1080")
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
        unit = DISPLAY_UNIT
        border = self.scene.addRect(unit, unit, SCENE_WIDTH - 2 * unit, SCENE_HEIGHT - 2 * unit, QPen(QColor("#dce4e8"), 2.2 * unit))
        border.setZValue(-10)
        margin = 12 * unit
        safe_pen = QPen(QColor("#69747a"), unit, Qt.DashLine)
        safe = self.scene.addRect(margin, margin, SCENE_WIDTH - 2 * margin, SCENE_HEIGHT - 2 * margin, safe_pen)
        safe.setZValue(-10)
        guide_pen = QPen(QColor(74, 82, 87, 150), unit, Qt.DashLine)

        for x in (SCENE_WIDTH / 2,):
            guide = self.scene.addLine(x, margin, x, SCENE_HEIGHT - margin, guide_pen)
            guide.setZValue(-10)

        for y in (SCENE_HEIGHT / 2,):
            guide = self.scene.addLine(margin, y, SCENE_WIDTH - margin, y, guide_pen)
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
