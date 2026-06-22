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
LAYOUT_ELEMENTS = {
    "playfield": {"position": (0.50, 0.50), "size": (150, 330)},
    "combo": {"position": (0.50, 0.25), "size": (92, 34)},
    "judgement": {"position": (0.50, 0.38), "size": (112, 34)},
    "side_stats": {"position": (0.90, 0.20), "size": (108, 92)},
    "key_input": {"position": (0.90, 0.56), "size": (108, 108)},
    "timeline": {"position": (0.90, 0.80), "size": (104, 34)},
    "strain_graph": {"position": (0.82, 0.93), "size": (210, 48)},
    "star_rating": {"position": (0.08, 0.82), "size": (94, 34)},
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


def selected_glyph(glyphs, character):
    variants = glyphs.get(character, {})

    if not variants:
        return None

    return variants[max(variants)]


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
        lane_width = width / keys

        for lane in range(keys):
            x = int(lane * lane_width)
            painter.fillRect(QRectF(x, 0, lane_width, height), QColor(9, 11, 12, 245))
            painter.setPen(QPen(QColor(92, 98, 102), 0.8))
            painter.drawLine(x, 0, x, height)
            note = skin.get("notes", [None] * keys)[lane]
            receptor = skin.get("keys", [None] * keys)[lane]
            draw_fitted(painter, note, QRectF(x + 2, 58 + lane * 17, lane_width - 4, lane_width - 4))
            draw_fitted(painter, receptor, QRectF(x + 1, height - lane_width - 8, lane_width - 2, lane_width - 2))

        painter.setPen(QPen(QColor(92, 98, 102), 0.8))
        painter.drawLine(width - 1, 0, width - 1, height)
    elif key == "combo":
        glyphs = skin.get("combo_glyphs", {})
        glyph_images = [selected_glyph(glyphs, character) for character in "123"]
        glyph_images = [image for image in glyph_images if image is not None]

        if glyph_images:
            cell_width = width / len(glyph_images)
            for index, image in enumerate(glyph_images):
                draw_fitted(painter, image, QRectF(index * cell_width, 0, cell_width + 2, height), 0)
        else:
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "123")
    elif key == "judgement":
        image = skin.get("hit_images", {}).get("300g")

        if image is None or (image.ndim == 3 and image.shape[2] == 4 and (image[:, :, 3] > 8).sum() < 64):
            image = skin.get("hit_images", {}).get("300")

        if not draw_fitted(painter, image, QRectF(pixmap.rect()), 0):
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "300")
    elif key == "side_stats":
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 238))
        painter.drawText(QRectF(5, 3, width - 10, 24), Qt.AlignRight, "128x")
        painter.drawText(QRectF(5, 25, width - 10, 20), Qt.AlignRight, "99.82%")
        painter.setFont(QFont("Sans Serif", 7))
        painter.drawText(QRectF(5, 49, width - 10, 38), Qt.AlignRight, "MAX  82\n300  46\nMISS   0")
    elif key == "key_input":
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 238))
        lane_width = 16
        gap = 7
        start_x = int((width - (lane_width * 4 + gap * 3)) / 2)

        for lane in range(4):
            x = start_x + lane * (lane_width + gap)
            for row in range(5):
                if (row + lane) % 3 != 1:
                    painter.fillRect(x, 8 + row * 14, lane_width, 8, QColor(205, 211, 185))
            painter.setPen(QPen(QColor(211, 171, 63), 1))
            painter.drawRect(x, height - 22, lane_width, 14)
    elif key == "timeline":
        painter.setPen(QPen(QColor(220, 223, 226), 3))
        painter.drawEllipse(4, 3, 27, 27)
        painter.setPen(QPen(QColor(145, 148, 152), 5))
        painter.drawArc(7, 6, 21, 21, 90 * 16, -210 * 16)
        painter.setPen(QPen(QColor("#eef2f4"), 1))
        painter.drawText(QRectF(37, 0, width - 39, height), Qt.AlignVCenter | Qt.AlignLeft, "1:42")
    elif key == "strain_graph":
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 238))
        points = QPolygonF([
            QPointF(3, height - 7), QPointF(width * 0.18, 18), QPointF(width * 0.34, 31),
            QPointF(width * 0.49, 8), QPointF(width * 0.66, 25), QPointF(width * 0.82, 12),
            QPointF(width - 3, height - 10),
        ])
        painter.setPen(QPen(QColor(75, 224, 104), 2))
        painter.drawPolyline(points)
    else:
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 210))
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

    def set_skin(self, skin):
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

        for key, definition in LAYOUT_ELEMENTS.items():
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

        for item in self.items.values():
            item.set_skin(self.skin)

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

        for key, definition in LAYOUT_ELEMENTS.items():
            self._position_item(self.items[key], definition["position"])

        self.positions_changed.emit({})

    def positions(self):
        return dict(self.custom_positions)
