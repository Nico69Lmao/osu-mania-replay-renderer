from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QBrush, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)


SCENE_WIDTH = 640
SCENE_HEIGHT = 360
LAYOUT_ELEMENTS = {
    "playfield": {"label": "Playfield + skin", "position": (0.50, 0.50), "size": (150, 330), "colour": "#3b8f77"},
    "combo": {"label": "Skin combo", "position": (0.50, 0.25), "size": (92, 34), "colour": "#d0a84c"},
    "judgement": {"label": "Hit judgement", "position": (0.50, 0.38), "size": (112, 34), "colour": "#c56f57"},
    "side_stats": {"label": "Statistics", "position": (0.90, 0.20), "size": (108, 92), "colour": "#5688b8"},
    "key_input": {"label": "Key input", "position": (0.90, 0.56), "size": (108, 108), "colour": "#8872b8"},
    "timeline": {"label": "Timeline", "position": (0.90, 0.80), "size": (104, 34), "colour": "#5e9dad"},
    "strain_graph": {"label": "Strain graph", "position": (0.82, 0.93), "size": (210, 48), "colour": "#58a86a"},
    "star_rating": {"label": "Star rating", "position": (0.08, 0.82), "size": (94, 34), "colour": "#aa8752"},
}


class MovableLayoutItem(QGraphicsRectItem):
    def __init__(self, key, definition, moved_callback):
        width, height = definition["size"]
        super().__init__(0, 0, width, height)
        self.key = key
        self.moved_callback = moved_callback
        colour = QColor(definition["colour"])
        colour.setAlpha(185)
        self.setBrush(QBrush(colour))
        self.setPen(QPen(QColor("#e6eaed"), 1.2))
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setCursor(Qt.OpenHandCursor)

        label = QGraphicsSimpleTextItem(definition["label"], self)
        label.setBrush(QBrush(QColor("#ffffff")))
        bounds = label.boundingRect()
        label.setPos((width - bounds.width()) / 2, (height - bounds.height()) / 2)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            rect = self.rect()
            x = min(max(0.0, value.x()), SCENE_WIDTH - rect.width())
            y = min(max(0.0, value.y()), SCENE_HEIGHT - rect.height())
            return value.__class__(x, y)

        return super().itemChange(change, value)

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
        self.view = LayoutView(self.scene)
        self.view.setObjectName("layoutPreview")
        self.view.setRenderHints(self.view.renderHints())
        self.view.setMinimumHeight(430)
        layout.addWidget(self.view, 1)

        for key, definition in LAYOUT_ELEMENTS.items():
            item = MovableLayoutItem(key, definition, self._item_moved)
            self.scene.addItem(item)
            self.items[key] = item
            self._position_item(item, self.custom_positions.get(key, definition["position"]))

    @staticmethod
    def _normalised_position(item):
        rect = item.rect()
        return [
            round((item.x() + rect.width() / 2) / SCENE_WIDTH, 4),
            round((item.y() + rect.height() / 2) / SCENE_HEIGHT, 4),
        ]

    @staticmethod
    def _position_item(item, position):
        x, y = position
        rect = item.rect()
        item.setPos(float(x) * SCENE_WIDTH - rect.width() / 2, float(y) * SCENE_HEIGHT - rect.height() / 2)

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
