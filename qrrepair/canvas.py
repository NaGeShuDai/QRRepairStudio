from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF, QTransform, QPainterPath
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsItem
from .engine import finder_mask, warp, render_matrix, region_masks
from .overlays import grid_colors


def pixmap(array):
    a = np.ascontiguousarray(array)
    if a.ndim == 2:
        image = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format.Format_Grayscale8)
    elif a.shape[2] == 4:
        image = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format.Format_RGBA8888)
    else:
        image = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class ZoomView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#e9eef5"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._panning = False
        self._auto_fit = True

    def fit(self):
        self._auto_fit = True
        rect = self.scene().sceneRect()
        if not rect.isEmpty():
            self.fitInView(rect.adjusted(-10, -10, 10, 10), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit()

    def wheelEvent(self, event):
        self._auto_fit = False
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        zoom = self.transform().m11()
        if .02 < zoom * factor < 80:
            self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._auto_fit = False
            self._panning = True
            self._last_pan = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._last_pan
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self._last_pan = event.position()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.unsetCursor()
            return
        super().mouseReleaseEvent(event)


class Handle(QGraphicsEllipseItem):
    def __init__(self, view, index):
        super().__init__(-7, -7, 14, 14)
        self.view, self.index = view, index
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges |
                      QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setBrush(QColor("#2de0ba"))
        self.setPen(QPen(QColor("#071d22"), 2))
        self.setZValue(5)
        self.setToolTip(["左上角", "右上角", "右下角", "左下角"][index])

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.view.image is not None:
            h, w = self.view.image.shape[:2]
            return QPointF(max(0, min(w, value.x())), max(0, min(h, value.y())))
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and not self.view.loading:
            self.view.update_corners()
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.view.cornersChanged.emit(self.view.corners.copy())


class CropView(ZoomView):
    """Rectified source evidence in the same orientation as the module editor."""
    def __init__(self):
        super().__init__()
        self.crop = None
        self.key = None
        self.highlight = None
        self.dimension = 0
        self.setMinimumSize(160, 160)
        self.setAccessibleName("摆正的选区原图")

    def set_crop(self, image, corners):
        key = (id(image), corners.tobytes()) if image is not None and corners is not None else None
        if key == self.key:
            return
        self.key = key
        self.scene().clear()
        self.crop = None
        self.highlight = None
        if key is not None:
            try:
                self.crop = warp(image, corners, 720)
            except ValueError:
                self.key = None
            if self.crop is not None:
                self.scene().addPixmap(pixmap(self.crop))
        self.scene().setSceneRect(0, 0, 720, 720)
        self.fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def drawForeground(self, painter, rect):
        if self.highlight is not None and self.dimension:
            r, c = self.highlight
            cell = 720 / self.dimension
            painter.setPen(QPen(QColor("#e69727"), 0))
            painter.drawRect(QRectF(c*cell, r*cell, cell, cell))


class SourceView(ZoomView):
    cornersChanged = Signal(object)
    colorPicked = Signal(object)
    findersPicked = Signal(object)

    def __init__(self):
        super().__init__()
        self.image = None
        self.corners = None
        self.handles = []
        self.loading = False
        self.dimension = 0
        self.highlight = None
        self.contour_item = None
        self.base_item = None
        self.picking_color = False
        self.finder_clicks = None
        self.setAccessibleName("原图透视编辑器")

    def enable_navigation(self, enabled=True):
        policy = Qt.ScrollBarPolicy.ScrollBarAsNeeded if enabled else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        self.setHorizontalScrollBarPolicy(policy)
        self.setVerticalScrollBarPolicy(policy)

    def set_image(self, image, corners=None):
        self.loading = True
        self.scene().clear()
        self.contour_item = None
        self.base_item = None
        self.handles = []
        self.image = image
        self.corners = None if corners is None else np.array(corners, np.float32)
        if image is not None:
            self.base_item = self.scene().addPixmap(pixmap(image))
            self.scene().setSceneRect(0, 0, image.shape[1], image.shape[0])
            if self.corners is not None:
                for i, point in enumerate(self.corners):
                    h = Handle(self, i)
                    h.setPos(float(point[0]), float(point[1]))
                    self.scene().addItem(h)
                    self.handles.append(h)
        else:
            self.scene().setSceneRect(0, 0, 500, 500)
            item = self.scene().addText("导入图片")
            item.setDefaultTextColor(QColor("#7b899d"))
            item.setPos(90, 190)
        self.loading = False
        self.fit()
        self.viewport().update()

    def update_corners(self):
        self.corners = np.array([[h.pos().x(), h.pos().y()] for h in self.handles], np.float32)
        if self.contour_item is not None:
            self.contour_item.setVisible(False)
        self.viewport().update()

    def focus_selection(self):
        if self.corners is not None:
            self._auto_fit = False
            rect = QPolygonF([QPointF(float(x),float(y)) for x,y in self.corners]).boundingRect()
            margin = max(rect.width(),rect.height())*.12
            self.fitInView(rect.adjusted(-margin,-margin,margin,margin),Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event):
        if self.finder_clicks is not None and event.button() == Qt.MouseButton.LeftButton and self.image is not None:
            p = self.mapToScene(event.position().toPoint())
            if 0 <= p.x() < self.image.shape[1] and 0 <= p.y() < self.image.shape[0]:
                self.finder_clicks.append([p.x(), p.y()])
                self.viewport().update()
                if len(self.finder_clicks) == 3:
                    points = np.float32(self.finder_clicks)
                    self.finder_clicks = None
                    self.unsetCursor()
                    self.findersPicked.emit(points)
            return
        if self.picking_color and event.button() == Qt.MouseButton.LeftButton and self.image is not None:
            p = self.mapToScene(event.position().toPoint())
            x,y = int(np.floor(p.x())),int(np.floor(p.y()))
            if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                self.picking_color = False
                self.unsetCursor()
                self.colorPicked.emit(tuple(int(v) for v in self.image[y,x]))
            return
        super().mousePressEvent(event)

    def set_contours(self, array, opacity, visible):
        if self.base_item is not None:
            self.base_item.setOpacity(1)
        if self.contour_item is not None:
            self.scene().removeItem(self.contour_item)
            self.contour_item = None
        if array is None or self.corners is None or not visible:
            return
        size = array.shape[0]
        src = np.float32([[0,0],[size,0],[size,size],[0,size]])
        h = cv2.getPerspectiveTransform(src, self.corners.astype(np.float32))
        item = self.scene().addPixmap(pixmap(array))
        item.setTransform(QTransform(*h.T.flatten().tolist()))
        item.setOpacity(opacity)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        item.setZValue(1)
        self.contour_item = item

    def drawForeground(self, painter, rect):
        if self.finder_clicks:
            painter.setPen(QPen(QColor("#ef8b24"), 2))
            for x,y in self.finder_clicks:
                painter.drawEllipse(QPointF(x,y), 5, 5)
        if self.corners is None:
            return
        shade = QPainterPath()
        shade.setFillRule(Qt.FillRule.OddEvenFill)
        shade.addRect(self.sceneRect())
        shade.addPolygon(QPolygonF([QPointF(float(x),float(y)) for x,y in self.corners]))
        shade.closeSubpath()
        painter.fillPath(shade,QColor(19,30,48,145))
        painter.setPen(QPen(QColor("#2de0ba"), 0))
        painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in self.corners]))
        n = self.dimension
        if n:
            target = np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float32)
            h = cv2.getPerspectiveTransform(target, self.corners.astype(np.float32))

            def point(x, y):
                a = h @ np.array([x, y, 1])
                if abs(a[2]) < 1e-8:
                    return QPointF()
                return QPointF(float(a[0]/a[2]), float(a[1]/a[2]))

            painter.setPen(QPen(QColor(45, 224, 186, 80), 0))
            for k in range(1, n):
                painter.drawLine(point(k, 0), point(k, n))
                painter.drawLine(point(0, k), point(n, k))
            painter.setPen(QPen(QColor("#b19aff"), 1.5))
            for x,y in ((0,0),(n-7,0),(0,n-7)):
                painter.drawPolygon(QPolygonF([point(x,y),point(x+7,y),point(x+7,y+7),point(x,y+7)]))
            if self.highlight:
                r, c = self.highlight
                painter.setBrush(QColor(255, 177, 66, 110))
                painter.setPen(QPen(QColor("#ffb142"), 0))
                painter.drawPolygon(QPolygonF([point(c, r), point(c+1, r), point(c+1, r+1), point(c, r+1)]))


class GridView(ZoomView):
    editStarted = Signal()
    editFinished = Signal()
    hovered = Signal(int, int)
    CELL = 12

    def __init__(self):
        super().__init__()
        self.grid = None
        self.preview = None
        self.tool = "black"
        self.heatmap = False
        self.show_lines = True
        self.dragging = False
        self.rect_start = None
        self.selection = None
        self.visited = set()
        self.contours = None
        self.contour_visible = False
        self.contour_opacity = .65
        self.qr_opacity = .3
        self.preview_mode = False
        self.show_regions = False
        self.repair_status = None
        self.base_item = None
        self.setAccessibleName("二维码模块编辑器")

    def set_grid(self, grid, fit=False):
        self.grid = grid
        self.preview = None
        self.refresh()
        if fit:
            self.fit()

    def refresh(self):
        self.scene().clear()
        self.base_item = None
        if self.grid is None:
            self.scene().setSceneRect(0, 0, 500, 500)
            if self.contour_visible and self.contours is not None:
                self.add_contours(500)
                return
            item = self.scene().addText("模块网格")
            item.setDefaultTextColor(QColor("#7b899d"))
            item.setPos(85, 190)
            return
        g = self.grid
        a = self.preview if self.preview is not None else g.values
        n = len(a)
        if self.preview_mode:
            matrix = np.where(a < 0, g.probability >= .5, a).astype(np.int8)
            pixels = render_matrix(matrix, self.CELL, border=4)
            self.base_item = self.scene().addPixmap(pixmap(pixels))
            self.scene().setSceneRect(0, 0, pixels.shape[1], pixels.shape[0])
            self.viewport().update()
            return
        rgb = grid_colors(g, self.heatmap, self.preview)
        rgb = np.repeat(np.repeat(rgb, self.CELL, 0), self.CELL, 1)
        base = self.scene().addPixmap(pixmap(rgb))
        self.base_item = base
        if self.contour_visible and self.contours is not None:
            base.setOpacity(self.qr_opacity)
            self.add_contours(n*self.CELL)
        self.scene().setSceneRect(0, 0, n*self.CELL, n*self.CELL)
        self.viewport().update()

    def add_contours(self, size):
        item = self.scene().addPixmap(pixmap(self.contours))
        item.setScale(size / self.contours.shape[0])
        item.setOpacity(self.contour_opacity)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def drawForeground(self, painter, rect):
        if self.grid is None or self.preview_mode:
            return
        n, cell = len(self.grid.values), self.CELL
        if self.repair_status:
            masks = region_masks(self.grid.version)
            data_mask = masks.get("数据")
            if data_mask is not None:
                state = self.repair_status
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(52, 190, 110, 78) if state == "通过" else QColor(235, 70, 70, 78) if state == "未通过" else QColor(150, 150, 150, 45))
                for r, c in np.argwhere(data_mask):
                    painter.drawRect(QRectF(c*cell, r*cell, cell, cell))
        if self.show_regions:
            colors = {"定位": QColor(72, 118, 220, 75), "时序": QColor(30, 170, 145, 85),
                      "格式": QColor(232, 151, 44, 90), "版本": QColor(165, 91, 205, 90),
                      "校正": QColor(230, 87, 130, 75), "数据": QColor(80, 80, 80, 25)}
            masks = region_masks(self.grid.version)
            for name, mask in masks.items():
                painter.setBrush(colors[name])
                painter.setPen(Qt.PenStyle.NoPen)
                for r, c in np.argwhere(mask):
                    painter.drawRect(QRectF(c*cell, r*cell, cell, cell))
        if self.show_lines and self.transform().m11()*cell > 6:
            painter.setPen(QPen(QColor(90, 115, 139, 75), 0))
            for i in range(n+1):
                painter.drawLine(i*cell, 0, i*cell, n*cell)
                painter.drawLine(0, i*cell, n*cell, i*cell)
        painter.setPen(QPen(QColor("#feaf47"), 0))
        for r, c in np.argwhere(self.grid.locked):
            painter.drawRect(QRectF(c*cell+2, r*cell+2, cell-4, cell-4))
        if self.selection:
            r0, c0, r1, c1 = self.selection
            painter.setBrush(QColor(45, 224, 186, 60))
            painter.setPen(QPen(QColor("#2de0ba"), 0))
            painter.drawRect(QRectF(c0*cell, r0*cell, (c1-c0+1)*cell, (r1-r0+1)*cell))

    def cell_at(self, pos):
        if self.grid is None:
            return None
        p = self.mapToScene(pos.toPoint())
        r, c = int(np.floor(p.y()/self.CELL)), int(np.floor(p.x()/self.CELL))
        return (r, c) if 0 <= r < len(self.grid.values) and 0 <= c < len(self.grid.values) else None

    def paint_cell(self, cell, right=False):
        if self.preview_mode or cell is None or cell in self.visited:
            return
        self.visited.add(cell)
        r, c = cell
        if finder_mask(self.grid.version)[r,c]:
            return
        if self.tool == "lock":
            if self.grid.values[r, c] >= 0:
                self.grid.locked[r, c] = not self.grid.locked[r, c]
        elif not self.grid.locked[r, c]:
            if self.tool not in ("black", "white"):
                return
            self.grid.values[r, c] = 0 if right else {"black": 1, "white": 0}[self.tool]
            self.grid.source[r, c] = 1
        self.refresh()

    def mousePressEvent(self, event):
        if self.preview_mode or event.button() == Qt.MouseButton.MiddleButton:
            return super().mousePressEvent(event)
        cell = self.cell_at(event.position())
        if cell is not None and self.preview is None:
            if self.tool == "select":
                self.rect_start = cell
                self.selection = (*cell, *cell)
            else:
                self.editStarted.emit()
                self.dragging = True
                self.visited.clear()
                self.paint_cell(cell, event.button() == Qt.MouseButton.RightButton)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.preview_mode:
            return super().mouseMoveEvent(event)
        cell = self.cell_at(event.position())
        if cell:
            self.hovered.emit(*cell)
            if self.dragging:
                self.paint_cell(cell, bool(event.buttons() & Qt.MouseButton.RightButton))
            elif self.rect_start:
                r, c = self.rect_start
                y, x = cell
                self.selection = (min(r, y), min(c, x), max(r, y), max(c, x))
                self.viewport().update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging:
            self.dragging = False
            self.editFinished.emit()
        self.rect_start = None
        super().mouseReleaseEvent(event)
