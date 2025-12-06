#!/usr/bin/python
# -*- coding: utf-8 -*-


try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from lib import distance
import math

# 方案1：保持绿色边框，填充改为淡蓝色
DEFAULT_LINE_COLOR = QColor(0, 255, 0, 128)          # 绿色边框（保持不变）
DEFAULT_FILL_COLOR = QColor(100, 150, 255, 50)       # 淡蓝色填充，透明度50（很淡）
DEFAULT_SELECT_LINE_COLOR = QColor(255, 255, 255)    # 白色选中边框（保持不变）
DEFAULT_SELECT_FILL_COLOR = QColor(100, 150, 255, 60) # 选中时的淡蓝色填充，透明度80
DEFAULT_VERTEX_FILL_COLOR = QColor(0, 255, 0, 255)   # 绿色顶点（保持不变）
DEFAULT_HVERTEX_FILL_COLOR = QColor(255, 0, 0)       # 红色高亮顶点（保持不变）

# 新增：直线模式的颜色
DEFAULT_LINE_MODE_COLOR = QColor(255, 165, 0, 200)  # 橙色


class Shape(object):
    P_SQUARE, P_ROUND = range(2)

    MOVE_VERTEX, NEAR_VERTEX = range(2)

    # The following class variables influence the drawing
    # of _all_ shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    def __init__(self, label=None, line_color=None, difficult=False):
        self.label = label
        self.points = []
        self.fill = False
        self.selected = False
        self.difficult = difficult

        self.direction = 0  # added by hy
        self.center = None  # added by hy
        self.isRotated = True

        # 新增：标识是否为直线模式
        self.isLine = False
        self.lineWidth = 0.01  # 直线转换为旋转矩形时的宽度（短边）

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

    def rotate(self, theta):
        for i, p in enumerate(self.points):
            self.points[i] = self.rotatePoint(p, theta)
        self.direction -= theta
        self.direction = self.direction % (2 * math.pi)

    def rotatePoint(self, p, theta):
        order = p - self.center
        cosTheta = math.cos(theta)
        sinTheta = math.sin(theta)
        pResx = cosTheta * order.x() + sinTheta * order.y()
        pResy = -sinTheta * order.x() + cosTheta * order.y()
        pRes = QPointF(self.center.x() + pResx, self.center.y() + pResy)
        return pRes

    def close(self):
        if self.isLine:
            # 直线模式：只需要两个点
            if len(self.points) >= 2:
                self.center = QPointF(
                    (self.points[0].x() + self.points[1].x()) / 2,
                    (self.points[0].y() + self.points[1].y()) / 2
                )
                # 计算直线的角度
                dx = self.points[1].x() - self.points[0].x()
                dy = self.points[1].y() - self.points[0].y()
                self.direction = math.atan2(dy, dx)
                self._closed = True
        else:
            # 原有的旋转矩形模式
            self.center = QPointF(
                (self.points[0].x() + self.points[2].x()) / 2,
                (self.points[0].y() + self.points[2].y()) / 2
            )
            self._closed = True


    def reachMaxPoints(self):
        if self.isLine:
            # 直线模式：只需要2个点
            if len(self.points) >= 2:
                return True
            return False
        else:
            # 旋转矩形模式：需要4个点
            if len(self.points) >= 4:
                return True
            return False

    def addPoint(self, point):
        if self.isLine:
            # 直线模式：只接受2个点
            if len(self.points) < 2:
                self.points.append(point)
            if len(self.points) == 2:
                self.close()
        else:
            # 原有逻辑
            if self.points and len(self.points) == 4 and point == self.points[0]:
                self.close()
            else:
                self.points.append(point)

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def paint(self, painter):
        if self.points:
            color = self.select_line_color if self.selected else self.line_color

            # 直线模式使用不同的颜色
            if self.isLine and not self.selected:
                color = DEFAULT_LINE_MODE_COLOR

            pen = QPen(color)
            # 直线模式使用更粗的线条
            pen.setWidth(max(1, int(round(3.0 / self.scale))) if self.isLine else max(1, int(round(2.0 / self.scale))))
            painter.setPen(pen)

            line_path = QPainterPath()
            vrtx_path = QPainterPath()

            line_path.moveTo(self.points[0])

            for i, p in enumerate(self.points):
                line_path.lineTo(p)
                self.drawVertex(vrtx_path, i)

            if self.isClosed() and not self.isLine:
                # 只有非直线模式才闭合路径
                line_path.lineTo(self.points[0])

            painter.drawPath(line_path)
            painter.drawPath(vrtx_path)
            painter.fillPath(vrtx_path, self.vertex_fill_color)

            if self.fill and not self.isLine:
                # 直线模式不填充
                color = self.select_fill_color if self.selected else self.fill_color
                painter.fillPath(line_path, color)

            # 绘制中心点
            if self.center is not None:
                center_path = QPainterPath()
                d = self.point_size / self.scale
                center_path.addRect(self.center.x() - d / 2, self.center.y() - d / 2, d, d)
                painter.drawPath(center_path)
                if self.isRotated or self.isLine:
                    painter.fillPath(center_path, self.vertex_fill_color)
                else:
                    painter.fillPath(center_path, QColor(0, 0, 0))

                # ====== 新增：绘制两条中心线（方案2）======
                if self.isClosed() and len(self.points) >= 4 and not self.isLine:
                    # 设置中心线的颜色和样式
                    center_line_color = QColor(100, 150, 255, 200)  # 淡蓝色
                    center_pen = QPen(center_line_color)
                    center_pen.setWidth(max(1, int(round(1.5 / self.scale))))  # 线条粗细
                    center_pen.setStyle(Qt.DashLine)  # 虚线样式
                    painter.setPen(center_pen)

                    # 计算旋转矩形的宽度和高度的一半
                    half_width = distance(self.points[0] - self.points[1]) / 2
                    half_height = distance(self.points[1] - self.points[2]) / 2

                    # 使用旋转角度
                    angle = self.direction
                    cos_angle = math.cos(angle)
                    sin_angle = math.sin(angle)

                    # 计算水平中心线的两个端点
                    h_line_p1 = QPointF(
                        self.center.x() - half_width * cos_angle,
                        self.center.y() - half_width * sin_angle
                    )
                    h_line_p2 = QPointF(
                        self.center.x() + half_width * cos_angle,
                        self.center.y() + half_width * sin_angle
                    )

                    # 计算垂直中心线的两个端点
                    v_line_p1 = QPointF(
                        self.center.x() + half_height * sin_angle,
                        self.center.y() - half_height * cos_angle
                    )
                    v_line_p2 = QPointF(
                        self.center.x() - half_height * sin_angle,
                        self.center.y() + half_height * cos_angle
                    )

                    # 绘制两条中心线
                    painter.drawLine(h_line_p1, h_line_p2)  # 水平中心线
                    painter.drawLine(v_line_p1, v_line_p2)  # 垂直中心线
                # ====== 中心线绘制结束 ======


    def paintNormalCenter(self, painter):
        if self.center is not None:
            center_path = QPainterPath()
            d = self.point_size / self.scale
            center_path.addRect(self.center.x() - d / 2, self.center.y() - d / 2, d, d)
            painter.drawPath(center_path)
            if not self.isRotated and not self.isLine:
                painter.fillPath(center_path, QColor(0, 0, 0))

    def drawVertex(self, path, i):
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightIndex is not None:
            self.vertex_fill_color = self.hvertex_fill_color
        else:
            self.vertex_fill_color = Shape.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearestVertex(self, point, epsilon):
        for i, p in enumerate(self.points):
            if distance(p - point) <= epsilon:
                return i
        return None

    def containsPoint(self, point):
        return self.makePath().contains(point)

    def makePath(self):
        path = QPainterPath(self.points[0])
        for p in self.points[1:]:
            path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]
        if self.center:
            self.center = self.center + offset

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset
        # 如果是直线模式，移动顶点后需要重新计算中心和角度
        if self.isLine and len(self.points) == 2:
            self.center = QPointF(
                (self.points[0].x() + self.points[1].x()) / 2,
                (self.points[0].y() + self.points[1].y()) / 2
            )
            dx = self.points[1].x() - self.points[0].x()
            dy = self.points[1].y() - self.points[0].y()
            self.direction = math.atan2(dy, dx)

    def highlightVertex(self, i, action):
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        self._highlightIndex = None

    def copy(self):
        shape = Shape("%s" % self.label)
        shape.points = [p for p in self.points]

        shape.center = self.center
        shape.direction = self.direction
        shape.isRotated = self.isRotated
        shape.isLine = self.isLine  # 新增
        shape.lineWidth = self.lineWidth  # 新增

        shape.fill = self.fill
        shape.selected = self.selected
        shape._closed = self._closed
        if self.line_color != Shape.line_color:
            shape.line_color = self.line_color
        if self.fill_color != Shape.fill_color:
            shape.fill_color = self.fill_color
        shape.difficult = self.difficult
        return shape

    # 新增：将直线转换为旋转矩形框的参数
    def getRotatedBoxParams(self):
        """
        返回旋转矩形框的参数，用于保存到 XML
        如果是直线模式，自动转换为窄矩形
        """
        if not self.isClosed():
            return None

        if self.isLine:
            # 直线模式：计算窄矩形的参数
            if len(self.points) < 2:
                return None

            # 计算长度（w）
            dx = self.points[1].x() - self.points[0].x()
            dy = self.points[1].y() - self.points[0].y()
            length = math.sqrt(dx * dx + dy * dy)

            return {
                'cx': self.center.x(),
                'cy': self.center.y(),
                'w': length,
                'h': self.lineWidth,  # 使用固定的窄宽度
                'angle': self.direction
            }
        else:
            # 旋转矩形模式：计算实际的矩形参数
            if len(self.points) < 4:
                return None

            # 计算宽度和高度
            width = distance(self.points[0] - self.points[1])
            height = distance(self.points[1] - self.points[2])

            return {
                'cx': self.center.x(),
                'cy': self.center.y(),
                'w': width,
                'h': height,
                'angle': self.direction
            }

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value