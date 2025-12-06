#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测结果查看器 - roLabelImg扩展
在主窗口左下角显示预测结果图片的可停靠窗口
"""

import os

try:
    from PyQt5.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QLabel,
                                 QPushButton, QHBoxLayout, QScrollArea, QSizePolicy)
    from PyQt5.QtCore import Qt, QSize, pyqtSignal
    from PyQt5.QtGui import QPixmap, QImage
except ImportError:
    from PyQt4.QtGui import (QDockWidget, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QHBoxLayout, QScrollArea, QSizePolicy, QPixmap, QImage)
    from PyQt4.QtCore import Qt, QSize, pyqtSignal


class ResultImageViewer(QWidget):
    """结果图片查看器控件"""

    def __init__(self, parent=None):
        super(ResultImageViewer, self).__init__(parent)
        self.current_pixmap = None
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # 控制按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(2)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_out = QPushButton("-")
        self.btn_fit = QPushButton("适应")
        self.btn_original = QPushButton("1:1")
        self.btn_swap = QPushButton("⇄ 交换")
        self.btn_swap.setToolTip("交换主界面与结果图片的显示位置")

        # 设置按钮大小
        for btn in [self.btn_zoom_in, self.btn_zoom_out, self.btn_fit, self.btn_original]:
            btn.setMaximumWidth(50)
            btn.setMaximumHeight(25)

        # 交换按钮稍大一些
        self.btn_swap.setMaximumWidth(70)
        self.btn_swap.setMaximumHeight(25)
        self.btn_swap.setStyleSheet("QPushButton { font-weight: bold; color: #0066cc; }")

        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_fit.clicked.connect(self.fit_to_window)
        self.btn_original.clicked.connect(self.original_size)
        # btn_swap 的连接将在 ResultViewerDock 中处理

        btn_layout.addWidget(self.btn_zoom_in)
        btn_layout.addWidget(self.btn_zoom_out)
        btn_layout.addWidget(self.btn_fit)
        btn_layout.addWidget(self.btn_original)
        btn_layout.addWidget(self.btn_swap)
        btn_layout.addStretch()

        # 信息标签
        self.info_label = QLabel("预测结果: 未加载")
        self.info_label.setMaximumHeight(20)
        self.info_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")

        # 图片显示区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #2b2b2b; }")
        self.image_label.setText("未加载结果图片")
        self.image_label.setMinimumSize(200, 150)

        self.scroll_area.setWidget(self.image_label)

        # 添加到布局
        layout.addWidget(self.info_label)
        layout.addLayout(btn_layout)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

    def load_image(self, image_path):
        """加载图片"""
        if not image_path or not os.path.exists(image_path):
            self.clear_image()
            self.info_label.setText("预测结果: 未找到")
            return False

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.clear_image()
            self.info_label.setText("预测结果: 加载失败")
            return False

        self.original_pixmap = pixmap
        self.current_pixmap = pixmap
        self.zoom_factor = 1.0

        # 默认适应窗口
        self.fit_to_window()

        # 更新信息
        filename = os.path.basename(image_path)
        self.info_label.setText(f"预测结果: {filename}")
        return True

    def clear_image(self):
        """清空图片"""
        self.image_label.clear()
        self.image_label.setText("未加载结果图片")
        self.original_pixmap = None
        self.current_pixmap = None
        self.zoom_factor = 1.0
        self.info_label.setText("预测结果: 未加载")

    def zoom_in(self):
        """放大"""
        if self.original_pixmap:
            self.zoom_factor *= 1.2
            self.update_image()

    def zoom_out(self):
        """缩小"""
        if self.original_pixmap:
            self.zoom_factor /= 1.2
            self.update_image()

    def fit_to_window(self):
        """适应窗口"""
        if not self.original_pixmap:
            return

        # 获取可用空间
        available_size = self.scroll_area.size()
        available_width = available_size.width() - 10
        available_height = available_size.height() - 10

        # 计算缩放比例
        pixmap_size = self.original_pixmap.size()
        width_ratio = available_width / pixmap_size.width()
        height_ratio = available_height / pixmap_size.height()

        self.zoom_factor = min(width_ratio, height_ratio)
        self.update_image()

    def original_size(self):
        """原始大小"""
        if self.original_pixmap:
            self.zoom_factor = 1.0
            self.update_image()

    def update_image(self):
        """更新图片显示"""
        if not self.original_pixmap:
            return

        # 计算新尺寸
        new_size = self.original_pixmap.size() * self.zoom_factor

        # 缩放图片
        scaled_pixmap = self.original_pixmap.scaled(
            new_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.current_pixmap = scaled_pixmap
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())


class ResultViewerDock(QDockWidget):
    """结果查看器停靠窗口"""

    # 添加交换信号
    swapRequested = pyqtSignal()

    def __init__(self, parent=None):
        super(ResultViewerDock, self).__init__("预测结果查看器", parent)

        # 创建查看器widget
        self.viewer = ResultImageViewer(self)
        self.setWidget(self.viewer)

        # 连接交换按钮的信号
        self.viewer.btn_swap.clicked.connect(self.swapRequested.emit)

        # 设置停靠窗口特性
        self.setFeatures(
            QDockWidget.DockWidgetClosable |
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable
        )

        # 设置最小尺寸
        self.setMinimumSize(250, 200)

    def load_result_image(self, image_path):
        """加载结果图片"""
        return self.viewer.load_image(image_path)

    def clear(self):
        """清空显示"""
        self.viewer.clear_image()