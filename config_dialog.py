#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型配置对话框
"""

import os
import json

try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *


class ModelConfigDialog(QDialog):
    """模型配置对话框"""

    def __init__(self, parent=None):
        super(ModelConfigDialog, self).__init__(parent)
        self.setWindowTitle("模型配置")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        # 配置文件路径
        self.config_file = "model_config.json"

        # 初始化UI
        self.init_ui()

        # 加载已保存的配置
        self.load_config()

    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout()

        # 标题
        title_label = QLabel("模型路径配置")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title_label)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # 表单布局
        form_layout = QFormLayout()

        # 指针检测模型
        self.zhizhen_path = QLineEdit()
        zhizhen_btn = QPushButton("浏览...")
        zhizhen_btn.clicked.connect(lambda: self.browse_file(self.zhizhen_path, "指针检测模型 (*.pt)"))
        zhizhen_layout = QHBoxLayout()
        zhizhen_layout.addWidget(self.zhizhen_path)
        zhizhen_layout.addWidget(zhizhen_btn)
        form_layout.addRow("指针检测模型:", zhizhen_layout)

        # 液位检测模型
        self.youwei_path = QLineEdit()
        youwei_btn = QPushButton("浏览...")
        youwei_btn.clicked.connect(lambda: self.browse_file(self.youwei_path, "液位检测模型 (*.pt)"))
        youwei_layout = QHBoxLayout()
        youwei_layout.addWidget(self.youwei_path)
        youwei_layout.addWidget(youwei_btn)
        form_layout.addRow("液位检测模型:", youwei_layout)

        # 表盘检测模型
        self.dial_path = QLineEdit()
        dial_btn = QPushButton("浏览...")
        dial_btn.clicked.connect(lambda: self.browse_file(self.dial_path, "表盘检测模型 (*.pt)"))
        dial_layout = QHBoxLayout()
        dial_layout.addWidget(self.dial_path)
        dial_layout.addWidget(dial_btn)
        form_layout.addRow("表盘检测模型:", dial_layout)

        # 置信度阈值
        self.conf_thresh = QDoubleSpinBox()
        self.conf_thresh.setRange(0.0, 1.0)
        self.conf_thresh.setSingleStep(0.05)
        self.conf_thresh.setValue(0.2)
        form_layout.addRow("置信度阈值:", self.conf_thresh)

        layout.addLayout(form_layout)

        # 添加弹性空间
        layout.addStretch()

        # 按钮
        btn_layout = QHBoxLayout()

        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.save_config)

        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def browse_file(self, line_edit, file_filter):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            "",
            file_filter
        )

        if file_path:
            line_edit.setText(file_path)

    def get_config(self):
        """获取配置"""
        return {
            'zhizhen_file': self.zhizhen_path.text(),
            'youwei_file': self.youwei_path.text(),
            'dial_detection_model': self.dial_path.text(),
            'conf_thresh': self.conf_thresh.value()
        }

    def save_config(self):
        """保存配置到文件"""
        config = self.get_config()

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)

            QMessageBox.information(self, "成功", "配置已保存！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置失败：{str(e)}")

    def load_config(self):
        """从文件加载配置"""
        if not os.path.exists(self.config_file):
            return

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.zhizhen_path.setText(config.get('zhizhen_file', ''))
            self.youwei_path.setText(config.get('youwei_file', ''))
            self.dial_path.setText(config.get('dial_detection_model', ''))
            self.conf_thresh.setValue(config.get('conf_thresh', 0.2))

        except Exception as e:
            print(f"加载配置失败：{str(e)}")