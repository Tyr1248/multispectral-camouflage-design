"""
主窗口模块 - 优化版，窗口比例调整为高>宽，字体放大
"""
import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt
from .step_windows import Step1Window
from .window_utils import apply_main_geometry
from utils.config import load_app_config, save_app_state

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.step1_window = None
        self.config = None
        self.init_ui()
        self.load_config()

    def init_ui(self):
        self.setWindowTitle("Intelligent Camouflage Design System")
        apply_main_geometry(self)  # 统一主窗口尺寸并居中

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.setSpacing(35)               # 组件间距
        layout.setContentsMargins(60, 70, 60, 70)  # 调整边距

        # 标题
        title = QLabel("Intelligent Camouflage Design System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 36px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)

        # 副标题
        subtitle = QLabel("cGAN-based Adaptive Camouflage Generation")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 26px; color: #7f8c8d; margin-bottom: 40px;")
        layout.addWidget(subtitle)

        # 开始设计按钮
        start_btn = QPushButton("Start Design")
        start_btn.setStyleSheet("""
            QPushButton {
                font-size: 28px;
                padding: 20px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        start_btn.clicked.connect(self.start_design)
        layout.addWidget(start_btn)

        # 添加弹性空间使按钮居中
        layout.addStretch()

        central_widget.setLayout(layout)

    def load_config(self):
        """加载应用程序配置"""
        self.config = load_app_config()

    def start_design(self):
        """启动设计流程"""
        try:
            self.hide()
            self.step1_window = Step1Window(self)
            self.step1_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start design: {str(e)}")

    def save_state(self):
        """保存应用程序状态"""
        save_app_state(self)

    def closeEvent(self, event):
        """处理关闭事件"""
        self.save_state()
        event.accept()