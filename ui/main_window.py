"""
Main window module - optimized version; window aspect ratio adjusted to height > width, fonts enlarged.
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
        apply_main_geometry(self)  # Apply unified main window size and center

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.setSpacing(35)               # Spacing between widgets
        layout.setContentsMargins(60, 70, 60, 70)  # Adjust margins

        # Title
        title = QLabel("Intelligent Camouflage Design System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 36px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("cGAN-based Adaptive Camouflage Generation")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 26px; color: #7f8c8d; margin-bottom: 40px;")
        layout.addWidget(subtitle)

        # Start design button
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

        # Add stretch to center the button
        layout.addStretch()

        central_widget.setLayout(layout)

    def load_config(self):
        """Load application configuration"""
        self.config = load_app_config()

    def start_design(self):
        """Start the design workflow"""
        try:
            self.hide()
            self.step1_window = Step1Window(self)
            self.step1_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start design: {str(e)}")

    def save_state(self):
        """Save application state"""
        save_app_state(self)

    def closeEvent(self, event):
        """Handle the close event"""
        self.save_state()
        event.accept()