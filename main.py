#!/usr/bin/env python3
"""
Smart Camouflage Design System - Main entry point
Dynamic tunable camouflage generation system based on cGAN
Supports multi-color input, dynamic multi-solution design, and static optimized design
"""


import os
import warnings
import matplotlib
import platform
import sys
import traceback

def excepthook(exc_type, exc_value, exc_traceback):
    """Global exception handler"""
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    with open('debug.log', 'a', encoding='utf-8') as f:
        f.write(f"Uncaught exception:\n{error_msg}\n")
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.critical(None, "程序错误", f"发生未捕获的异常:\n{str(exc_value)}")

sys.excepthook = excepthook
# Set Matplotlib Chinese fonts to avoid warnings
def setup_chinese_fonts():
    """Configure Chinese font support"""

    # Ignore font warnings (temporary workaround)
    warnings.filterwarnings("ignore", message="Glyph.*missing from font")

    # Select Chinese fonts based on the operating system
    system = platform.system()

    if system == 'Windows':
        # Chinese fonts on Windows
        font_candidates = [
            'Microsoft YaHei',
            'SimHei',
            'SimSun',
            'KaiTi',
            'FangSong',
            'Arial Unicode MS'
        ]
    elif system == 'Darwin':  # macOS
        font_candidates = [
            'Arial Unicode MS',
            'PingFang SC',
            'Heiti SC',
            'Hiragino Sans GB',
            'STHeiti'
        ]
    else:  # Linux
        font_candidates = [
            'WenQuanYi Micro Hei',
            'Noto Sans CJK SC',
            'AR PL UMing CN',
            'DejaVu Sans'
        ]

    # Set Matplotlib default fonts
    matplotlib.rcParams['font.sans-serif'] = font_candidates
    matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

    # Print the font currently in use
    current_font = matplotlib.rcParams.get('font.sans-serif', ['unknown'])[0]
    print(f"Matplotlib font set to: {current_font}")

    return font_candidates[0] if font_candidates else None


# Set fonts before importing UI modules
chinese_font = setup_chinese_fonts()

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from ui.main_window import MainWindow


def main():
    """Main function"""
    # Create the application
    app = QApplication(sys.argv)
    app.setApplicationName("智能迷彩设计系统")
    app.setOrganizationName("Panavy")

    # Set the application style
    app.setStyle('Fusion')

    # Set the global application font (ensure Chinese renders correctly in Qt widgets)
    if chinese_font:
        # Create the font
        font = QFont(chinese_font, 10)
        app.setFont(font)
        print(f"Qt application font set to: {chinese_font}")
    else:
        # Fall back to default Chinese fonts
        default_fonts = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        for font_name in default_fonts:
            font = QFont(font_name, 10)
            if font.exactMatch():
                app.setFont(font)
                print(f"Qt application font set to: {font_name}")
                break

    # Create and show the main window
    main_window = MainWindow()
    main_window.show()

    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()