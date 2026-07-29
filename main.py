#!/usr/bin/env python3
"""
智能迷彩设计系统 - 主程序入口
基于cGAN的动态可调迷彩生成系统
支持多组颜色输入、动态多解设计和静态优化设计
"""


import os
import warnings
import matplotlib
import platform
import sys
import traceback

def excepthook(exc_type, exc_value, exc_traceback):
    """全局异常处理"""
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    with open('debug.log', 'a', encoding='utf-8') as f:
        f.write(f"Uncaught exception:\n{error_msg}\n")
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.critical(None, "程序错误", f"发生未捕获的异常:\n{str(exc_value)}")

sys.excepthook = excepthook
# 设置Matplotlib中文字体，避免警告
def setup_chinese_fonts():
    """配置中文字体支持"""

    # 忽略字体警告（临时解决方案）
    warnings.filterwarnings("ignore", message="Glyph.*missing from font")

    # 根据操作系统设置中文字体
    system = platform.system()

    if system == 'Windows':
        # Windows系统中文
        font_candidates = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',  # 黑体
            'SimSun',  # 宋体
            'KaiTi',  # 楷体
            'FangSong',  # 仿宋
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

    # 设置Matplotlib默认字体
    matplotlib.rcParams['font.sans-serif'] = font_candidates
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 打印当前使用的字体
    current_font = matplotlib.rcParams.get('font.sans-serif', ['unknown'])[0]
    print(f"Matplotlib字体设置: {current_font}")

    return font_candidates[0] if font_candidates else None


# 在导入UI模块前设置字体
chinese_font = setup_chinese_fonts()

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from ui.main_window import MainWindow


def main():
    """主函数"""
    # 创建应用程序
    app = QApplication(sys.argv)
    app.setApplicationName("智能迷彩设计系统")
    app.setOrganizationName("Panavy")

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 设置应用程序全局字体（确保Qt控件中的中文正常显示）
    if chinese_font:
        # 创建字体
        font = QFont(chinese_font, 10)
        app.setFont(font)
        print(f"Qt应用程序字体设置为: {chinese_font}")
    else:
        # 尝试设置默认中文字体
        default_fonts = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        for font_name in default_fonts:
            font = QFont(font_name, 10)
            if font.exactMatch():
                app.setFont(font)
                print(f"Qt应用程序字体设置为: {font_name}")
                break

    # 创建并显示主窗口
    main_window = MainWindow()
    main_window.show()

    # 启动事件循环
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()