"""
窗口几何工具 - 统一各窗口的大小与位置

主流程窗口（主窗口、Step1-4、结果窗口）统一使用 MAIN_SIZE 并居中；
辅助弹窗（颜色提取确认、颜色输入）使用 POPUP_SIZE 并居中。
所有尺寸都会根据屏幕可用区域自动收缩，避免小屏幕溢出。
"""

from PyQt5.QtWidgets import QApplication

# 主流程窗口统一尺寸
MAIN_SIZE = (1280, 800)
# 辅助/确认弹窗尺寸
POPUP_SIZE = (1100, 700)

# 窗口不超过屏幕可用区域的比例
_SCREEN_FILL_RATIO = 0.92


def _available_geometry():
    screen = QApplication.primaryScreen()
    if screen is None:
        return None
    return screen.availableGeometry()


def apply_window_geometry(window, size=MAIN_SIZE, set_minimum=True):
    """将窗口调整为统一尺寸并移动到屏幕中央

    参数:
        window:       目标 QWidget/QMainWindow
        size:         (宽, 高) 元组
        set_minimum:  是否将目标尺寸设为最小尺寸（防止后续 adjustSize 收缩）
    """
    width, height = size
    avail = _available_geometry()
    if avail is not None:
        width = min(width, int(avail.width() * _SCREEN_FILL_RATIO))
        height = min(height, int(avail.height() * _SCREEN_FILL_RATIO))

    if set_minimum:
        window.setMinimumSize(width, height)
    window.resize(width, height)

    if avail is not None:
        frame = window.frameGeometry()
        frame.moveCenter(avail.center())
        window.move(frame.topLeft())


def apply_main_geometry(window):
    """主流程窗口：统一大小 + 居中"""
    apply_window_geometry(window, MAIN_SIZE, set_minimum=True)


def apply_popup_geometry(window):
    """辅助/确认弹窗：独立尺寸 + 居中"""
    apply_window_geometry(window, POPUP_SIZE, set_minimum=True)
