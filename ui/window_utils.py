"""
Window geometry utilities - unify window sizes and positions.

Main-flow windows (main window, Step1-4, result windows) all use MAIN_SIZE
and are centered; auxiliary popups (color extraction confirmation, color
input) use POPUP_SIZE and are centered. All sizes shrink automatically to
fit the available screen area, avoiding overflow on small screens.
"""

from PyQt5.QtWidgets import QApplication

# Unified size for main-flow windows
MAIN_SIZE = (1280, 800)
# Size for auxiliary/confirmation popups
POPUP_SIZE = (1100, 700)

# Maximum fraction of the available screen area a window may occupy
_SCREEN_FILL_RATIO = 0.92


def _available_geometry():
    screen = QApplication.primaryScreen()
    if screen is None:
        return None
    return screen.availableGeometry()


def apply_window_geometry(window, size=MAIN_SIZE, set_minimum=True):
    """Resize the window to the unified size and center it on screen

    Args:
        window:       target QWidget/QMainWindow
        size:         (width, height) tuple
        set_minimum:  whether to set the target size as the minimum size
                      (prevents a later adjustSize() from shrinking it)
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
    """Main-flow window: unified size + centered"""
    apply_window_geometry(window, MAIN_SIZE, set_minimum=True)


def apply_popup_geometry(window):
    """Auxiliary/confirmation popup: dedicated size + centered"""
    apply_window_geometry(window, POPUP_SIZE, set_minimum=True)
