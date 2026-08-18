"""
后台工作线程与进度对话框

用于将耗时操作（颜色提取、cGAN 设计生成、光谱计算、迷彩生成）
移出 GUI 主线程，避免窗口"无响应"，并提供可视化进度反馈。

用法:
    def work(report, worker):
        # report(percent, message) 更新进度; percent=-1 表示不确定进度
        # worker.check_cancel() 在可取消任务中周期性调用
        ...
        return result

    run_with_progress(self, "Title", "Starting...", work,
                      on_success, on_error, on_cancel, cancellable=True)
"""

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel,
                             QProgressBar, QPushButton, QMessageBox)
import time


class CancelledError(Exception):
    """用户取消后台任务时抛出"""
    pass


# 防止 Python 垃圾回收掉仍在运行的 worker
_ACTIVE_WORKERS = set()


class WorkerThread(QThread):
    """通用后台工作线程

    信号:
        progress(int, str)   - 进度百分比(0-100, -1 表示不确定)与消息
        succeeded(object)    - 任务完成，携带返回值
        failed(str)          - 任务失败，携带错误信息
        was_cancelled()      - 任务被用户取消
    """

    progress = pyqtSignal(int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    was_cancelled = pyqtSignal()

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self):
        return self._cancelled

    def check_cancel(self):
        """在任务循环中周期性调用；若已取消则抛出 CancelledError"""
        if self._cancelled:
            raise CancelledError()

    def run(self):
        try:
            def report(percent, message=""):
                self.progress.emit(int(percent), str(message))

            result = self._fn(report, self)
            self.succeeded.emit(result)
        except CancelledError:
            self.was_cancelled.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class ProgressDialog(QDialog):
    """模态进度对话框：消息标签 + 进度条 + 可选取消按钮"""

    def __init__(self, title, message, parent=None, cancellable=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(560, 210)
        # 去掉关闭按钮，防止用户在任务进行中强行关掉对话框
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        self.message_label = QLabel(message)
        self.message_label.setStyleSheet("font-size: 20px; color: #2c3e50;")
        self.message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 默认不确定进度（滚动条）
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                font-size: 16px;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                text-align: center;
                height: 26px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self._worker = None
        if cancellable:
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setStyleSheet("font-size: 18px; padding: 6px 20px;")
            cancel_btn.clicked.connect(self._on_cancel_clicked)
            layout.addWidget(cancel_btn, alignment=Qt.AlignCenter)

        # 实时耗时显示（演示用计时器）
        self.elapsed_label = QLabel("Elapsed: 0.0 s")
        self.elapsed_label.setStyleSheet("font-size: 16px; color: #7f8c8d;")
        self.elapsed_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.elapsed_label)

        self.setLayout(layout)

        self._start_time = time.perf_counter()
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._elapsed_timer.start(100)

    def _update_elapsed(self):
        elapsed = time.perf_counter() - self._start_time
        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f} s")

    def elapsed_seconds(self):
        """返回从对话框打开至今的秒数"""
        return time.perf_counter() - self._start_time

    def stop_timer(self):
        self._elapsed_timer.stop()

    def _on_cancel_clicked(self):
        if self._worker is not None:
            self.message_label.setText("Cancelling...")
            self._worker.cancel()

    def update_progress(self, percent, message):
        if message:
            self.message_label.setText(message)
        if percent is not None and percent >= 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percent)
        else:
            self.progress_bar.setRange(0, 0)  # 不确定进度


def run_with_progress(parent, title, message, work_fn,
                      on_success, on_error=None, on_cancel=None,
                      cancellable=False):
    """在后台线程中执行 work_fn，同时显示模态进度对话框

    参数:
        parent:      父窗口（通常是调用者 self）
        title:       对话框标题
        message:     初始提示消息
        work_fn:     callable(report, worker) -> result，在后台线程执行，
                     不得操作任何 GUI 控件
        on_success:  callable(result)，在主线程执行
        on_error:    callable(error_message)，在主线程执行（可选）
        on_cancel:   callable()，用户取消后在主线程执行（可选）
        cancellable: 是否显示取消按钮
    """
    dialog = ProgressDialog(title, message, parent=parent, cancellable=cancellable)
    worker = WorkerThread(work_fn, parent=parent)
    dialog._worker = worker
    _ACTIVE_WORKERS.add(worker)

    def cleanup():
        _ACTIVE_WORKERS.discard(worker)

    def handle_success(result):
        dialog.stop_timer()
        dialog.accept()
        cleanup()
        if on_success:
            on_success(result)

    def handle_error(err):
        dialog.stop_timer()
        dialog.accept()
        cleanup()
        if on_error:
            on_error(err)
        else:
            QMessageBox.critical(parent, "Error", f"Operation failed:\n{err}")

    def handle_cancel():
        dialog.stop_timer()
        dialog.accept()
        cleanup()
        if on_cancel:
            on_cancel()

    worker.progress.connect(dialog.update_progress)
    worker.succeeded.connect(handle_success)
    worker.failed.connect(handle_error)
    worker.was_cancelled.connect(handle_cancel)

    worker.start()
    dialog.exec_()
