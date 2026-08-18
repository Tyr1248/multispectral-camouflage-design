"""
Background worker thread and progress dialog.

Moves time-consuming operations (color extraction, cGAN design generation,
spectrum calculation, camouflage generation) off the GUI main thread to
avoid "not responding" windows, and provides visual progress feedback.

Usage:
    def work(report, worker):
        # report(percent, message) updates progress; percent=-1 means indeterminate
        # call worker.check_cancel() periodically in cancellable tasks
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
    """Raised when the user cancels a background task"""
    pass


# Prevent Python from garbage-collecting workers that are still running
_ACTIVE_WORKERS = set()


class WorkerThread(QThread):
    """Generic background worker thread

    Signals:
        progress(int, str)   - progress percentage (0-100, -1 means indeterminate) and message
        succeeded(object)    - task finished, carries the return value
        failed(str)          - task failed, carries the error message
        was_cancelled()      - task was cancelled by the user
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
        """Call periodically inside the task loop; raises CancelledError if cancelled"""
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
    """Modal progress dialog: message label + progress bar + optional cancel button"""

    def __init__(self, title, message, parent=None, cancellable=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(560, 210)
        # Remove the close button so the user cannot force-close the dialog mid-task
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        self.message_label = QLabel(message)
        self.message_label.setStyleSheet("font-size: 20px; color: #2c3e50;")
        self.message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress by default (busy indicator)
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

        # Live elapsed-time display (demo timer)
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
        """Return the seconds elapsed since the dialog was opened"""
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
            self.progress_bar.setRange(0, 0)  # Indeterminate progress


def run_with_progress(parent, title, message, work_fn,
                      on_success, on_error=None, on_cancel=None,
                      cancellable=False):
    """Run work_fn in a background thread while showing a modal progress dialog

    Args:
        parent:      parent window (usually the caller's self)
        title:       dialog title
        message:     initial status message
        work_fn:     callable(report, worker) -> result, runs in the
                     background thread; must not touch any GUI widget
        on_success:  callable(result), runs in the main thread
        on_error:    callable(error_message), runs in the main thread (optional)
        on_cancel:   callable(), runs in the main thread after user cancel (optional)
        cancellable: whether to show a cancel button
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
