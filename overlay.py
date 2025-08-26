#!/usr/bin/env python3
"""Simple overlay window displaying live transcriptions.

Launches ``listener.py`` and shows the running transcript from
``transcript.txt``. The window stays on top of other applications and can
be resized normally.
"""

import sys
import subprocess
import pathlib
from PySide6.QtWidgets import (
    QApplication,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor

TRANSCRIPT_FILE = pathlib.Path("transcript.txt")


def read_transcript() -> str:
    if TRANSCRIPT_FILE.exists():
        return TRANSCRIPT_FILE.read_text(encoding="utf-8")
    return ""


class Overlay(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Transcription Overlay")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text.setAcceptRichText(True)
        layout.addWidget(self.text)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_text)
        self.timer.start(500)
        self.update_text()

    def update_text(self) -> None:
        """Refresh overlay with annotated text."""
        lines = read_transcript().splitlines()
        self.text.setHtml("<br><br>".join(lines))
        self.text.moveCursor(QTextCursor.End)
        self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())


def main() -> None:
    listener_proc = subprocess.Popen([sys.executable, "listener.py"])
    app = QApplication(sys.argv)
    window = Overlay()
    window.resize(400, 200)
    window.show()
    ret = app.exec()
    listener_proc.terminate()
    listener_proc.wait()
    sys.exit(ret)


if __name__ == "__main__":
    main()
