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
from PySide6.QtGui import QTextCursor, QPainter, QPen, QColor, QPainterPath

TRANSCRIPT_FILE = pathlib.Path("transcript.txt")
WAVEFORM_FILE = pathlib.Path("waveform.json")


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
        # Increase base font size by +3pt
        f = self.text.font()
        sz = f.pointSize()
        if sz <= 0:
            sz = 12
        f.setPointSize(sz + 3)
        self.text.setFont(f)
        layout.addWidget(self.text)

        # Waveform widget
        self.wave = _Waveform()
        layout.addWidget(self.wave)

        # Track file position and tail-follow state
        self._last_size = 0
        self._follow_tail = True

        # Watch user scroll to toggle follow-tail
        self.text.verticalScrollBar().valueChanged.connect(self._on_scroll)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_text)
        self.timer.start(150)
        self.update_text()

        # Independent waveform refresh timer so it’s not tied to text updates
        self.wave_timer = QTimer(self)
        self.wave_timer.timeout.connect(self.wave.refresh)
        self.wave_timer.start(100)

    def update_text(self) -> None:
        """Incrementally append new transcript without resetting scroll/selection."""
        try:
            st = TRANSCRIPT_FILE.stat()
        except FileNotFoundError:
            return

        sb = self.text.verticalScrollBar()
        at_bottom_before = sb.value() >= sb.maximum() - 4
        has_sel = self.text.textCursor().hasSelection()

        # If file truncated (new session), reload all
        if st.st_size < self._last_size:
            data = read_transcript()
            self.text.setHtml(data.replace("\n", "<br>"))
            self._last_size = st.st_size
        elif st.st_size > self._last_size:
            with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
                f.seek(self._last_size)
                chunk = f.read()
                self._last_size = st.st_size
            if chunk:
                # Append each new line as HTML; lines already contain HTML
                for line in chunk.splitlines():
                    if line == "":
                        self.text.moveCursor(QTextCursor.End)
                        self.text.insertHtml("<br>")
                    else:
                        self.text.moveCursor(QTextCursor.End)
                        self.text.insertHtml(line + "<br>")

        # Auto-scroll only if user is at bottom and following tail and not selecting
        if self._follow_tail and not has_sel and at_bottom_before:
            self.text.moveCursor(QTextCursor.End)
            sb.setValue(sb.maximum())

    def _on_scroll(self) -> None:
        sb = self.text.verticalScrollBar()
        # Consider within 8 px of bottom as at-bottom
        at_bottom = sb.value() >= sb.maximum() - 8
        # If user scrolls away from bottom, stop following
        self._follow_tail = at_bottom
        # Refresh waveform
        self.wave.refresh()


class _Waveform(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(80)
        self._data = []  # list of ints [-100..100]

    def refresh(self) -> None:
        try:
            if WAVEFORM_FILE.exists():
                import json
                d = json.loads(WAVEFORM_FILE.read_text(encoding="utf-8"))
                arr = d.get("data") or []
                self._data = arr[-1000:]
        except Exception:
            pass
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()
        # subtle background
        p.fillRect(rect, QColor(24, 24, 24, 20))
        mid = rect.center().y()
        # baseline
        p.setPen(QPen(QColor(136, 136, 136), 1))
        p.drawLine(rect.left(), mid, rect.right(), mid)
        if not self._data or len(self._data) < 2:
            p.end(); return
        n = len(self._data)
        w = max(1, rect.width() - 2)
        h = max(1, rect.height() - 8)
        scale = (h/2) / 100.0
        path = QPainterPath()
        def xy(i, v):
            x = rect.left() + 1 + int(i * w / max(1, n-1))
            y = int(mid - v * scale)
            return x, y
        x0, y0 = xy(0, self._data[0])
        path.moveTo(x0, y0)
        for i in range(1, n):
            x, y = xy(i, self._data[i])
            path.lineTo(x, y)
        p.setPen(QPen(QColor(30, 144, 255), 1.5))
        p.drawPath(path)
        p.end()


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
