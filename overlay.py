#!/usr/bin/env python3
"""Simple overlay window displaying live transcriptions.

Launches ``listener.py`` and shows the running transcript from
``transcript.txt``. The window stays on top of other applications and can
be resized normally.
"""

import sys
import subprocess
import pathlib
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, QTimer

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
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label.setTextFormat(Qt.RichText)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_text)
        self.timer.start(500)
        self.update_text()

    def update_text(self) -> None:
        """Refresh overlay with annotated text and English translation."""
        lines = read_transcript().splitlines()
        chunks: list[str] = []
        # Pair every two lines as [annotated JP, translation]
        for i in range(0, len(lines), 2):
            jp = lines[i]
            en = lines[i + 1] if i + 1 < len(lines) else ""
            if en:
                chunks.append(f"{jp}<br><span style='color: gray'>{en}</span>")
            else:
                chunks.append(jp)
        self.label.setText("<br><br>".join(chunks))


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
