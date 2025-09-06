import subprocess
import sys
import os
import signal
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse


ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_FILE = ROOT / "transcript.txt"
WAVEFORM_FILE = ROOT / "waveform.json"

listener_proc: Optional[subprocess.Popen] = None


def _read_text(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""


def _read_wave(path: Path):
    try:
        if path.exists():
            import json
            d = json.loads(path.read_text(encoding="utf-8"))
            arr = d.get("data") or []
            # ensure list of ints in [-100,100]
            out = []
            for v in arr:
                try:
                    iv = int(v)
                except Exception:
                    continue
                if iv < -100:
                    iv = -100
                if iv > 100:
                    iv = 100
                out.append(iv)
            return out
    except Exception:
        pass
    return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global listener_proc
    # Spawn listener.py with the current Python, inheriting env/conda
    try:
        listener_proc = subprocess.Popen([sys.executable, str(ROOT / "listener.py")])
    except Exception:
        listener_proc = None
    try:
        yield
    finally:
        # Gracefully terminate listener on shutdown
        proc = listener_proc
        listener_proc = None
        try:
            if proc and proc.poll() is None:
                if os.name == "nt":
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
def health():
    try:
        lt = TRANSCRIPT_FILE.stat().st_mtime if TRANSCRIPT_FILE.exists() else None
    except Exception:
        lt = None
    try:
        lw = WAVEFORM_FILE.stat().st_mtime if WAVEFORM_FILE.exists() else None
    except Exception:
        lw = None
    return {
        "status": "ok",
        "listener_running": (listener_proc is not None and listener_proc.poll() is None),
        "transcript_mtime": lt,
        "waveform_mtime": lw,
    }


@app.get("/api/overlay/transcript")
def get_transcript():
    html = _read_text(TRANSCRIPT_FILE)
    return JSONResponse({"html": html})


@app.get("/api/overlay/waveform")
def get_waveform():
    data = _read_wave(WAVEFORM_FILE)
    return JSONResponse({"data": data})

