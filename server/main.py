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

# --- load .env to pick up SHOW_DETAILS_LINE and others ---
def _load_env_file(path: Path):
    try:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip()
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass

# try project .env
_load_env_file(ROOT / ".env")
TRANSCRIPT_FILE = ROOT / "transcript.txt"
WAVEFORM_FILE = ROOT / "waveform.json"
import re
from typing import List, Dict
try:
    from fugashi import Tagger as _Tagger
    from pykakasi import kakasi as _kakasi
    _tagger = _Tagger()
    _kks = _kakasi(); _kks.setMode("J","H"); _conv = _kks.getConverter()
except Exception:
    _tagger = None; _conv = None
ASR_DEBUG_FILE = ROOT / "asr_debug.json"

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
    show_details = (os.getenv("SHOW_DETAILS_LINE", "false").lower() in {"1","true","yes","y"})
    show_asr_debug = (os.getenv("SHOW_ASR_DEBUG", "false").lower() in {"1","true","yes","y"})
    return {
        "status": "ok",
        "listener_running": (listener_proc is not None and listener_proc.poll() is None),
        "transcript_mtime": lt,
        "waveform_mtime": lw,
        "show_details": show_details,
        "show_asr_debug": show_asr_debug,
    }


@app.get("/api/overlay/transcript")
def get_transcript():
    text = _read_text(TRANSCRIPT_FILE)
    # For compatibility, return both raw text and html alias
    return JSONResponse({"text": text, "html": text})


@app.get("/api/overlay/waveform")
def get_waveform():
    data = _read_wave(WAVEFORM_FILE)
    return JSONResponse({"data": data})


def _to_hira(s: str) -> str:
    if _conv is None: return s
    try:
        return "".join(p["hira"] for p in _conv.convert(s))
    except Exception:
        return s

def _reading_for(jp: str) -> str:
    if _tagger is None: return _to_hira(jp)
    try:
        toks = list(_tagger(jp))
        if not toks: return _to_hira(jp)
        kana = []
        for m in toks:
            k = getattr(m.feature, 'pron', None) or getattr(m.feature, 'kana', None)
            kana.append(k if k else m.surface)
        return _to_hira("".join(kana))
    except Exception:
        return _to_hira(jp)

_KANJI = re.compile(r"[一-龯々〆ヵヶ]")
_KATA = re.compile(r"[ァ-ヴ]")

def _has_kana_or_kanji(s: str) -> bool:
    return bool(_KANJI.search(s) or _KATA.search(s))

def _jisho_best_gloss(word: str) -> str:
    try:
        import requests
        url = f"https://jisho.org/api/v1/search/words?keyword={requests.utils.quote(word)}"
        r = requests.get(url, timeout=5)
        j = r.json(); d = j.get('data') or []
        if not d: return ''
        senses = d[0].get('senses') or []
        for s in senses:
            ed = s.get('english_definitions') or []
            if ed: return ed[0]
    except Exception:
        return ''
    return ''

def _extract_suggestions(text: str, exclude: set, max_items=30) -> List[Dict[str,str]]:
    out = []
    seen = set()
    for line in text.splitlines()[-200:]:
        if line.strip().startswith('Vocab:'): continue
        # candidates = contiguous Katakana or Kanji tokens
        for m in re.finditer(r"[\u30A0-\u30FF]+|[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF々〆ヵヶ]+[\wぁ-ゖァ-ヺー]*", line):
            jp = m.group(0).strip()
            if not jp or jp in seen or jp in exclude: continue
            seen.add(jp)
            rd = _reading_for(jp)
            en = _jisho_best_gloss(jp)
            out.append({"ja": jp, "read": rd, "en": en})
            if len(out) >= max_items: return out
    return out

@app.get("/api/overlay/suggestions")
def get_suggestions():
    try:
        text = _read_text(TRANSCRIPT_FILE)
        # extract exclude set from existing Vocab: lines
        exclude = set()
        for line in text.splitlines():
            if line.startswith('Vocab:'):
                # surf(reading、meaning) ・ surf(...)
                parts = [p.strip() for p in line[6:].split('・') if p.strip()]
                for p in parts:
                    if '(' in p:
                        surf = p.split('(', 1)[0].strip()
                        if surf: exclude.add(surf)
        items = _extract_suggestions(text, exclude)
        return JSONResponse({"items": items})
    except Exception:
        return JSONResponse({"items": []})


@app.get("/api/debug/asr")
def get_asr_debug():
    try:
        import json
        if ASR_DEBUG_FILE.exists():
            arr = json.loads(ASR_DEBUG_FILE.read_text(encoding="utf-8")) or []
            # normalize
            out = []
            for it in arr:
                out.append({
                    "id": it.get("id"),
                    "ts": it.get("ts"),
                    "openai": it.get("openai") or {},
                    "remote": it.get("remote") or {},
                    "local": it.get("local") or {},
                })
            return JSONResponse({"items": out})
    except Exception:
        pass
    return JSONResponse({"items": []})
