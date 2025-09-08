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
            k = k.strip()
            v = v.strip()
            # Remove comments from values
            if "#" in v:
                v = v.split("#")[0].strip()
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass

# try project .env
_load_env_file(ROOT / ".env")
TRANSCRIPT_FILE = ROOT / "transcript.txt"
WAVEFORM_FILE = ROOT / "waveform.json"
import re
from typing import List, Dict, Optional
import json as _json
try:
    from fugashi import Tagger as _Tagger
    from pykakasi import kakasi as _kakasi
    _tagger = _Tagger()
    _kks = _kakasi(); _kks.setMode("J","H"); _conv = _kks.getConverter()
    _rk = _kakasi(); _rk.setMode("H","a"); _rk.setMode("K","a"); _rk.setMode("J","a"); _romaji_conv = _rk.getConverter()
except Exception:
    _tagger = None; _conv = None; _romaji_conv = None
ASR_DEBUG_FILE = ROOT / "asr_debug.json"

# Import the ChatGPT vocabulary analyzer
try:
    sys.path.insert(0, str(ROOT))
    from gpt_vocab_analyzer import analyze_transcript_vocab, USE_GPT_VOCAB_ANALYSIS
except ImportError:
    def analyze_transcript_vocab(text: str):
        return {"vocabulary": [], "kanji_only": [], "katakana_words": []}
    USE_GPT_VOCAB_ANALYSIS = False

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
        "asr_backend": (os.getenv("ASR_BACKEND", "").lower()),
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

def _to_romaji(s: str) -> str:
    if _romaji_conv is None:
        return s
    try:
        return str(_romaji_conv.do(s)).lower()
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
        r = requests.get(url, timeout=3)
        j = r.json(); d = j.get('data') or []
        if not d: return ''
        senses = d[0].get('senses') or []
        for s in senses:
            ed = s.get('english_definitions') or []
            if ed: return ed[0]
    except Exception:
        return ''
    return ''

def best_gloss(word: str) -> str:
    """Get the best English gloss for a Japanese word"""
    return _jisho_best_gloss(word)

def _internal_suggestions(bases: List[str], max_items=30) -> List[Dict[str,str]]:
    """Optional hook to an internal JP library/service.
    If env SIM_API_URL is set, POST { bases: [...], top_k } and expect
    { items: [ { ja, read, en } ... ] }.
    If a local module jp_internal is available with similar_words(bases, top_k), use it.
    Returns a list of dicts { ja, read, en } or [].
    """
    out: List[Dict[str, str]] = []
    try:
        url = os.getenv('SIM_API_URL')
        if url:
            import requests as _rq
            r = _rq.post(url, json={ 'bases': bases, 'top_k': max_items }, timeout=3)
            r.raise_for_status()
            data = r.json() or {}
            items = data.get('items') or []
            for it in items:
                ja = (it.get('ja') or it.get('jp') or '').strip()
                if not ja: continue
                rd = (it.get('read') or it.get('reading') or it.get('reading_kana') or '').strip()
                en = (it.get('en') or it.get('gloss') or '').strip()
                out.append({ 'ja': ja, 'read': rd, 'en': en })
            if out:
                return out[:max_items]
    except Exception:
        pass
    # Try local module
    try:
        import jp_internal  # type: ignore
        try:
            items = jp_internal.similar_words(bases, max_items)  # expected list of { ja, read?, en? }
        except Exception:
            items = []
        for it in (items or []):
            ja = (it.get('ja') or it.get('jp') or '').strip()
            if not ja: continue
            rd = (it.get('read') or it.get('reading') or it.get('reading_kana') or '').strip()
            en = (it.get('en') or it.get('gloss') or '').strip()
            out.append({ 'ja': ja, 'read': rd, 'en': en })
        return out[:max_items]
    except Exception:
        return []

def _extract_suggestions(text: str, exclude: set, max_items=30) -> List[Dict[str,str]]:
    # Build from current Vocab: entries by semantic proximity (see_also) via Jisho
    out: List[Dict[str,str]] = []
    seen: set = set()
    bases: List[str] = []
    for line in text.splitlines():
        if not line.startswith('Vocab:'): continue
        parts = [p.strip() for p in line[6:].split('・') if p.strip()]
        for p in parts:
            if '(' in p:
                b = p.split('(',1)[0].strip()
                if b: bases.append(b)
    def _similar_words(base: str) -> List[str]:
        try:
            import requests
            url = f"https://jisho.org/api/v1/search/words?keyword={requests.utils.quote(base)}"
            r = requests.get(url, timeout=3)
            j = r.json(); data = j.get('data') or []
            if not data: return []
            sims: List[str] = []
            for sense in (data[0].get('senses') or []):
                for sa in (sense.get('see_also') or []):
                    # keep only Japanese-ish entries (strip explanation)
                    w = str(sa).split(' ')[0].strip()
                    if _has_kana_or_kanji(w): sims.append(w)
            return sims
        except Exception:
            return []
    # Prefer internal provider if configured
    internal = _internal_suggestions(bases, max_items)
    if internal:
        filtered: List[Dict[str,str]] = []
        seen = set()
        for it in internal:
            ja = it.get('ja','').strip()
            if not ja or ja in exclude or ja in seen: continue
            if len(ja) > 14: continue
            if not _has_kana_or_kanji(ja): continue
            en = (it.get('en') or '').strip()
            if not en: continue
            rd = (it.get('read') or '').strip() or _reading_for(ja)
            seen.add(ja)
            filtered.append({ 'ja': ja, 'read': rd, 'en': en })
            if len(filtered) >= max_items: break
        return filtered

    for base in bases:
        if len(out) >= max_items: break
        for w in _similar_words(base):
            if len(out) >= max_items: break
            if w in exclude or w in seen: continue
            seen.add(w)
            rd = _reading_for(w)
            en = _jisho_best_gloss(w)
            # Require a real English gloss; skip if none
            if not en: continue
            out.append({ 'ja': w, 'read': rd, 'en': en })
    return out

def _extract_from_raw(text: str, exclude: set, max_items=30) -> List[Dict[str,str]]:
    """Build suggestions directly from recent raw transcript lines.
    Priority order:
      1) Vocab-like tokens (名詞/動詞/形容詞)
      2) Tokens containing Kanji
      3) Katakana tokens
    For each item, include ja, reading_kana in hira, and en gloss via Jisho if available.
    """
    try:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Consider only the last ~80 content lines
        lines = lines[-120:]
        # Skip meta lines
        lines = [ln for ln in lines if not (ln.startswith('Vocab:') or ln.startswith('📚'))]
        seen: set = set()
        vocab: List[str] = []
        kanji: List[str] = []
        kata: List[str] = []
        for line in lines:
            if _tagger is not None:
                try:
                    toks = list(_tagger(line))
                except Exception:
                    toks = []
                for m in toks:
                    surf = m.surface.strip()
                    if not surf:
                        continue
                    if surf in exclude or surf in seen:
                        continue
                    pos1 = getattr(m.feature, 'pos1', '')
                    if pos1 in {'名詞','動詞','形容詞'}:
                        vocab.append(surf); seen.add(surf)
                        continue
                    if _KANJI.search(surf):
                        kanji.append(surf); seen.add(surf)
                        continue
                    if _KATA.search(surf):
                        kata.append(surf); seen.add(surf)
            else:
                # Regex fallback when tagger is unavailable
                try:
                    for w in re.findall(r"[一-龯々〆ヵヶ]+", line):
                        w = w.strip()
                        if w and (w not in seen) and (w not in exclude):
                            kanji.append(w); seen.add(w)
                    for w in re.findall(r"[ァ-ヴー]+", line):
                        w = w.strip()
                        if w and (w not in seen) and (w not in exclude):
                            kata.append(w); seen.add(w)
                except Exception:
                    pass
        ordered = vocab + kanji + kata
        out: List[Dict[str,str]] = []
        uniq: set = set()
        for w in ordered:
            if w in uniq:
                continue
            uniq.add(w)
            # reading via fugashi first token, fallback to hira
            rd = ''
            try:
                toks = list(_tagger(w)) if _tagger else []
                if toks:
                    k = getattr(toks[0].feature, 'pron', None) or getattr(toks[0].feature, 'kana', None)
                    rd = _to_hira(k if k else w)
                else:
                    rd = _to_hira(w)
            except Exception:
                rd = _to_hira(w)
            en = best_gloss(w) or ''
            out.append({'ja': w, 'read': rd, 'en': en})
            if len(out) >= max_items:
                break
        # If still empty, try one more pass with regex-only from all lines
        if not out:
            try:
                words = []
                for ln in lines:
                    words += re.findall(r"[一-龯々〆ヵヶ]+", ln)
                    words += re.findall(r"[ァ-ヴー]+", ln)
                for w in words:
                    w = w.strip()
                    if not w or w in uniq or w in exclude:
                        continue
                    rd = _to_hira(w)
                    en = best_gloss(w) or ''
                    out.append({'ja': w, 'read': rd, 'en': en})
                    uniq.add(w)
                    if len(out) >= max_items:
                        break
            except Exception:
                pass
        return out
    except Exception:
        return []

@app.get("/api/overlay/suggestions")
def get_suggestions():
    try:
        # Allow disabling suggestions entirely via env
        if os.getenv("SUGGESTIONS_ENABLED", "true").lower() not in {"1","true","yes","y"}:
            return JSONResponse({"items": []})
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
        # First preference: build suggestions from raw transcript directly
        items = _extract_from_raw(text, exclude)
        if not items:
            # Fallback to similarity-based suggestions
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
                })
            return JSONResponse({"items": out})
    except Exception:
        pass
    return JSONResponse({"items": []})


@app.get("/api/vocab/enhanced")
def get_enhanced_vocab():
    """
    Get enhanced vocabulary analysis using ChatGPT.
    Returns detailed vocabulary with readings, meanings, kanji breakdowns, etc.
    """
    try:
        if not USE_GPT_VOCAB_ANALYSIS:
            return JSONResponse({
                "enabled": False,
                "vocabulary": [],
                "kanji_only": [],
                "katakana_words": []
            })
        
        text = _read_text(TRANSCRIPT_FILE)
        if not text.strip():
            return JSONResponse({
                "enabled": True,
                "vocabulary": [],
                "kanji_only": [],
                "katakana_words": []
            })
        
        # Analyze the transcript text (items include original line_index)
        analysis = analyze_transcript_vocab(text)

        return JSONResponse({
            "enabled": True,
            "vocabulary": analysis.get("vocabulary", []),
            "kanji_only": analysis.get("kanji_only", []),
            "katakana_words": analysis.get("katakana_words", [])
        })
        
    except Exception as e:
        print(f"Enhanced vocab analysis error: {e}")
        return JSONResponse({
            "enabled": USE_GPT_VOCAB_ANALYSIS,
            "error": str(e),
            "vocabulary": [],
            "kanji_only": [],
            "katakana_words": []
        })
