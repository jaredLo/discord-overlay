#!/usr/bin/env python3
# listener.py — Accuracy-first live JP ASR with inline annotations
# Strategy: end-of-utterance chunks (silence or max length) → beam search (base/int8) → annotate tokens
# Output example: みんな、小学校(しょうがっこう、elementary school)の時(とき、time)以来(いらい、since)だね。

import time, queue, threading, sys, re, os, functools
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from pykakasi import kakasi
from fugashi import Tagger
from typing import Union
import requests
from deep_translator import DeeplTranslator
import json, pathlib
import sqlite3

# ================= Config (latency-leaning translate) =================
TARGET_DEVICE_SUBSTR = "BlackHole"
SAMPLE_RATE = 16000
FRAME_MS = 20          # WebRTC VAD requires 10/20/30ms
CHANNELS_IN = 2

# Utterance chunking tuned for lower latency
VAD_LEVEL = 2
SILENCE_HANG_MS = 120   # quicker end detection for faster flush
MIN_CHUNK_SEC = 0.40    # smaller chunks for near real-time
MAX_CHUNK_SEC = 1.2     # force periodic flush during long speech
TRANSCRIPT_OUT = "transcript.txt"
MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")   # on M1/8GB this is the safe choice
ASR_BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE", "1"))

# Output mode: "translate" for English, "annotate" for JP+furigana+gloss
OUTPUT_MODE = os.getenv("OUTPUT_MODE", "annotate").lower()

# ================= UniDic tagger (full if available) =================
def make_tagger():
    try:
        import unidic
        return Tagger('-d ' + unidic.DICDIR)  # full UniDic
    except Exception:
        return Tagger()  # fallback (likely unidic-lite)
tagger = make_tagger()
ALLOW_ONLINE_GLOSS = os.getenv("ALLOW_ONLINE_GLOSS", "0") in {"1","true","TRUE","yes","YES"}
try:
    _deepl = DeeplTranslator(source="ja", target="en", api_key=os.getenv("DEEPL_API_KEY")) if ALLOW_ONLINE_GLOSS else None
except Exception:
    _deepl = None

# ================= Keywords (limit annotations to these) =================
_KW_PATH = pathlib.Path("keywords.json")
_kw_cache = {"mtime": None, "map": {}}

def _normalize_key(s: str) -> str:
    return s.strip()

def load_keywords():
    try:
        mtime = _KW_PATH.stat().st_mtime
    except FileNotFoundError:
        _kw_cache["map"] = {}
        _kw_cache["mtime"] = None
        return _kw_cache["map"]
    if _kw_cache["mtime"] == mtime:
        return _kw_cache["map"]
    try:
        data = json.loads(_KW_PATH.read_text(encoding="utf-8"))
        items = data.get("items") or []
        mapping = {}
        for it in items:
            surf = str(it.get("kanji") or "").strip()
            if not surf:
                continue
            key = _normalize_key(surf)
            mapping[key] = {
                "reading": (it.get("reading") or "").strip() or None,
                "en": (it.get("en") or "").strip() or None,
            }
        _kw_cache["map"] = mapping
        _kw_cache["mtime"] = mtime
        return mapping
    except Exception:
        _kw_cache["map"] = {}
        _kw_cache["mtime"] = mtime
        return _kw_cache["map"]

# ================= Offline glossary (fast local glosses) =================
_GLOSS_PATH = pathlib.Path("glossary.json")
_gl_cache = {"mtime": None, "map": {}}

def load_glossary():
    try:
        mtime = _GLOSS_PATH.stat().st_mtime
    except FileNotFoundError:
        _gl_cache["map"] = {}
        _gl_cache["mtime"] = None
        return _gl_cache["map"]
    if _gl_cache["mtime"] == mtime:
        return _gl_cache["map"]
    try:
        data = json.loads(_GLOSS_PATH.read_text(encoding="utf-8"))
        mapping = {}
        for k, v in (data or {}).items():
            if not k:
                continue
            mapping[_normalize_key(k)] = (v or "").strip() or None
        _gl_cache["map"] = mapping
        _gl_cache["mtime"] = mtime
        return mapping
    except Exception:
        _gl_cache["map"] = {}
        return _gl_cache["map"]

"""
Optional large offline dictionary (JMdict index)
- Provide a JSON file where keys are surface/reading strings and values are
  either a string gloss or a list of gloss strings. Path via env `JMDICT_JSON`
  or default `jmdict.min.json` in repo root. Loaded lazily and cached by mtime.
"""
_JMDICT_JSON_PATH = pathlib.Path(os.getenv("JMDICT_JSON", "jmdict.min.json"))
_jmd_cache = {"mtime": None, "map": {}}

def load_jmdict_min():
    try:
        mtime = _JMDICT_JSON_PATH.stat().st_mtime
    except FileNotFoundError:
        _jmd_cache["map"] = {}
        _jmd_cache["mtime"] = None
        return _jmd_cache["map"]
    if _jmd_cache["mtime"] == mtime:
        return _jmd_cache["map"]
    try:
        data = json.loads(_JMDICT_JSON_PATH.read_text(encoding="utf-8"))
        mapping = {}
        for k, v in (data or {}).items():
            if not k:
                continue
            if isinstance(v, list):
                mapping[_normalize_key(k)] = [str(x) for x in v if x]
            elif isinstance(v, str):
                mapping[_normalize_key(k)] = [v]
        _jmd_cache["map"] = mapping
        _jmd_cache["mtime"] = mtime
        return mapping
    except Exception:
        _jmd_cache["map"] = {}
        return _jmd_cache["map"]

# Optional SQLite index (faster cold start vs big JSON)
_JMDICT_DB_PATH = pathlib.Path(os.getenv("JMDICT_DB", "jmdict.sqlite"))
_jmd_db = {"conn": None, "mtime": None}

def _open_jmdict_db():
    try:
        mtime = _JMDICT_DB_PATH.stat().st_mtime
    except FileNotFoundError:
        _jmd_db["conn"] = None
        _jmd_db["mtime"] = None
        return None
    if _jmd_db["conn"] is not None and _jmd_db["mtime"] == mtime:
        return _jmd_db["conn"]
    try:
        conn = sqlite3.connect(str(_JMDICT_DB_PATH))
        conn.row_factory = sqlite3.Row
        _jmd_db["conn"] = conn
        _jmd_db["mtime"] = mtime
        return conn
    except Exception:
        _jmd_db["conn"] = None
        return None

# ================= Device pick =================
def pick_input(substr):
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 and substr.lower() in d['name'].lower():
            return i, d['name']
    raise RuntimeError(f"No input matching '{substr}'. Check Audio MIDI routing and system output device.")

# ================= Audio capture =================
def audio_thread(dev_index, out_q: queue.Queue):
    blocksize = int(SAMPLE_RATE * FRAME_MS / 1000)
    def cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        mono = indata[:, :CHANNELS_IN].mean(axis=1).astype(np.float32)
        out_q.put(mono.copy())
    with sd.InputStream(device=dev_index, channels=CHANNELS_IN, samplerate=SAMPLE_RATE,
                        blocksize=blocksize, dtype='float32', callback=cb):
        while True:
            time.sleep(1)

# ================= Light preprocess =================
def preprocess(audio: np.ndarray, target_dbfs=-22.0):
    a = audio - np.mean(audio)
    if len(a) >= 2:
        a = np.concatenate([[a[0]], a[1:] - 0.97 * a[:-1]])  # pre-emphasis
    rms = np.sqrt(np.mean(a*a)) + 1e-12
    gain = 10 ** ((target_dbfs - 20*np.log10(rms + 1e-12))/20)
    return np.clip(a * gain, -1.0, 1.0).astype(np.float32)

# ================= Utterance chunker (silence or max) =================
def utterance_chunks(frames_iter):
    vad = webrtcvad.Vad(VAD_LEVEL)
    frame_len = int(SAMPLE_RATE * FRAME_MS / 1000)
    hang_frames = int(SILENCE_HANG_MS / FRAME_MS)
    max_frames = int(MAX_CHUNK_SEC * 1000 / FRAME_MS)
    min_frames = int(MIN_CHUNK_SEC * 1000 / FRAME_MS)

    buf = np.zeros(0, dtype=np.float32)
    seg = []
    speaking = False
    hang = 0

    while True:
        f_in = next(frames_iter)
        buf = np.concatenate([buf, f_in])

        while len(buf) >= frame_len:
            f = buf[:frame_len]; buf = buf[frame_len:]
            b = (np.clip(f, -1, 1) * 32767).astype(np.int16).tobytes()
            is_sp = vad.is_speech(b, SAMPLE_RATE)

            if is_sp:
                speaking = True
                hang = hang_frames
                seg.append(f)
                # force flush at max length
                if len(seg) >= max_frames:
                    audio = np.concatenate(seg, axis=0); seg = []
                    if len(audio) >= min_frames * frame_len:
                        yield audio
            else:
                if speaking:
                    hang -= 1
                    if hang <= 0:
                        speaking = False
                        if seg:
                            audio = np.concatenate(seg, axis=0); seg = []
                            if len(audio) >= min_frames * frame_len:
                                yield audio

# ================= ASR with quality gate =================
def good_seg(s):
    if getattr(s, "no_speech_prob", 0) > 0.6: return False
    if getattr(s, "avg_logprob", 0) < -1.0:   return False
    if getattr(s, "compression_ratio", 0) > 2.4: return False
    return True

def transcribe_iter(chunks_iter, model, task: str = "transcribe"):
    for audio in chunks_iter:
        audio = preprocess(audio)
        segs, _ = model.transcribe(
            audio,
            language="ja",
            task=("translate" if task == "translate" else "transcribe"),
            beam_size=ASR_BEAM_SIZE,     # fastest default
            temperature=0.0,             # deterministic
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True
        )
        txt = "".join(s.text for s in segs if good_seg(s)).strip()
        if txt:
            yield txt

# ================= Annotation engine =================
_kks = kakasi(); _kks.setMode("J","H")
_conv = _kks.getConverter()

KANJI_RE = re.compile(r"[一-龯々〆ヵヶ]")
KATA_LETTER_RE = re.compile(r"[ァ-ヴ]")  # excludes 'ー'
NUM_RE = re.compile(r"\d+")
KANJI_NUM_RE = re.compile(r"[一二三四五六七八九十百千]+")
PUNCT_POS = {"記号"}
SKIP_POS = {"助詞","助動詞"}  # keep interjections; skip particles/aux
SKIP_COUNTER_ANNOTATION = True  # skip translating counters like 三人, 3枚

def to_hira(s: str) -> str:
    return "".join(p["hira"] for p in _conv.convert(s))

def get_reading(m) -> Union[str, None]:
    return getattr(m.feature, "pron", None) or getattr(m.feature, "kana", None)

def has_kata_letter(s: str) -> bool:
    return bool(KATA_LETTER_RE.search(s))

# Counter readings 1..10 (common set)
COUNTER_READ = {
    "人": ["ひとり","ふたり","さんにん","よにん","ごにん","ろくにん","ななにん","はちにん","きゅうにん","じゅうにん"],
    "本": ["いっぽん","にほん","さんぼん","よんほん","ごほん","ろっぽん","ななほん","はっぽん","きゅうほん","じゅっぽん"],
    "匹": ["いっぴき","にひき","さんびき","よんひき","ごひき","ろっぴき","ななひき","はっぴき","きゅうひき","じゅっぴき"],
    "回": ["いっかい","にかい","さんかい","よんかい","ごかい","ろっかい","ななかい","はちかい","きゅうかい","じゅっかい"],
    "分": ["いっぷん","にふん","さんぷん","よんぷん","ごふん","ろっぷん","ななふん","はっぷん","きゅうふん","じゅっぷん"],
    "歳": ["いっさい","にさい","さんさい","よんさい","ごさい","ろくさい","ななさい","はっさい","きゅうさい","じゅっさい"],
    "冊": ["いっさつ","にさつ","さんさつ","よんさつ","ごさつ","ろくさつ","ななさつ","はっさつ","きゅうさつ","じゅっさつ"],
    "杯": ["いっぱい","にはい","さんばい","よんはい","ごはい","ろっぱい","ななはい","はっぱい","きゅうはい","じゅっぱい"],
    "台": ["いちだい","にだい","さんだい","よんだい","ごだい","ろくだい","ななだい","はちだい","きゅうだい","じゅうだい"],
    "個": ["いっこ","にこ","さんこ","よんこ","ごこ","ろっこ","ななこ","はっこ","きゅうこ","じゅっこ"],
    "円": ["いちえん","にえん","さんえん","よえん","ごえん","ろくえん","ななえん","はちえん","きゅうえん","じゅうえん"],
    "枚": ["いちまい","にまい","さんまい","よんまい","ごまい","ろくまい","ななまい","はちまい","きゅうまい","じゅうまい"],
}
KANJI_NUM_MAP = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}

def to_int_number(s: str) -> Union[int, None]:
    m = NUM_RE.fullmatch(s)
    if m:
        try: return int(m.group(0))
        except: return None
    if KANJI_NUM_RE.fullmatch(s):
        val = 0
        for ch in s:
            if ch == "十": val = (val or 1)*10
            elif ch in KANJI_NUM_MAP: val += KANJI_NUM_MAP[ch]
            else: return None
        return val or None
    return None

@functools.lru_cache(maxsize=1024)
def jisho_lookup(word: str):
    if not ALLOW_ONLINE_GLOSS:
        return None
    try:
        r = requests.get(
            "https://jisho.org/api/v1/search/words",
            params={"keyword": word},
            timeout=0.3,
        )
        data = r.json()
        if data.get("data"):
            return data["data"][0]
    except Exception:
        return None
    return None

@functools.lru_cache(maxsize=4096)
def best_gloss(word: str, prefer_counter=False) -> Union[str, None]:
    w = _normalize_key(word)
    # 1) keywords override (fast)
    kw = load_keywords().get(w)
    if kw and kw.get("en"):
        return kw["en"]
    # 2) offline glossary (fast)
    gl = load_glossary().get(w)
    if gl:
        return gl
    # 3) JMdict SQLite (if present)
    conn = _open_jmdict_db()
    if conn is not None:
        try:
            cur = conn.execute("SELECT gloss FROM entries WHERE term = ? LIMIT 1", (w,))
            row = cur.fetchone()
            if row is not None:
                try:
                    arr = json.loads(row["gloss"]) if row["gloss"] else []
                    if isinstance(arr, list) and arr:
                        return arr[0]
                except Exception:
                    if row["gloss"]:
                        return row["gloss"]
        except Exception:
            pass
    # 4) JMdict JSON index (if present)
    jmd = load_jmdict_min().get(w)
    if jmd:
        return jmd[0] if isinstance(jmd, list) and jmd else jmd
    # 5) optional online (short timeouts)
    if _deepl:
        try:
            return _deepl.translate(word)
        except Exception:
            pass
    entry = jisho_lookup(word)
    if not entry:
        return None
    cands = []
    for sense in entry.get("senses", []):
        pos_tags = set(sense.get("parts_of_speech") or [])
        for gloss in sense.get("english_definitions") or []:
            score = 0
            if "Counter" in pos_tags:
                score += 1
            if prefer_counter and "Counter" in pos_tags:
                score += 2
            cands.append((score, gloss))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]

def dict_has_entry(s: str) -> bool:
    s = _normalize_key(s)
    if s in load_glossary() or s in load_keywords():
        return True
    # JMdict SQLite
    conn = _open_jmdict_db()
    if conn is not None:
        try:
            cur = conn.execute("SELECT 1 FROM entries WHERE term = ? LIMIT 1", (s,))
            if cur.fetchone() is not None:
                return True
        except Exception:
            pass
    # JMdict JSON
    if s in load_jmdict_min():
        return True
    # Optional online
    return jisho_lookup(s) is not None

def compound_reading(tokens) -> str:
    kana = []
    for m in tokens:
        k = get_reading(m)
        kana.append(k if k else m.surface)
    return to_hira("".join(kana))

def _is_number_counter(surf: str) -> bool:
    # Detect forms like 3人, 三人, 10枚 etc.
    if len(surf) >= 2 and surf[-1] in COUNTER_READ:
        n = to_int_number(surf[:-1])
        if n is not None:
            return True
    return False

def maybe_merge_compound(tokens, i, max_len=4):
    # Allow compounds including prefix/suffix parts commonly used in words
    # e.g., お + 名詞 + さん (お姉さん), お + 疲れ + 様 (お疲れ様)
    ALLOW = {"名詞", "接頭詞", "接尾辞"}
    for L in range(min(max_len, len(tokens)-i), 1, -1):
        seg = tokens[i:i+L]
        pos_ok = all(getattr(t.feature, "pos1", "") in ALLOW for t in seg)
        if not pos_ok:
            continue
        surf = "".join(t.surface for t in seg)
        # Only consider if likely a lexicalized phrase
        if (KANJI_RE.search(surf) or has_kata_letter(surf)) and dict_has_entry(surf):
            if SKIP_COUNTER_ANNOTATION and _is_number_counter(surf):
                # Treat as plain text when it's number+counter
                return None
            reading = compound_reading(seg)
            gloss = best_gloss(surf) or ""
            anno = f"{surf}({to_hira(reading)}" + (f"、{gloss})" if gloss else ")")
            return anno, L
    return None

def find_keyword_match(tokens, i, max_len=3):
    kws = load_keywords()
    # longest match first
    for L in range(min(max_len, len(tokens)-i), 0, -1):
        surf = "".join(t.surface for t in tokens[i:i+L])
        key = _normalize_key(surf)
        if key in kws:
            meta = kws[key]
            # reading fallback using tokenizer/kakasi
            reading = meta.get("reading") or compound_reading(tokens[i:i+L])
            gloss = meta.get("en") or None
            if not gloss:
                # final fallback if provided file has no gloss
                gloss = best_gloss(surf) or None
            anno = f"{surf}({to_hira(reading)}" + (f"、{gloss})" if gloss else ")")
            return anno, L
    return None

def annotate_text(text: str) -> str:
    toks = list(tagger(text))
    out = []; i = 0
    while i < len(toks):
        # try to match configured keywords (up to 3 tokens)
        matched = find_keyword_match(toks, i, max_len=3)
        if matched:
            anno, L = matched
            out.append(anno)
            i += L
            continue
        m = toks[i]
        surf = m.surface
        pos1 = getattr(m.feature, "pos1", "")

        # Compound merge if recognized word
        merged = maybe_merge_compound(toks, i)
        if merged:
            anno, L = merged
            out.append(anno)
            i += L
            continue

        # Skip punctuation/particles/aux as-is
        if pos1 in PUNCT_POS or pos1 in SKIP_POS:
            out.append(surf)
            i += 1
            continue

        # Counter patterns: [number][counter] or prev number + counter -> skip annotation
        if SKIP_COUNTER_ANNOTATION:
            if len(surf) >= 2 and surf[-1] in COUNTER_READ:
                n = to_int_number(surf[:-1])
                if n and 1 <= n <= 10:
                    out.append(surf)
                    i += 1
                    continue
            if surf in COUNTER_READ and i > 0:
                n = to_int_number(toks[i-1].surface)
                if n and 1 <= n <= 10:
                    out.append(surf)
                    i += 1
                    continue

        # Annotate only if token contains kanji or katakana letters
        # Pure hiragana words are left unannotated for readability
        annotatable = bool(KANJI_RE.search(surf) or has_kata_letter(surf))
        if not annotatable:
            out.append(surf)
            i += 1
            continue

        # Reading from tagger or kakasi
        k = get_reading(m)
        reading = to_hira(k) if k else to_hira(surf)

        # Prefer lemma for dictionary lookup
        lemma = getattr(m.feature, "lemma", None) or surf
        gloss = best_gloss(lemma) or best_gloss(surf)
        out.append(f"{surf}({reading}" + (f"、{gloss})" if gloss else ")"))
        i += 1
    return "".join(out)
 
 
def append_transcript(text: str):
     with open(TRANSCRIPT_OUT, "a", encoding="utf-8") as f:
         f.write(text + "\n")
# ================= Main =================
def main():
    dev_idx, dev_name = pick_input(TARGET_DEVICE_SUBSTR)
    print(f"Listening on: {dev_name}")
    open(TRANSCRIPT_OUT, "w", encoding="utf-8").close()
    q = queue.Queue(maxsize=128)
    threading.Thread(target=audio_thread, args=(dev_idx, q), daemon=True).start()

    def frames():
        while True:
            yield q.get()

    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)

    # generator of 20ms frames -> utterance audio arrays
    def chunks():
        frame_len = int(SAMPLE_RATE * FRAME_MS / 1000)
        while True:
            f = q.get()
            yield f
    task = "translate" if OUTPUT_MODE == "translate" else "transcribe"
    for text in transcribe_iter(utterance_chunks(chunks()), model, task=task):
        if OUTPUT_MODE == "translate":
            # Already translated to English by Whisper
            append_transcript(text)
            print(text)
        else:
            annotated = annotate_text(text)
            append_transcript(annotated)
            print(annotated)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
