#!/usr/bin/env python3
# listener.py — YouTube → BlackHole → hybrid VAD/time chunking → Whisper → keywords.json
# Emits keywords as: JP（reading, EN）

import json, time, queue, threading, sys, re
from collections import deque
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from fugashi import Tagger
from pykakasi import kakasi
from jamdict import Jamdict

# =====================
# Config
# =====================
TARGET_DEVICE_SUBSTR = "BlackHole"   # input device name fragment to capture system audio
SAMPLE_RATE = 16000                  # VAD-friendly sample rate
FRAME_MS = 20                        # 10/20/30 only for WebRTC VAD
CHANNELS_IN = 2                      # BlackHole 2ch; downmix to mono
MODEL_SIZE = "small"                 # tiny|base|small|medium|large-v3
COMPUTE_TYPE = "int8"                # "int8" is fast on CPU; try "float16" on fast M-series

# Chunking behavior (prevents long stalls during nonstop speech)
VAD_AGGRESSIVENESS = 2               # 0..3 (higher = more strict)
HANG_MS = 350                        # wait after last speech frame before closing chunk
MAX_CHUNK_SEC = 2.0                  # force flush every ~3s even if speaker never pauses
MIN_CHUNK_SEC = 0.8                  # ignore tiny fragments
OVERLAP_SEC = 0.25                   # context overlap between consecutive chunks

# Rolling window for stabler keywords
WINDOW_SEC = 8.0                    # lookback window for keyword extraction

# Keywords output
MAX_WORDS = 8
KEYWORDS_OUT = "keywords.json"

# Optional stop words to keep the list useful
JA_STOP = {"こと","とき","ところ","ため","もの","これ","それ","あれ"}

# =====================
# Device pick
# =====================
def pick_input(substr):
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 and substr.lower() in d['name'].lower():
            return i, d['name']
    raise RuntimeError(f"No input matching '{substr}'. Check Audio MIDI routing and system output device.")

# =====================
# Audio capture thread
# =====================
def audio_thread(dev_index, out_q: queue.Queue):
    blocksize = int(SAMPLE_RATE * FRAME_MS/1000)
    def cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # downmix stereo → mono float32 [-1,1]
        mono = indata[:, :CHANNELS_IN].mean(axis=1).astype(np.float32)
        out_q.put(mono.copy())
    with sd.InputStream(device=dev_index, channels=CHANNELS_IN, samplerate=SAMPLE_RATE,
                        blocksize=blocksize, dtype='float32', callback=cb):
        while True:
            time.sleep(1)

# =====================
# Hybrid chunker: VAD + time-based flush
# =====================
def vad_or_time_chunks(frames_iter,
                       sample_rate=SAMPLE_RATE,
                       frame_ms=FRAME_MS,
                       vad_level=VAD_AGGRESSIVENESS,
                       hang_ms=HANG_MS,
                       max_chunk_sec=MAX_CHUNK_SEC,
                       min_chunk_sec=MIN_CHUNK_SEC,
                       overlap_sec=OVERLAP_SEC):
    vad = webrtcvad.Vad(vad_level)
    frame_len = int(sample_rate * frame_ms / 1000)
    max_frames = int(max_chunk_sec * 1000 / frame_ms)
    min_frames = int(min_chunk_sec * 1000 / frame_ms)
    overlap_frames = int(overlap_sec * 1000 / frame_ms)
    hang_frames = int(hang_ms / frame_ms)

    buf = np.zeros(0, dtype=np.float32)
    chunk_frames = []        # list of float32 frames
    speaking = False
    hang = 0

    while True:
        f_in = next(frames_iter)
        buf = np.concatenate([buf, f_in])

        while len(buf) >= frame_len:
            f = buf[:frame_len]; buf = buf[frame_len:]
            b = (np.clip(f, -1, 1) * 32767).astype(np.int16).tobytes()
            is_sp = vad.is_speech(b, sample_rate)

            if is_sp:
                speaking = True
                hang = hang_frames
                chunk_frames.append(f)

                # Time-based flush during continuous speech
                if len(chunk_frames) >= max_frames:
                    audio = np.concatenate(chunk_frames, axis=0)
                    yield audio
                    # keep a small tail for context
                    tail = chunk_frames[-overlap_frames:] if overlap_frames < len(chunk_frames) else chunk_frames
                    chunk_frames = list(tail)
            else:
                if speaking:
                    hang -= 1
                    if hang <= 0:
                        speaking = False
                        if len(chunk_frames) >= min_frames:
                            yield np.concatenate(chunk_frames, axis=0)
                        chunk_frames = []

# =====================
# Transcription
# =====================
def transcribe_iter(chunks_iter, model):
    for audio in chunks_iter:
        if len(audio) < SAMPLE_RATE * 0.25:  # ignore blips
            continue
        segs, _ = model.transcribe(audio, language="ja", beam_size=1, vad_filter=False)
        text = "".join(s.text for s in segs).strip()
        if text:
            yield text

# =====================
# Rolling window of recent text
# =====================
class RollingText:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.buf: deque[tuple[float,str]] = deque()
    def add(self, text: str):
        now = time.time()
        self.buf.append((now, text))
        cutoff = now - self.seconds
        while self.buf and self.buf[0][0] < cutoff:
            self.buf.popleft()
    def text(self) -> str:
        return "。".join(t for _, t in self.buf)

# =====================
# Keywords (noun-first, with fallbacks) + furigana + English gloss
# =====================
tagger = Tagger()

_kks = kakasi()
_kks.setMode("J", "H")              # Kanji → Hiragana
_conv = _kks.getConverter()

def to_hira(s: str) -> str:
    # pykakasi v2+: use convert(); join hira parts
    return "".join(p["hira"] for p in _conv.convert(s))

_jam = Jamdict()
_GLOSS_CACHE: dict[str, str | None] = {}

def ja_gloss(word: str) -> str | None:
    g = _GLOSS_CACHE.get(word)
    if g is not None:
        return g
    try:
        res = _jam.lookup(word)
        for e in res.entries:
            for s in e.senses:
                # prefer English or unlabeled glosses
                gls = []
                for gl in getattr(s, "gloss", []):
                    txt = getattr(gl, "text", None) or str(gl)
                    lang = getattr(gl, "lang", None)
                    gls.append((lang, txt))
                for lang, txt in gls:
                    if (lang is None) or str(lang).lower().startswith(("en","eng")):
                        _GLOSS_CACHE[word] = txt
                        return txt
                if gls:
                    _GLOSS_CACHE[word] = gls[0][1]
                    return gls[0][1]
        _GLOSS_CACHE[word] = None
        return None
    except Exception:
        _GLOSS_CACHE[word] = None
        return None

_num_re = re.compile(r"\d+(?:[.,]\d+)?")
_katakana_re = re.compile(r"[ァ-ヴー]{2,}")

def extract_keywords_ja(text: str, topn=MAX_WORDS):
    # 1) collect nouns (allow 1-char) by lemma; 2) fallback verbs/adjectives; 3) pad with numbers/katakana
    freq: dict[str,int] = {}
    tokens = list(tagger(text))

    def add_word(w: str):
        if len(w) >= 1 and w not in JA_STOP:
            freq[w] = freq.get(w, 0) + 1

    for m in tokens:
        pos1 = getattr(m.feature, "pos1", "")
        base = getattr(m.feature, "lemma", None) or m.surface
        if pos1 == "名詞":
            add_word(base)

    ranked = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    chosen = [w for w,_ in ranked]

    if len(chosen) < topn:
        for m in tokens:
            pos1 = getattr(m.feature, "pos1", "")
            base = getattr(m.feature, "lemma", None) or m.surface
            if pos1 in ("動詞","形容詞") and base not in JA_STOP and base != "する":
                if base not in chosen:
                    chosen.append(base)
                    if len(chosen) >= topn:
                        break

    if len(chosen) < topn:
        for n in _num_re.findall(text):
            if n not in chosen:
                chosen.append(n)
                if len(chosen) >= topn:
                    break

    if len(chosen) < topn:
        for k in _katakana_re.findall(text):
            if k not in chosen:
                chosen.append(k)
                if len(chosen) >= topn:
                    break

    # build keyword objects
    items = []
    for w in chosen[:topn]:
        if _num_re.fullmatch(w):
            items.append({"kanji": w, "reading": w, "en": None})
        else:
            items.append({"kanji": w, "reading": to_hira(w), "en": ja_gloss(w)})
    return items

# =====================
# Output writer
# =====================
def write_keywords(items):
    with open(KEYWORDS_OUT, "w", encoding="utf-8") as f:
        json.dump({"updated": time.time(), "items": items}, f, ensure_ascii=False)

# =====================
# Main
# =====================
def main():
    dev_idx, dev_name = pick_input(TARGET_DEVICE_SUBSTR)
    print(f"Listening on: {dev_name}")
    q = queue.Queue(maxsize=128)
    threading.Thread(target=audio_thread, args=(dev_idx, q), daemon=True).start()

    def frames():
        while True:
            yield q.get()

    print("Loading Whisper (first run downloads model)…")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)

    window = RollingText(WINDOW_SEC)

    for text in transcribe_iter(vad_or_time_chunks(frames()), model):
        window.add(text)
        kw_items = extract_keywords_ja(window.text(), topn=MAX_WORDS)
        write_keywords(kw_items)
        pretty = " / ".join(
            f"{k['kanji']}（{k['reading']}" + (f", {k['en']}" if k.get('en') else "") + "）"
            for k in kw_items
        )
        # print("ASR:", text)
        print("KW :", pretty)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass