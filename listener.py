#!/usr/bin/env python3
# listener.py — YouTube → BlackHole → hybrid VAD/time chunking → Whisper → keywords.json (kanji + furigana)

import json, time, queue, threading, sys
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from fugashi import Tagger
from pykakasi import kakasi

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
HANG_MS = 350                        # how long to wait after last speech frame before closing chunk
MAX_CHUNK_SEC = 3.0                  # force flush every ~3s even if speaker never pauses
MIN_CHUNK_SEC = 0.8                  # ignore tiny fragments
OVERLAP_SEC = 0.25                   # context overlap between consecutive chunks

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
# Keywords (noun-only) + furigana (Kakasi convert API)
# =====================
tagger = Tagger()

_kks = kakasi()
_kks.setMode("J", "H")              # Kanji → Hiragana
_conv = _kks.getConverter()

def to_hira(s: str) -> str:
    # pykakasi v2+: use convert(); join hira parts
    return "".join(p["hira"] for p in _conv.convert(s))

def extract_keywords_ja(text, topn=MAX_WORDS):
    freq = {}
    for m in tagger(text):
        if getattr(m.feature, "pos1", "") == "名詞":
            w = m.surface
            if len(w) >= 2 and w not in JA_STOP:
                freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    return [{"kanji": w, "reading": to_hira(w)} for w, _ in ranked[:topn]]

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

    for text in transcribe_iter(vad_or_time_chunks(frames()), model):
        kws = extract_keywords_ja(text)
        write_keywords(kws)
        print("ASR:", text)
        print("KW :", " / ".join(k['kanji'] for k in kws))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass