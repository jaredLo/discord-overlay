#!/usr/bin/env python3
# Accuracy-first JP ASR with inline annotations + DeepL per-utterance + Jisho caching
# Output:
#   みんな、小学校(しょうがっこう、elementary school)の時(とき、time)以来(いらい、since)だね。
#   ⇒ It's been since elementary school days.

import time, queue, threading, sys, re, os, functools, hashlib, json, sqlite3
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from pykakasi import kakasi
from fugashi import Tagger
from typing import Union, Optional, Dict, Tuple, List
import requests
from io import BytesIO
import wave
import html
import requests

# =============== .env loader (lightweight) ===============
def _load_env_file(path: str = ".env"):
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
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

_load_env_file()

# Reuse a single HTTP session for keep-alive
_HTTP = requests.Session()

# =============== Config (accuracy-first, M1/8GB) ===============
TARGET_DEVICE_SUBSTR = os.getenv("TARGET_DEVICE", "BlackHole")
SAMPLE_RATE = 16000
FRAME_MS = 20
CHANNELS_IN = 2

# Utterance chunking: accuracy > latency
VAD_LEVEL = int(os.getenv("VAD_LEVEL", "2"))
SILENCE_HANG_MS = int(os.getenv("SILENCE_HANG_MS", "320"))
# Defaults can be overridden by env
MIN_CHUNK_SEC = float(os.getenv("MIN_CHUNK_SEC", "1.0"))
MAX_CHUNK_SEC = float(os.getenv("MAX_CHUNK_SEC", "3.2"))

# Whisper
MODEL_SIZE = os.getenv("FASTER_WHISPER_SIZE", "base")
COMPUTE_TYPE = os.getenv("FASTER_WHISPER_COMPUTE", "int8")
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))          # tuned for M1
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "1"))

# Backend selection
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "faster-whisper")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = (WHISPER_MODEL.lower() == "whisper-1" and bool(OPENAI_API_KEY))
REMOTE_ASR_URL = os.getenv("REMOTE_ASR_URL", "http://192.168.1.12:8585/api/transcribe")
# Backend selector: remote | openai | local
ASR_BACKEND = os.getenv(
    "ASR_BACKEND",
    ("remote" if REMOTE_ASR_URL else ("openai" if USE_OPENAI else "local"))
).lower()

# Capture-all mode
CAPTURE_ALL = os.getenv("CAPTURE_ALL", "false").lower() in {"1","true","yes","y"}

# GPT formatting (annotation + translation)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano-2025-04-14")
USE_GPT_FORMATTER = (os.getenv("USE_GPT_FORMATTER", "true").lower() in {"1","true","yes","y"}) and bool(OPENAI_API_KEY)
GPT_COMPRESS_ELONG = os.getenv("GPT_COMPRESS_ELONGATIONS", "false").lower() in {"1","true","yes","y"}
GPT_MAX_VOCAB = int(os.getenv("GPT_MAX_VOCAB", "10"))
GPT_MAX_CHARS = int(os.getenv("GPT_MAX_CHARS", "280"))
GPT_RATE_LIMIT_MS = int(os.getenv("GPT_RATE_LIMIT_MS", "600"))
GPT_CLEAN_REPEATS = os.getenv("GPT_CLEAN_REPEATS", "true").lower() in {"1","true","yes","y"}
GPT_SKIP_ON_REPEAT = os.getenv("GPT_SKIP_ON_REPEAT", "true").lower() in {"1","true","yes","y"}
LINE4_TABLE_MODE = os.getenv("LINE4_TABLE_MODE", "false").lower() in {"1","true","yes","y"}
JP_VERBATIM = os.getenv("JP_VERBATIM", "true").lower() in {"1","true","yes","y"}
CHUNK_OVERLAP_MS = int(os.getenv("CHUNK_OVERLAP_MS", "200"))
DISABLE_OVERLAP = os.getenv("DISABLE_OVERLAP", "true").lower() in {"1","true","yes","y"}
DISABLE_CARRY = os.getenv("DISABLE_CARRY", "true").lower() in {"1","true","yes","y"}
ASR_WORKERS = int(os.getenv("ASR_WORKERS", "1"))
REMOTE_ASR_TIMEOUT_MS = int(os.getenv("REMOTE_ASR_TIMEOUT_MS", "60000"))
FORMAT_WORKERS = int(os.getenv("FORMAT_WORKERS", "4"))

# Output line controls
SHOW_VOCAB_LINE = os.getenv("SHOW_VOCAB_LINE", "false").lower() in {"1","true","yes","y"}
SHOW_DETAILS_LINE = os.getenv("SHOW_DETAILS_LINE", "false").lower() in {"1","true","yes","y"}
SHOW_ASR_DEBUG = os.getenv("SHOW_ASR_DEBUG", "true").lower() in {"1","true","yes","y"}

# Output
TRANSCRIPT_OUT = "transcript.txt"
WAVEFORM_OUT = "waveform.json"   # rolling amplitude data for overlay
WAVEFORM_LEN = 1000               # points kept in file
SHOW_SENTENCE_EN = True           # append DeepL line per utterance if API key present
MIN_CHARS_FOR_DEEPL = 6           # skip super-short blips
USER_DICT_CSV = "user_dict.csv"   # optional overrides: surface,reading,english

# Networking politeness
JISHO_MIN_INTERVAL_S = 0.6        # >= 0.6s between remote requests

# =============== UniDic tagger (full if installed) ===============
def make_tagger():
    try:
        import unidic
        return Tagger('-d ' + unidic.DICDIR)  # full UniDic
    except Exception:
        return Tagger()  # fallback (likely unidic-lite)
tagger = make_tagger()

# =============== Device pick ===============
def pick_input(substr):
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 and substr.lower() in d['name'].lower():
            return i, d['name']
    raise RuntimeError(f"No input matching '{substr}'. Check Audio MIDI routing and system output device.")

# =============== Audio capture ===============
def audio_thread(dev_index, out_q: queue.Queue, wave_q: queue.Queue):
    blocksize = int(SAMPLE_RATE * FRAME_MS / 1000)

    def cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        mono = indata[:, :CHANNELS_IN].mean(axis=1).astype(np.float32)
        # Non-blocking put to avoid starving audio callback if downstream is slow
        try:
            out_q.put_nowait(mono.copy())
        except queue.Full:
            try:
                _ = out_q.get_nowait()  # drop oldest frame
                out_q.put_nowait(mono.copy())
            except Exception:
                pass
        # Downsample block into ~50 points per callback and push to non-blocking queue
        try:
            step = max(1, len(mono) // 50)
            sl = mono[::step]
            sl = np.clip(sl, -1.0, 1.0)
            vals = (sl * 100.0).astype(np.int16).tolist()  # -100..100
            wave_q.put_nowait(vals)
        except Exception:
            pass
    with sd.InputStream(device=dev_index, channels=CHANNELS_IN, samplerate=SAMPLE_RATE,
                        blocksize=blocksize, dtype='float32', callback=cb):
        while True:
            time.sleep(1)

def _wave_flush(buf: List[int]):
    try:
        tmp = WAVEFORM_OUT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"data": buf[-WAVEFORM_LEN:]}, f, ensure_ascii=False)
        os.replace(tmp, WAVEFORM_OUT)
    except Exception:
        pass

def chunker_thread(frames_q: queue.Queue, out_chunk_q: queue.Queue, wave_q: queue.Queue):
    # Use VAD-based logic to segment utterances; accuracy over latency
    vad = webrtcvad.Vad(VAD_LEVEL)
    frame_len = int(SAMPLE_RATE * FRAME_MS / 1000)
    hang_frames = int(SILENCE_HANG_MS / FRAME_MS)
    max_frames = int(MAX_CHUNK_SEC * 1000 / FRAME_MS)
    min_frames = int(MIN_CHUNK_SEC * 1000 / FRAME_MS)
    overlap_frames = 0 if DISABLE_OVERLAP else max(0, int(CHUNK_OVERLAP_MS / FRAME_MS))

    buf = np.zeros(0, dtype=np.float32)
    seg: List[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float32)
    speaking = False
    hang = 0
    silence_run = 0
    min_flush_frames = int(0.35 * 1000 / FRAME_MS)
    overlap_prepend_frames: List[np.ndarray] = []

    # Waveform aggregation and timed flush
    wave_buf: List[int] = []
    last_wave_write = time.time()

    while True:
        # Drain some wave samples without blocking
        try:
            while True:
                vals = wave_q.get_nowait()
                wave_buf.extend(vals)
                if len(wave_buf) > WAVEFORM_LEN:
                    wave_buf = wave_buf[-WAVEFORM_LEN:]
        except queue.Empty:
            pass
        now = time.time()
        if now - last_wave_write >= 0.1:
            last_wave_write = now
            _wave_flush(wave_buf)

        f_in = frames_q.get()
        buf = np.concatenate([buf, f_in])

        while len(buf) >= frame_len:
            f = buf[:frame_len]; buf = buf[frame_len:]
            b = (np.clip(f, -1, 1) * 32767).astype(np.int16).tobytes()
            is_sp = vad.is_speech(b, SAMPLE_RATE)
            if is_sp:
                speaking = True
                hang = hang_frames
                silence_run = 0
                if not DISABLE_OVERLAP and (not seg and overlap_prepend_frames):
                    seg.extend(overlap_prepend_frames)
                    overlap_prepend_frames = []
                seg.append(f)
                if len(seg) >= max_frames:
                    audio = np.concatenate(seg, axis=0); seg = []
                    if not DISABLE_OVERLAP and overlap_frames > 0:
                        k = min(overlap_frames, len(audio) // frame_len)
                        if k > 0:
                            tail = audio[-k*frame_len:]
                            overlap_prepend_frames = [tail[i*frame_len:(i+1)*frame_len] for i in range(k)]
                    if not DISABLE_CARRY and carry.size:
                        audio = np.concatenate([carry, audio]); carry = np.zeros(0, dtype=np.float32)
                    if len(audio) >= min_frames * frame_len:
                        # Non-blocking put to prevent backpressure stalls
                        try:
                            out_chunk_q.put_nowait(audio)
                        except queue.Full:
                            try:
                                _ = out_chunk_q.get_nowait()  # drop oldest chunk
                                out_chunk_q.put_nowait(audio)
                            except Exception:
                                pass
            else:
                if speaking:
                    hang -= 1
                    if hang <= 0:
                        speaking = False
                        if seg:
                            audio = np.concatenate(seg, axis=0); seg = []
                            if not DISABLE_OVERLAP and overlap_frames > 0:
                                k = min(overlap_frames, len(audio) // frame_len)
                                if k > 0:
                                    tail = audio[-k*frame_len:]
                                    overlap_prepend_frames = [tail[i*frame_len:(i+1)*frame_len] for i in range(k)]
                            if len(audio) < min_frames * frame_len:
                                if not DISABLE_CARRY:
                                    carry = np.concatenate([carry, audio]) if carry.size else audio
                            else:
                                if not DISABLE_CARRY and carry.size:
                                    audio = np.concatenate([carry, audio]); carry = np.zeros(0, dtype=np.float32)
                                try:
                                    out_chunk_q.put_nowait(audio)
                                except queue.Full:
                                    try:
                                        _ = out_chunk_q.get_nowait()
                                        out_chunk_q.put_nowait(audio)
                                    except Exception:
                                        pass
                else:
                    silence_run += 1
                    if not DISABLE_CARRY and carry.size and silence_run >= 3 * hang_frames:
                        if len(carry) >= min_flush_frames * frame_len:
                            try:
                                out_chunk_q.put_nowait(carry)
                            except queue.Full:
                                try:
                                    _ = out_chunk_q.get_nowait()
                                    out_chunk_q.put_nowait(carry)
                                except Exception:
                                    pass
                        carry = np.zeros(0, dtype=np.float32)

# =============== Light preprocess ===============
def preprocess(audio: np.ndarray, target_dbfs=-22.0):
    a = audio - np.mean(audio)
    if len(a) >= 2:
        a = np.concatenate([[a[0]], a[1:] - 0.97 * a[:-1]])  # pre-emphasis
    rms = np.sqrt(np.mean(a*a)) + 1e-12
    gain = 10 ** ((target_dbfs - 20*np.log10(rms + 1e-12))/20)
    return np.clip(a * gain, -1.0, 1.0).astype(np.float32)

# =============== Utterance chunker (silence or max) ===============
def utterance_chunks(frames_iter):
    vad = webrtcvad.Vad(VAD_LEVEL)
    frame_len = int(SAMPLE_RATE * FRAME_MS / 1000)
    hang_frames = int(SILENCE_HANG_MS / FRAME_MS)
    max_frames = int(MAX_CHUNK_SEC * 1000 / FRAME_MS)
    min_frames = int(MIN_CHUNK_SEC * 1000 / FRAME_MS)
    overlap_frames = 0 if DISABLE_OVERLAP else max(0, int(CHUNK_OVERLAP_MS / FRAME_MS))

    buf = np.zeros(0, dtype=np.float32)
    seg = []
    carry = np.zeros(0, dtype=np.float32)  # accumulate short segments instead of dropping
    speaking = False
    hang = 0
    silence_run = 0
    # minimum duration (in frames) to flush a lone short segment on long silence
    min_flush_frames = int(0.35 * 1000 / FRAME_MS)
    # tail frames to prepend to the next chunk to avoid boundary clipping
    overlap_prepend_frames: List[np.ndarray] = []

    while True:
        f_in = next(frames_iter)
        buf = np.concatenate([buf, f_in])

        while len(buf) >= frame_len:
            f = buf[:frame_len]; buf = buf[frame_len:]
            b = (np.clip(f, -1, 1) * 32767).astype(np.int16).tobytes()
            is_sp = vad.is_speech(b, SAMPLE_RATE)

            # Always use VAD to decide speech frames so we can split on silence
            if is_sp:
                speaking = True
                hang = hang_frames
                silence_run = 0
                if not DISABLE_OVERLAP and (not seg and overlap_prepend_frames):
                    seg.extend(overlap_prepend_frames)
                    overlap_prepend_frames = []
                seg.append(f)
                if len(seg) >= max_frames:
                    # compute overlap tail before clearing seg
                    audio = np.concatenate(seg, axis=0); seg = []
                    if not DISABLE_OVERLAP and overlap_frames > 0:
                        k = min(overlap_frames, len(audio) // frame_len)
                        if k > 0:
                            tail = audio[-k*frame_len:]
                            overlap_prepend_frames = [tail[i*frame_len:(i+1)*frame_len] for i in range(k)]
                    # prepend any carried short audio so we don't lose it (unless disabled)
                    if not DISABLE_CARRY and carry.size:
                        audio = np.concatenate([carry, audio]); carry = np.zeros(0, dtype=np.float32)
                    if len(audio) >= min_frames * frame_len:
                        yield audio
            else:
                if speaking:
                    hang -= 1
                    if hang <= 0:
                        speaking = False
                        if seg:
                            audio = np.concatenate(seg, axis=0); seg = []
                            if not DISABLE_OVERLAP and overlap_frames > 0:
                                k = min(overlap_frames, len(audio) // frame_len)
                                if k > 0:
                                    tail = audio[-k*frame_len:]
                                    overlap_prepend_frames = [tail[i*frame_len:(i+1)*frame_len] for i in range(k)]
                            if len(audio) < min_frames * frame_len:
                                if not DISABLE_CARRY:
                                    carry = np.concatenate([carry, audio]) if carry.size else audio
                                # if carry disabled, drop too-short tail
                            else:
                                if not DISABLE_CARRY and carry.size:
                                    audio = np.concatenate([carry, audio]); carry = np.zeros(0, dtype=np.float32)
                                yield audio
                else:
                    # accumulating silence while not speaking
                    silence_run += 1
                    # if we had a short carried segment and we've been silent long enough, flush it
                    if not DISABLE_CARRY and carry.size and silence_run >= 3 * hang_frames:
                        if len(carry) >= min_flush_frames * frame_len:
                            yield carry
                        # regardless, clear carry to avoid indefinite growth
                        carry = np.zeros(0, dtype=np.float32)

# =============== ASR with quality gate ===============
def good_seg(s):
    if CAPTURE_ALL:
        return True
    # Relaxed thresholds to avoid skipping plausible content
    if getattr(s, "no_speech_prob", 0) > 0.8: return False
    if getattr(s, "avg_logprob", 0) < -1.2:   return False
    if getattr(s, "compression_ratio", 0) > 2.4: return False
    return True

def _wav_bytes_from_pcm(audio: np.ndarray) -> bytes:
    buf = BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()

def _openai_transcribe(audio: np.ndarray) -> Optional[str]:
    url = os.getenv("OPENAI_AUDIO_URL", "https://api.openai.com/v1/audio/transcriptions")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {
        "file": ("audio.wav", _wav_bytes_from_pcm(audio), "audio/wav"),
    }
    data = {
        "model": WHISPER_MODEL,
        "language": "ja",
        "temperature": "0.0",
        "response_format": "json",
    }
    try:
        r = _HTTP.post(url, headers=headers, files=files, data=data, timeout=REMOTE_ASR_TIMEOUT_MS/1000)
        r.raise_for_status()
        j = r.json()
        # new API returns { text: ... }
        return (j.get("text") or "").strip()
    except Exception:
        return None

def _remote_transcribe(audio: np.ndarray) -> Optional[str]:
    url = REMOTE_ASR_URL
    if not url:
        return None
    wav_bytes = _wav_bytes_from_pcm(audio)
    files = {
        # server accepts keys: audio, file, or data — we use 'audio'
        "audio": ("audio.wav", wav_bytes, "audio/wav"),
    }
    data = {"language": "ja"}
    try:
        r = _HTTP.post(url, files=files, data=data, timeout=REMOTE_ASR_TIMEOUT_MS/1000)
        r.raise_for_status()
        j = r.json()
        # whisper server returns { transcript: "..." }
        txt = (j.get("transcript") or j.get("text") or "").strip()
        return txt
    except Exception:
        return None

_local_model_cache = [None]

def _ensure_local_model():
    if _local_model_cache[0] is None:
        _local_model_cache[0] = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type=COMPUTE_TYPE,
            cpu_threads=CPU_THREADS,
            num_workers=NUM_WORKERS,
        )
    return _local_model_cache[0]

def _local_transcribe_text(audio: np.ndarray, model=None) -> str:
    try:
        m = model if model is not None else _ensure_local_model()
        segs, _ = m.transcribe(
            audio,
            language="ja",
            beam_size=1 if CAPTURE_ALL else 5,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True
        )
        return ("".join(s.text for s in segs if good_seg(s)).strip())
    except Exception:
        return ""

def transcribe_iter(chunks_iter, model):
    for audio in chunks_iter:
        audio = preprocess(audio)
        if ASR_BACKEND == "remote":
            txt = _remote_transcribe(audio) or ""
            if not txt:
                # fallback to local
                try:
                    m = model if model is not None else _ensure_local_model()
                    segs, _ = m.transcribe(
                        audio,
                        language="ja",
                        beam_size=1,
                        temperature=0.0,
                        vad_filter=False,
                        condition_on_previous_text=False,
                        without_timestamps=True
                    )
                    txt = "".join(s.text for s in segs if good_seg(s)).strip()
                except Exception:
                    txt = ""
            if txt:
                yield txt
            continue
        if ASR_BACKEND == "openai":
            txt = _openai_transcribe(audio) or ""
            if not txt:
                # Fallback to local model to avoid dropping content
                try:
                    m = model if model is not None else _ensure_local_model()
                    segs, _ = m.transcribe(
                        audio,
                        language="ja",
                        beam_size=1,
                        temperature=0.0,
                        vad_filter=False,
                        condition_on_previous_text=False,
                        without_timestamps=True
                    )
                    txt = "".join(s.text for s in segs if good_seg(s)).strip()
                except Exception:
                    txt = ""
            if txt:
                yield txt
            continue
        # local faster-whisper
        segs, _ = model.transcribe(
            audio,
            language="ja",
            beam_size=1 if CAPTURE_ALL else 5,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True
        )
        txt = "".join(s.text for s in segs if good_seg(s)).strip()
        if txt:
            yield txt

def transcribe_chunk(audio: np.ndarray, model) -> str:
    audio = preprocess(audio)
    if ASR_BACKEND == "remote":
        txt = _remote_transcribe(audio) or ""
        if txt:
            return txt
        # fallback to local
        try:
            m = model if model is not None else _ensure_local_model()
            segs, _ = m.transcribe(
                audio,
                language="ja",
                beam_size=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True
            )
            return ("".join(s.text for s in segs if good_seg(s)).strip())
        except Exception:
            return ""
    if ASR_BACKEND == "openai":
        txt = _openai_transcribe(audio) or ""
        if txt:
            return txt
        try:
            m = model if model is not None else _ensure_local_model()
            segs, _ = m.transcribe(
                audio,
                language="ja",
                beam_size=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True
            )
            return ("".join(s.text for s in segs if good_seg(s)).strip())
        except Exception:
            return ""
    # local
    segs, _ = model.transcribe(
        audio,
        language="ja",
        beam_size=1 if CAPTURE_ALL else 5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True
    )
    return ("".join(s.text for s in segs if good_seg(s)).strip())

# =============== Caching (SQLite + LRU) ===============
_CACHE_DB = "dict_cache.sqlite"
def _db():
    conn = sqlite3.connect(_CACHE_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS deepl (k TEXT PRIMARY KEY, v TEXT, ts REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS jisho (k TEXT PRIMARY KEY, v TEXT, ts REAL)")
    return conn

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# =============== DeepL (per-utterance) ===============
try:
    from deep_translator import DeeplTranslator
    _deepl = DeeplTranslator(source="ja", target="en", api_key=os.getenv("DEEPL_API_KEY"))
except Exception:
    _deepl = None

def cached_deepl_translate(text: str) -> Optional[str]:
    if not _deepl: return None
    if len(text) < MIN_CHARS_FOR_DEEPL: return None
    # very quick check: only call if text has Japanese script
    if not re.search(r"[ぁ-んァ-ヴ一-龯]", text): return None
    conn = _db()
    k = _sha1("deepl:" + text)
    row = conn.execute("SELECT v FROM deepl WHERE k=?", (k,)).fetchone()
    if row:
        conn.close(); return row[0]
    try:
        en = _deepl.translate(text)
        conn.execute("INSERT OR REPLACE INTO deepl (k,v,ts) VALUES (?,?,?)", (k, en, time.time()))
        conn.commit(); conn.close()
        return en
    except Exception:
        conn.close(); return None

# =============== Jisho (per-word, cached + throttled) ===============
_last_jisho = [0.0]
def _throttle():
    dt = time.time() - _last_jisho[0]
    if dt < JISHO_MIN_INTERVAL_S:
        time.sleep(JISHO_MIN_INTERVAL_S - dt)
    _last_jisho[0] = time.time()

@functools.lru_cache(maxsize=2048)
def _jisho_lookup_mem(word: str):
    try:
        _throttle()
        r = requests.get(
            "https://jisho.org/api/v1/search/words",
            params={"keyword": word},
            headers={"User-Agent":"gelato-overlay/1.0"},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("data"):
            return data["data"][0]
    except Exception:
        return None
    return None

def jisho_lookup(word: str):
    conn = _db()
    row = conn.execute("SELECT v FROM jisho WHERE k=?", (word,)).fetchone()
    if row:
        conn.close()
        try: return json.loads(row[0])
        except: return None
    data = _jisho_lookup_mem(word)
    if data is not None:
        conn.execute("INSERT OR REPLACE INTO jisho (k,v,ts) VALUES (?,?,?)",
                     (word, json.dumps(data, ensure_ascii=False), time.time()))
        conn.commit()
    conn.close()
    return data

def dict_has_entry(s: str) -> bool:
    return jisho_lookup(s) is not None

# =============== User dictionary (overrides) ===============
# CSV columns: surface,reading,english
def load_user_dict(path: str) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    d = {}
    if not os.path.exists(path): return d
    try:
        import csv
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"): continue
                surface = row[0].strip()
                reading = row[1].strip() if len(row) > 1 and row[1].strip() else None
                english = row[2].strip() if len(row) > 2 and row[2].strip() else None
                if surface:
                    d[surface] = (reading, english)
    except Exception:
        pass
    return d

USER_OVERRIDES = load_user_dict(USER_DICT_CSV)

# =============== Annotation engine ===============
_kks = kakasi(); _kks.setMode("J","H")
_conv = _kks.getConverter()

KANJI_RE = re.compile(r"[一-龯々〆ヵヶ]")
KATA_LETTER_RE = re.compile(r"[ァ-ヴ]")  # excludes 'ー'
NUM_RE = re.compile(r"\d+")
KANJI_NUM_RE = re.compile(r"[一二三四五六七八九十百千]+")
PUNCT_POS = {"記号"}
SKIP_POS = {"助詞","助動詞"}

def to_hira(s: str) -> str:
    return "".join(p["hira"] for p in _conv.convert(s))

def get_reading(m) -> Union[str, None]:
    return getattr(m.feature, "pron", None) or getattr(m.feature, "kana", None)

def has_kata_letter(s: str) -> bool:
    return bool(KATA_LETTER_RE.search(s))

# Counters 1..10
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

def best_gloss(word: str, prefer_counter=False) -> Union[str, None]:
    # User override first
    ov = USER_OVERRIDES.get(word)
    if ov and ov[1]:
        return ov[1]

    entry = jisho_lookup(word)
    if not entry:
        return None

    cands = []
    for sense in entry.get("senses", []):
        pos_tags = set(sense.get("parts_of_speech") or [])
        for gloss in sense.get("english_definitions") or []:
            score = 0
            if "Counter" in pos_tags: score += 3
            if "Noun" in pos_tags:    score += 2
            if "Suffix" in pos_tags:  score -= 2
            if prefer_counter and "Counter" in pos_tags: score += 2
            cands.append((score, gloss))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]

def jamlike_has_entry(s: str) -> bool:
    return dict_has_entry(s)

def compound_reading(tokens) -> str:
    kana = []
    for m in tokens:
        k = get_reading(m)
        kana.append(k if k else m.surface)
    return to_hira("".join(kana))

def maybe_merge_compound(tokens, i, max_len=3):
    for L in (3,2):
        if i+L <= len(tokens):
            seg = tokens[i:i+L]
            if all(getattr(t.feature,"pos1","")=="名詞" for t in seg):
                surf = "".join(t.surface for t in seg)
                if KANJI_RE.search(surf) and jamlike_has_entry(surf):
                    # user dict override?
                    ov = USER_OVERRIDES.get(surf)
                    if ov:
                        reading = ov[0] if ov[0] else compound_reading(seg)
                        gloss = ov[1]
                    else:
                        reading = compound_reading(seg)
                        gloss = best_gloss(surf) or ""
                    anno = f"{surf}({reading}" + (f"、{gloss})" if gloss else ")")
                    return anno, L
    return None

def annotate_text(text: str) -> str:
    toks = list(tagger(text))
    out = []; i = 0
    # per-utterance de-dupe: avoid repeated remote lookups for same lemma
    seen_lemmas = set()

    while i < len(toks):
        m = toks[i]
        surf = m.surface
        pos1 = getattr(m.feature, "pos1", "")
        lemma = getattr(m.feature, "lemma", None) or surf

        # Compound merge first
        merged = maybe_merge_compound(toks, i)
        if merged:
            anno, L = merged
            out.append(anno); i += L; continue

        # Skip punctuation/particles/aux
        if pos1 in PUNCT_POS or pos1 in SKIP_POS:
            out.append(surf); i += 1; continue

        # Counter: [number][counter] or prev number + counter
        if len(surf) >= 2 and surf[-1] in COUNTER_READ:
            n = to_int_number(surf[:-1])
            if n and 1 <= n <= 10:
                reading = COUNTER_READ[surf[-1]][n-1]
                gloss = best_gloss(surf[-1], prefer_counter=True)
                out.append(f"{surf}({reading}" + (f"、{gloss})" if gloss else ")"))
                i += 1; continue
        if surf in COUNTER_READ and i > 0:
            n = to_int_number(toks[i-1].surface)
            if n and 1 <= n <= 10:
                reading = COUNTER_READ[surf][n-1]
                gloss = best_gloss(surf, prefer_counter=True)
                out.append(f"{surf}({reading}" + (f"、{gloss})" if gloss else ")"))
                i += 1; continue

        # Annotate only if kanji present or a katakana letter (not just 'ー')
        annotatable = bool(KANJI_RE.search(surf) or has_kata_letter(surf))
        if not annotatable:
            out.append(surf); i += 1; continue

        # User overrides
        if surf in USER_OVERRIDES or lemma in USER_OVERRIDES:
            rd, en = USER_OVERRIDES.get(surf, USER_OVERRIDES.get(lemma))
            reading = rd if rd else (to_hira(get_reading(m)) if get_reading(m) else to_hira(surf))
            if en:
                out.append(f"{surf}({reading}、{en})")
            else:
                out.append(f"{surf}({reading})")
            i += 1; continue

        # Reading: fugashi > kakasi
        k = get_reading(m)
        reading = to_hira(k) if k else to_hira(surf)

        prefer_counter = False
        if surf == "人" and not (i > 0 and to_int_number(toks[i-1].surface)):
            prefer_counter = False

        # De-dupe lookups per utterance (still cached globally anyway)
        gloss = None
        if lemma not in seen_lemmas:
            gloss = best_gloss(lemma, prefer_counter=prefer_counter)
            seen_lemmas.add(lemma)
        if gloss is None:
            gloss = best_gloss(surf, prefer_counter=prefer_counter)

        out.append(f"{surf}({reading}" + (f"、{gloss})" if gloss else ")"))
        i += 1
    return "".join(out)

# =============== IO helpers ===============
def append_transcript(text: str):
    with open(TRANSCRIPT_OUT, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# =============== Suggestions persistence (for right sidebar) ===============
SUGGESTIONS_OUT = "suggestions.json"

def _enrich_suggestion_fields(jp: str, rd: str, en: str) -> (str, str):
    try:
        rd_out = rd
        if not rd_out:
            toks = list(tagger(jp))
            if toks:
                kana = []
                for m in toks:
                    k = get_reading(m)
                    kana.append(k if k else m.surface)
                rd_out = to_hira("".join(kana))
            else:
                rd_out = to_hira(jp)
        en_out = en or (best_gloss(jp) or "")
        return rd_out, en_out
    except Exception:
        return rd, en

def update_suggestions(items):
    """Append new suggestions to suggestions.json, de-duplicated by jp string."""
    try:
        import json as _json
        existing = []
        try:
            with open(SUGGESTIONS_OUT, "r", encoding="utf-8") as f:
                existing = _json.load(f) or []
        except Exception:
            existing = []
        new_items = []
        batch_seen = set()
        now_ts = time.time()
        for it in (items or []):
            jp = (it.get("jp") or it.get("ja") or "").strip()
            rd = (it.get("reading_kana") or it.get("reading") or "").strip()
            en = (it.get("en") or it.get("meaning_en") or "").strip()
            if not jp:
                continue
            # avoid duplicates within the same batch only
            if jp in batch_seen:
                continue
            batch_seen.add(jp)
            rd2, en2 = _enrich_suggestion_fields(jp, rd, en)
            new_items.append({"jp": jp, "reading_kana": rd2 or "", "en": en2 or "", "ts": now_ts})
        if not new_items:
            return
        out = (existing + new_items)[-800:]
        tmp = SUGGESTIONS_OUT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(out, f, ensure_ascii=False)
        os.replace(tmp, SUGGESTIONS_OUT)
    except Exception:
        pass

# =============== ASR debug compare (openai vs remote) ===============
ASR_DEBUG_FILE = "asr_debug.json"
_ASR_DEBUG_LOCK = threading.Lock()

def _asr_debug_update(item_id: str, backend: str, text: Optional[str], ms: Optional[int]):
    try:
        import json as _json
        with _ASR_DEBUG_LOCK:
            data = []
            try:
                with open(ASR_DEBUG_FILE, "r", encoding="utf-8") as f:
                    data = _json.load(f) or []
            except Exception:
                data = []
            found = None
            for it in data:
                if it.get("id") == item_id:
                    found = it; break
            if not found:
                found = {"id": item_id, "ts": int(time.time()*1000)}
                data.append(found)
            found[backend] = {"text": text or "", "ms": ms if ms is not None else None}
            data = data[-300:]
            tmp = ASR_DEBUG_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, ASR_DEBUG_FILE)
    except Exception:
        pass

# =============== GPT formatter (annotation + translation) ===============
def _gpt_request(jp_text: str) -> Optional[Dict]:
    if not USE_GPT_FORMATTER:
        return None
    url = os.getenv("OPENAI_CHAT_URL", "https://api.openai.com/v1/chat/completions")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    system = (
        "You are a precise Japanese annotator and translator. "
        "Given a single Japanese utterance, return STRICT JSON with ONLY these keys: "
        "jp_html, en_html, vocabs, suggestions. "
        "jp_html: EXACT utterance verbatim (no paraphrase/normalization). Preserve all characters and whitespace. Do NOT add parentheses or readings; only wrap spans <span data-id=\\\"vN\\\">…</span> for chosen vocab occurrences. "
        "en_html: faithful, natural English translation covering the entire utterance. Wrap the English phrase corresponding to each vocab with the same <span data-id=\\\"vN\\\">…</span>. "
        "vocabs: array of items ACTUALLY PRESENT in the input. Each: { id: 'v1', surface, reading_kana, meaning_en }. Prioritize nouns/verbs/adjectives/set phrases; avoid particles/aux unless semantically important. "
        "suggestions: 3–8 related words or short replies (NOT appearing in vocabs.surface) suitable to say next in Japanese, each as { jp, reading_kana, en }. Keep concise and context-relevant; avoid duplication with vocabs.surface. "
        "Constraints: Only annotate tokens containing Japanese script. If no Japanese in input, set vocabs=[] and suggestions may be empty. Every vocabs[i].id must appear at least once as <span data-id=\\\"vN\\\"> in BOTH jp_html and en_html. Keep glosses 1–5 words. VALID JSON only."
    )
    user = {
        "utterance": jp_text,
        "compress_elongations": bool(GPT_COMPRESS_ELONG),
        "jp_verbatim": True,
        "max_vocab": int(GPT_MAX_VOCAB),
    }
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    }
    try:
        r = _HTTP.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        j = r.json()
        content = j["choices"][0]["message"]["content"].strip()
        return json.loads(content)
    except Exception:
        return None

_PALETTE = [
    "#d32f2f", "#1976d2", "#388e3c", "#f57c00", "#7b1fa2", "#0097a7",
    "#c2185b", "#512da8", "#00796b", "#afb42b", "#5d4037", "#0288d1",
]

def _colorize_spans(html_text: str, color_map: Dict[str, str]) -> str:
    def repl(m):
        head = m.group(1)
        vid = m.group(2)
        tail = m.group(3)
        color = color_map.get(vid)
        if not color:
            return m.group(0)
        if "style=" in head:
            return f"{head[:-1]}; color:{color}{tail}"
        return f"{head} style=\"color:{color}\"{tail}"
    return re.sub(r"(<span\b[^>]*data-id=\"([^\"]+)\"[^>]*)(>)", repl, html_text)

def _strip_span_tags(s: str) -> str:
    # Remove span tags but keep inner text; also unescape entities for comparison
    s_no_tags = re.sub(r"</?span\b[^>]*>", "", s)
    try:
        return html.unescape(s_no_tags)
    except Exception:
        return s_no_tags

def _render_four_lines(j: Dict) -> List[str]:
    # build stable colors per id in vocabs order
    ids = [v.get("id") for v in (j.get("vocabs") or []) if v.get("id")]
    color_map = {}
    for i, vid in enumerate(ids):
        color_map[vid] = _PALETTE[i % len(_PALETTE)]

    jp = j.get("jp_html") or html.escape(j.get("jp", "").strip())
    en = j.get("en_html") or ""
    jp_c = _colorize_spans(jp, color_map)
    en_c = _colorize_spans(en, color_map)
    en_c = f"<span style=\"color:#888\">{en_c}</span>"

    voc_items = []
    for v in (j.get("vocabs") or []):
        vid = v.get("id")
        surf = v.get("surface") or ""
        rd = v.get("reading_kana") or ""
        mean = v.get("meaning_en") or ""
        color = color_map.get(vid, "#888")
        voc_items.append(
            f"<span data-id=\"{html.escape(vid or '')}\" style=\"color:{color}\">{html.escape(surf)}({html.escape(rd)}) — {html.escape(mean)}</span>"
        )
    vocab_line = "Vocab: " + (" ・ ".join(voc_items) if voc_items else "(none)")

    # Line 4 composition
    discourse = [str(x) for x in (j.get("discourse") or []) if x]
    connectives = [str(x) for x in (j.get("connectives") or []) if x]
    cv = j.get("contextual_vocab") or []
    cv_parts = []
    for item in cv:
        s = item.get("surface") or ""
        r = item.get("reading_kana") or ""
        m = item.get("meaning_en") or ""
        if s:
            cv_parts.append(f"{html.escape(s)}({html.escape(r)}) — {html.escape(m)}")
    tense = j.get("tense_cheat") or {}
    tp = tense.get("present_polite") or ""
    tpp = tense.get("past_polite") or ""
    tc = tense.get("present_casual") or ""
    tcc = tense.get("past_casual") or ""
    fb = j.get("fallback") or {}
    fb_jp = fb.get("jp") or ""
    fb_rd = fb.get("reading_kana") or ""
    fb_en = fb.get("en") or ""

    parts = []
    # Icon compact style (Option C)
    markers = []
    if discourse:
        markers.extend(discourse)
    if connectives:
        markers.extend(connectives)
    if markers:
        parts.append("🏷️ Markers: " + " • ".join(html.escape(x) for x in markers))
    if cv_parts:
        parts.append("📚 Context Vocab: " + " • ".join(cv_parts))

    # Personal vocab (profile-based)
    pv = j.get("personal_vocab") or []
    pv_parts = []
    for item in pv:
        s = item.get("surface") or ""
        r = item.get("reading_kana") or ""
        m = item.get("meaning_en") or ""
        if s:
            pv_parts.append(f"{html.escape(s)}({html.escape(r)}) — {html.escape(m)}")
    if pv_parts:
        parts.append("🧩 Personal: " + " • ".join(pv_parts))
    # Tense with readings
    if any([tp, tpp, tc, tcc]):
        tpr = (tense.get("present_polite_reading") or "").strip()
        tppr = (tense.get("past_polite_reading") or "").strip()
        tcr = (tense.get("present_casual_reading") or "").strip()
        tccr = (tense.get("past_casual_reading") or "").strip()
        tense_bits = []
        if tp or tpp:
            left = html.escape(tp)
            right = html.escape(tpp)
            left_r = f"({html.escape(tpr)})" if tpr else ""
            right_r = f"({html.escape(tppr)})" if tppr else ""
            tense_bits.append(f"{left}{left_r}/{right}{right_r}".strip("/"))
        if tc or tcc:
            left = html.escape(tc)
            right = html.escape(tcc)
            left_r = f"({html.escape(tcr)})" if tcr else ""
            right_r = f"({html.escape(tccr)})" if tccr else ""
            tense_bits.append(f"{left}{left_r}/{right}{right_r}".strip("/"))
        if tense_bits:
            parts.append("⏱ Tense: " + " • ".join(tense_bits))
    if fb_jp:
        fb_str = f"🛟 Fallback: {html.escape(fb_jp)} ({html.escape(fb_rd)}) — {html.escape(fb_en)}"
        parts.append(fb_str)

    # Compliments and mood phrases (optional)
    comps = j.get("compliments") or []
    comp_parts = []
    for c in comps:
        jp = c.get("jp") or ""
        rd = c.get("reading_kana") or ""
        en = c.get("en") or ""
        if jp:
            comp_parts.append(f"{html.escape(jp)} ({html.escape(rd)}) — {html.escape(en)}")
    if comp_parts:
        parts.append("💐 Compliments: " + " • ".join(comp_parts))

    mood = j.get("mood") or ""
    mood_phrases = j.get("mood_phrases") or []
    if mood and mood_phrases:
        mp_parts = []
        for c in mood_phrases:
            jp = c.get("jp") or ""
            rd = c.get("reading_kana") or ""
            en = c.get("en") or ""
            if jp:
                mp_parts.append(f"{html.escape(jp)} ({html.escape(rd)}) — {html.escape(en)}")
        if mp_parts:
            parts.append("🧭 Mood: " + html.escape(str(mood)) + ": " + " • ".join(mp_parts))

    tips = j.get("tips") or []
    if tips:
        parts.append("💬 Replies: " + " / ".join(html.escape(t) for t in tips))

    if LINE4_TABLE_MODE:
        # Build table columns with one item per row per column
        col_markers = [html.escape(x) for x in (discourse + connectives)]
        col_context = cv_parts[:]  # already escaped/assembled as s(reading) — meaning
        col_personal = pv_parts[:]  # already escaped/assembled
        col_tense = []
        if any([tp, tpp]):
            label = "Present polite"
            left_r = f"({html.escape(tense.get('present_polite_reading') or '')})" if tense.get('present_polite_reading') else ""
            right_r = f"({html.escape(tense.get('past_polite_reading') or '')})" if tense.get('past_polite_reading') else ""
            col_tense.append(f"{label} — {html.escape(tp)}{left_r} / {html.escape(tpp)}{right_r}")
        if any([tc, tcc]):
            label = "Present casual"
            left_r = f"({html.escape(tense.get('present_casual_reading') or '')})" if tense.get('present_casual_reading') else ""
            right_r = f"({html.escape(tense.get('past_casual_reading') or '')})" if tense.get('past_casual_reading') else ""
            col_tense.append(f"{label} — {html.escape(tc)}{left_r} / {html.escape(tcc)}{right_r}")
        col_fallback = []
        if fb_jp:
            col_fallback.append(f"{html.escape(fb_jp)} ({html.escape(fb_rd)}) — {html.escape(fb_en)}")
        comp_parts = []
        for c in (j.get("compliments") or []):
            jp = c.get("jp") or ""
            rd = c.get("reading_kana") or ""
            en = c.get("en") or ""
            if jp:
                comp_parts.append(f"{html.escape(jp)} ({html.escape(rd)}) — {html.escape(en)}")
        col_mood = []
        mlabel = (j.get("mood") or "").strip()
        for c in (j.get("mood_phrases") or []):
            jp = c.get("jp") or ""
            rd = c.get("reading_kana") or ""
            en = c.get("en") or ""
            if jp:
                prefix = (html.escape(mlabel) + ": ") if mlabel else ""
                col_mood.append(prefix + f"{html.escape(jp)} ({html.escape(rd)}) — {html.escape(en)}")
        col_replies = [html.escape(t) for t in (j.get("tips") or [])]

        columns = [col_markers, col_context, col_personal, col_tense, col_fallback, col_mood, col_replies]
        max_rows = max([len(c) for c in columns] + [1])
        # Normalize column lengths
        norm = []
        for col in columns:
            rows = col[:] + [""] * (max_rows - len(col))
            norm.append(rows)
        # Build HTML table
        header = (
            "<tr>"
            "<th style=\"text-align:left;padding:4px 6px;\">🏷️ Markers</th>"
            "<th style=\"text-align:left;padding:4px 6px;\">📚 Context Vocab</th>"
            "<th style=\"text-align:left;padding:4px 6px;\">🧩 Personal</th>"
            "<th style=\"text-align:left;padding:4px 6px;\">⏱ Tense</th>"
            "<th style=\"text-align:left;padding:4px 6px;\">🛟 Fallback</th>"
            "<th style=\"text-align:left;padding:4px 6px;\">🧭 Mood</th>"
            "<th style=\"text-align:left;padding:4px 6px;\">💬 Replies</th>"
            "</tr>"
        )
        body_rows = []
        for i in range(max_rows):
            cells = [norm[c][i] for c in range(len(norm))]
            tds = "".join(
                f"<td style=\"vertical-align:top;padding:4px 6px;\">{cell}</td>" for cell in cells
            )
            body_rows.append(f"<tr>{tds}</tr>")
        table_html = (
            "<table style=\"border-collapse:collapse;width:100%;\">"
            f"<thead style=\"background:#f7f7f7;\">{header}</thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
        )
        line4 = table_html
    else:
        # Stack each section on its own line for readability
        line4 = "<br>".join(p for p in parts if p)

    lines: List[str] = [jp_c, en_c]
    if SHOW_VOCAB_LINE:
        lines.append(vocab_line)
    # No extra details line appended
    return lines

# =============== Sanitizers for GPT (keeps JP verbatim for display) ===============
def _collapse_long_runs(s: str, max_run: int = 12) -> str:
    # Limit long runs of the same character (e.g., ーーー or repeated kana)
    return re.sub(r"(.)\1{" + str(max_run) + r",}", lambda m: m.group(1) * max_run, s)

def _collapse_repeated_phrases(s: str, min_len: int = 2, max_len: int = 8, keep: int = 3) -> str:
    out = s
    try:
        for n in range(max_len, min_len-1, -1):
            pattern = re.compile(r"((?:.{" + str(n) + r"}))\1{3,}")
            out = pattern.sub(lambda m: m.group(1) * keep, out)
    except Exception:
        return s
    return out

def _repetition_score(s: str) -> float:
    if not s:
        return 0.0
    # crude: compression ratio proxy via removing duplicates of 3+ char phrases
    cleaned = _collapse_repeated_phrases(_collapse_long_runs(s), 2, 8, 1)
    try:
        return len(s) / max(1, len(cleaned))
    except Exception:
        return 1.0

def _sanitize_for_gpt(s: str) -> Optional[str]:
    if not s:
        return s
    t = s
    if GPT_CLEAN_REPEATS:
        t = _collapse_long_runs(t)
        t = _collapse_repeated_phrases(t)
    if len(t) > GPT_MAX_CHARS:
        t = t[:GPT_MAX_CHARS]
    if GPT_SKIP_ON_REPEAT and _repetition_score(s) >= 4.0:
        # too repetitive; skip GPT to avoid tokens
        return None
    return t

# =============== Main ===============
def main():
    dev_idx, dev_name = pick_input(TARGET_DEVICE_SUBSTR)
    print(f"Listening on: {dev_name}")
    open(TRANSCRIPT_OUT, "w", encoding="utf-8").close()
    q = queue.Queue(maxsize=128)
    wave_q = queue.Queue(maxsize=16)
    threading.Thread(target=audio_thread, args=(dev_idx, q, wave_q), daemon=True).start()

    # Build local model if needed (used in ASR thread as fallback or primary when local)
    model = None
    if ASR_BACKEND in ("local", "remote", "openai") and not (ASR_BACKEND == "openai" and USE_OPENAI is False):
        # Only construct if local is possible/used as fallback
        try:
            model = WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS,
                num_workers=NUM_WORKERS,
            )
        except Exception:
            model = None

    # Queues between stages
    chunk_q: queue.Queue = queue.Queue(maxsize=64)
    text_q: queue.Queue = queue.Queue(maxsize=128)

    # Stage 1: chunking + waveform flushing
    threading.Thread(target=chunker_thread, args=(q, chunk_q, wave_q), daemon=True).start()

    # Stage 2: ASR uploader/decoder
    def asr_worker():
        while True:
            audio = chunk_q.get()
            try:
                txt = transcribe_chunk(audio, model)
            except Exception:
                txt = ""
            # Debug compare: fire openai and remote on the same chunk (non-blocking)
            if SHOW_ASR_DEBUG:
                try:
                    def run_openai(a):
                        if not OPENAI_API_KEY:
                            return
                        t0 = time.time(); r = _openai_transcribe(a) or ""; ms = int((time.time()-t0)*1000)
                        _asr_debug_update(item_id, "openai", r, ms)
                    def run_remote(a):
                        if not REMOTE_ASR_URL:
                            return
                        t0 = time.time(); r = _remote_transcribe(a) or ""; ms = int((time.time()-t0)*1000)
                        _asr_debug_update(item_id, "remote", r, ms)
                    def run_local(a):
                        t0 = time.time(); r = _local_transcribe_text(a, model) or ""; ms = int((time.time()-t0)*1000)
                        _asr_debug_update(item_id, "local", r, ms)
                    item_id = f"{int(time.time()*1000)}-{hashlib.md5(audio.tobytes()).hexdigest()[:6]}"
                    threading.Thread(target=run_openai, args=(audio.copy(),), daemon=True).start()
                    threading.Thread(target=run_remote, args=(audio.copy(),), daemon=True).start()
                    threading.Thread(target=run_local, args=(audio.copy(),), daemon=True).start()
                except Exception:
                    pass
            if txt:
                text_q.put(txt)

    # Start N ASR workers for concurrency so a slow request won't block new chunks
    for _ in range(max(1, ASR_WORKERS)):
        threading.Thread(target=asr_worker, daemon=True).start()

    # Stage 3: Formatter/writer
    GPT_LAST_TS = {"t": 0.0}
    gpt_lock = threading.Lock()

    def formatter_worker():
        last_text = ""; last_t = 0.0
        while True:
            text = text_q.get().strip()
            if not text:
                continue
            now = time.time()
            if last_text and (now - last_t) <= 2.0:
                if text == last_text or text.startswith(last_text) or last_text.startswith(text):
                    continue
            out_lines: List[str] = []
            if not re.search(r"[ぁ-んァ-ヴ一-龯]", text):
                # Non-Japanese: use GPT EN as line 2 by sending as-is; line 1 mirrors
                payload_text = _sanitize_for_gpt(text) or text
                # Rate limit with blocking wait to avoid fallback output
                with gpt_lock:
                    now2 = time.time()
                    elapsed_ms = (now2 - GPT_LAST_TS["t"]) * 1000.0
                    wait_ms = GPT_RATE_LIMIT_MS - elapsed_ms
                    if wait_ms > 0:
                        time.sleep(wait_ms/1000.0)
                    GPT_LAST_TS["t"] = time.time()
                j = _gpt_request(payload_text)
                if j:
                    try:
                        out_lines = _render_four_lines(j)
                        if SHOW_DETAILS_LINE:
                            try:
                                update_suggestions(j.get("suggestions") or [])
                            except Exception:
                                pass
                    except Exception:
                        out_lines = []
            else:
                # Japanese present: always produce GPT output (block for rate limit)
                payload_text = _sanitize_for_gpt(text)
                if payload_text:
                    with gpt_lock:
                        now2 = time.time()
                        elapsed_ms = (now2 - GPT_LAST_TS["t"]) * 1000.0
                        wait_ms = GPT_RATE_LIMIT_MS - elapsed_ms
                        if wait_ms > 0:
                            time.sleep(wait_ms/1000.0)
                        GPT_LAST_TS["t"] = time.time()
                    j = _gpt_request(payload_text)
                    if j:
                        try:
                            out_lines = _render_four_lines(j)
                            if SHOW_DETAILS_LINE:
                                try:
                                    update_suggestions(j.get("suggestions") or [])
                                except Exception:
                                    pass
                        except Exception:
                            out_lines = []
            if not out_lines:
                # If GPT failed or returned nothing, skip this utterance entirely
                continue
            # Drop empty line4 if it has no content (just in case)
            if len(out_lines) == 4 and (not out_lines[-1] or out_lines[-1].strip() == ""):
                pass
            for line in out_lines:
                append_transcript(line)
                print(line)
            last_text = text; last_t = now
            append_transcript("")
            print("")

    for _ in range(max(1, FORMAT_WORKERS)):
        threading.Thread(target=formatter_worker, daemon=True).start()

    # Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
