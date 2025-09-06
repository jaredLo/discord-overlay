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

# Capture-all mode
CAPTURE_ALL = os.getenv("CAPTURE_ALL", "false").lower() in {"1","true","yes","y"}

# GPT formatting (annotation + translation)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
USE_GPT_FORMATTER = (os.getenv("USE_GPT_FORMATTER", "true").lower() in {"1","true","yes","y"}) and bool(OPENAI_API_KEY)
GPT_COMPRESS_ELONG = os.getenv("GPT_COMPRESS_ELONGATIONS", "false").lower() in {"1","true","yes","y"}
GPT_MAX_VOCAB = int(os.getenv("GPT_MAX_VOCAB", "10"))

# Output
TRANSCRIPT_OUT = "transcript.txt"
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

    buf = np.zeros(0, dtype=np.float32)
    seg = []
    carry = np.zeros(0, dtype=np.float32)  # accumulate short segments instead of dropping
    speaking = False
    hang = 0
    silence_run = 0
    # minimum duration (in frames) to flush a lone short segment on long silence
    min_flush_frames = int(0.35 * 1000 / FRAME_MS)

    while True:
        f_in = next(frames_iter)
        buf = np.concatenate([buf, f_in])

        while len(buf) >= frame_len:
            f = buf[:frame_len]; buf = buf[frame_len:]
            b = (np.clip(f, -1, 1) * 32767).astype(np.int16).tobytes()
            is_sp = vad.is_speech(b, SAMPLE_RATE)

            if is_sp or CAPTURE_ALL:
                speaking = True
                hang = hang_frames
                silence_run = 0
                seg.append(f)
                if len(seg) >= max_frames:
                    audio = np.concatenate(seg, axis=0); seg = []
                    # prepend any carried short audio so we don't lose it
                    if carry.size:
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
                            # if too short, carry it forward instead of dropping
                            if len(audio) < min_frames * frame_len:
                                carry = np.concatenate([carry, audio]) if carry.size else audio
                            else:
                                if carry.size:
                                    audio = np.concatenate([carry, audio]); carry = np.zeros(0, dtype=np.float32)
                                yield audio
                else:
                    # accumulating silence while not speaking
                    silence_run += 1
                    # if we had a short carried segment and we've been silent long enough, flush it
                    if carry.size and silence_run >= 3 * hang_frames:
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
        r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        r.raise_for_status()
        j = r.json()
        # new API returns { text: ... }
        return (j.get("text") or "").strip()
    except Exception:
        return None

def transcribe_iter(chunks_iter, model):
    for audio in chunks_iter:
        audio = preprocess(audio)
        if USE_OPENAI:
            txt = _openai_transcribe(audio) or ""
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
        "jp_html, en_html, vocabs, discourse, connectives, contextual_vocab, personal_vocab, tense_cheat, compliments, mood, mood_phrases, fallback, tips. "
        "jp_html: Full Japanese preserving order/punctuation. Annotate key vocabulary inline as surface(reading); reading is kana only. Kanji words use hiragana; katakana loanwords keep katakana. Wrap each chosen vocab occurrence in <span data-id=\\\"vN\\\">…</span> so ids align with English and vocab list. "
        "en_html: Faithful, natural English translation covering the entire utterance. Wrap the English phrase corresponding to each vocab with the same <span data-id=\\\"vN\\\">…</span>. "
        "vocabs: Array of items actually present in input. Each: { id: 'v1', surface, reading_kana, meaning_en }. Prioritize nouns, verbs, adjectives, set phrases, counters; avoid particles/aux unless semantically important. "
        "discourse: 1–3 discourse markers appropriate to the context (e.g., ところで, ちなみに, それなら, それで, でも). "
        "connectives: 1–3 connectives appropriate to the context (e.g., しかし, なので, つまり, それに). "
        "contextual_vocab: exactly 3 items { surface, reading_kana, meaning_en } relevant to the current topic; HOWEVER if the utterance asks where/when/what/time-of-year, replace these with relative-time examples such as 二年前, 二週間前, 三ヶ月前 (with readings/meanings). "
        "personal_vocab: exactly 3 items from the user's profile (motorcycle, programmer/job in software, freestyle swimming/watersports, Malaysian identity), as { surface, reading_kana, meaning_en }. "
        "tense_cheat: main action verb (or best candidate) with BOTH forms and readings: { present_polite, present_polite_reading, past_polite, past_polite_reading, present_casual, present_casual_reading, past_casual, past_casual_reading }. "
        "compliments: if appropriate to compliment, 3 short options as [{ jp, reading_kana, en }]. "
        "mood: optional short label of detected mood (e.g., encouraging/empathetic), and mood_phrases: 3 short options as [{ jp, reading_kana, en }]. "
        "fallback: a single concise, polite rescue phrase suited to the context: { jp, reading_kana, en }. "
        "tips: 1–2 concise reply suggestions in Japanese (optionally brief EN gloss). "
        "Constraints: No invented content. Every vocabs[i].id must appear at least once as <span data-id=\\\"vN\\\"> in BOTH jp_html and en_html. Keep outputs concise; glosses 1–5 words. "
        "If compress_elongations=true, shorten extreme ー/〜 runs and add an ellipsis; otherwise preserve as written. "
        "Select up to max_vocab items for jp/en spans. Respond with VALID JSON only—no Markdown, no extra text."
    )
    user = {
        "utterance": jp_text,
        "compress_elongations": bool(GPT_COMPRESS_ELONG),
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
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
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

    # Stack each section on its own line for readability
    line4 = "<br>".join(p for p in parts if p)

    return [jp_c, en_c, vocab_line, line4]

# =============== Main ===============
def main():
    dev_idx, dev_name = pick_input(TARGET_DEVICE_SUBSTR)
    print(f"Listening on: {dev_name}")
    open(TRANSCRIPT_OUT, "w", encoding="utf-8").close()
    q = queue.Queue(maxsize=128)
    threading.Thread(target=audio_thread, args=(dev_idx, q), daemon=True).start()

    def frames():
        while True:
            yield q.get()

    model = None
    if not USE_OPENAI:
        model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type=COMPUTE_TYPE,
            cpu_threads=CPU_THREADS,
            num_workers=NUM_WORKERS,
        )

    def chunks():
        while True:
            f = q.get()
            yield f

    for text in transcribe_iter(utterance_chunks(chunks()), model):
        text = text.strip()
        if not text:
            continue
        out_lines: List[str] = []
        j = _gpt_request(text)
        if j:
            try:
                out_lines = _render_four_lines(j)
            except Exception:
                out_lines = []
        if not out_lines:
            # Fallback minimal: inline annotation only, no GPT
            annotated = annotate_text(text)
            out_lines = [annotated, "", "", ""]
        for i, line in enumerate(out_lines):
            append_transcript(line)
            print(line)
        # blank line between utterances for readability
        append_transcript("")
        print("")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
