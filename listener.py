#!/usr/bin/env python3
# listener.py — Accuracy-first live JP ASR with inline annotations
# Strategy: end-of-utterance chunks (silence or max length) → beam search (base/int8) → annotate tokens
# Output example: みんな、小学校(しょうがっこう、elementary school)の時(とき、time)以来(いらい、since)だね。

import time, queue, threading, sys, re
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from pykakasi import kakasi
from jamdict import Jamdict
from fugashi import Tagger
from typing import Union
from deep_translator import GoogleTranslator

# ================= Config (accuracy-first) =================
TARGET_DEVICE_SUBSTR = "BlackHole"
SAMPLE_RATE = 16000
FRAME_MS = 20          # WebRTC VAD requires 10/20/30ms
CHANNELS_IN = 2

# Utterance chunking (accuracy > latency)
VAD_LEVEL = 2
SILENCE_HANG_MS = 320   # end-of-utterance when this much silence passes
MIN_CHUNK_SEC = 1.4     # avoid decoding tiny blips
MAX_CHUNK_SEC = 3.2     # flush even if continuous speech
TRANSCRIPT_OUT = "transcript.txt"
MODEL_SIZE = "base"
COMPUTE_TYPE = "int8"   # on M1/8GB this is the safe choice

# ================= UniDic tagger (full if available) =================
def make_tagger():
    try:
        import unidic
        return Tagger('-d ' + unidic.DICDIR)  # full UniDic
    except Exception:
        return Tagger()  # fallback (likely unidic-lite)
tagger = make_tagger()
translator = GoogleTranslator(source="ja", target="en")

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

def transcribe_iter(chunks_iter, model):
    for audio in chunks_iter:
        audio = preprocess(audio)
        segs, _ = model.transcribe(
            audio,
            language="ja",
            beam_size=5,                 # steadier than 1
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
_jam = Jamdict()

KANJI_RE = re.compile(r"[一-龯々〆ヵヶ]")
KATA_LETTER_RE = re.compile(r"[ァ-ヴ]")  # excludes 'ー'
NUM_RE = re.compile(r"\d+")
KANJI_NUM_RE = re.compile(r"[一二三四五六七八九十百千]+")
PUNCT_POS = {"記号"}
SKIP_POS = {"助詞","助動詞"}  # keep interjections; skip particles/aux

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
    try:
        r = _jam.lookup(word)
        cands = []
        for e in r.entries:
            for s in e.senses:
                pos_tags = set(getattr(s, "pos", []) or [])
                for gl in getattr(s, "gloss", []):
                    txt = getattr(gl, "text", None) or str(gl)
                    lang = getattr(gl, "lang", None)
                    if lang is not None and not str(lang).lower().startswith(("en","eng")):
                        continue
                    score = 0
                    if "ctr" in pos_tags: score += 3
                    if "n" in pos_tags:   score += 2
                    if "suf" in pos_tags: score -= 2
                    if prefer_counter and "ctr" in pos_tags: score += 2
                    cands.append((score, txt))
        if not cands: return None
        cands.sort(key=lambda x: x[0], reverse=True)
        return cands[0][1]
    except Exception:
        return None

def jam_has_entry(s: str) -> bool:
    try:
        return bool(Jamdict().lookup(s).entries)
    except Exception:
        return False

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
                if KANJI_RE.search(surf) and jam_has_entry(surf):
                    reading = compound_reading(seg)
                    gloss = best_gloss(surf) or ""
                    anno = f"{surf}({reading}" + (f"、{gloss})" if gloss else ")")
                    return anno, L
    return None

def annotate_text(text: str) -> str:
    toks = list(tagger(text))
    out = []; i = 0
    while i < len(toks):
        m = toks[i]
        surf = m.surface
        pos1 = getattr(m.feature, "pos1", "")

        # Compound merge first
        merged = maybe_merge_compound(toks, i)
        if merged:
            anno, L = merged
            out.append(anno); i += L; continue

        # Skip punctuation/particles/aux
        if pos1 in PUNCT_POS or pos1 in SKIP_POS:
            out.append(surf); i += 1; continue

        # Counter pattern: [number][counter] or previous number + counter
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

        # Annotate only if kanji present or has katakana letter (not just 'ー')
        annotatable = bool(KANJI_RE.search(surf) or has_kata_letter(surf))
        if not annotatable:
            out.append(surf); i += 1; continue

        # Reading: fugashi > kakasi
        k = get_reading(m)
        reading = to_hira(k) if k else to_hira(surf)

        # 人 defaults to 'ひと' unless used with a number (handled above)
        lemma = getattr(m.feature, "lemma", None) or surf
        prefer_counter = False
        if surf == "人" and not (i > 0 and to_int_number(toks[i-1].surface)):
            prefer_counter = False

        gloss = best_gloss(lemma, prefer_counter=prefer_counter) or best_gloss(surf, prefer_counter=prefer_counter)
        out.append(f"{surf}({reading}" + (f"、{gloss})" if gloss else ")"))
        i += 1
    return "".join(out)
 
 
def append_transcript(text: str):
     with open(TRANSCRIPT_OUT, "a", encoding="utf-8") as f:
         f.write(text + "\n")


def translate_text(text: str) -> str:
    """Translate Japanese text to English using Google Translate."""
    try:
        return translator.translate(text)
    except Exception:
        return ""


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
    for text in transcribe_iter(utterance_chunks(chunks()), model):
        annotated = annotate_text(text)
        translation = translate_text(text)
        append_transcript(annotated)
        if translation:
            append_transcript(translation)
        print(annotated)
        if translation:
            print(translation)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass