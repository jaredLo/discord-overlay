"""Japanese NLP pipeline — morphological analysis, reading, annotation.

Extracted from listener.py. Uses fugashi + pykakasi for tokenization
and JMdict for dictionary lookups (replaces Jisho API).
"""

from __future__ import annotations

import re
import os
import html
from typing import Optional, Union, List, Dict, Tuple

from server import jmdict

# --- Tagger setup ---

try:
    from fugashi import Tagger
    from pykakasi import kakasi

    def _make_tagger():
        try:
            import unidic
            return Tagger("-d " + unidic.DICDIR)
        except Exception:
            return Tagger()

    tagger = _make_tagger()
    _kks = kakasi()
    _kks.setMode("J", "H")
    _conv = _kks.getConverter()
except Exception:
    tagger = None
    _conv = None

# --- Regex patterns ---

KANJI_RE = re.compile(r"[一-龯々〆ヵヶ]")
KATA_LETTER_RE = re.compile(r"[ァ-ヴ]")
NUM_RE = re.compile(r"\d+")
KANJI_NUM_RE = re.compile(r"[一二三四五六七八九十百千]+")
PUNCT_POS = {"記号"}
SKIP_POS = {"助詞", "助動詞"}

# --- Counter readings (1-10) ---

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
KANJI_NUM_MAP = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

# --- User dictionary ---

USER_DICT_CSV = os.getenv("USER_DICT_CSV", "user_dict.csv")
_user_overrides: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

def _load_user_dict():
    global _user_overrides
    import csv
    path = USER_DICT_CSV
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                surface = row[0].strip()
                reading = row[1].strip() if len(row) > 1 and row[1].strip() else None
                english = row[2].strip() if len(row) > 2 and row[2].strip() else None
                if surface:
                    _user_overrides[surface] = (reading, english)
    except Exception:
        pass

_load_user_dict()


# --- Helpers ---

def to_hira(s: str) -> str:
    if _conv is None:
        return s
    try:
        return "".join(p["hira"] for p in _conv.convert(s))
    except Exception:
        return s


def get_reading(m) -> Optional[str]:
    return getattr(m.feature, "pron", None) or getattr(m.feature, "kana", None)


def has_kata_letter(s: str) -> bool:
    return bool(KATA_LETTER_RE.search(s))


def reading_for(jp: str) -> str:
    """Get hiragana reading for a Japanese string."""
    if tagger is None:
        return to_hira(jp)
    try:
        toks = list(tagger(jp))
        if not toks:
            return to_hira(jp)
        kana = []
        for m in toks:
            k = get_reading(m)
            kana.append(k if k else m.surface)
        return to_hira("".join(kana))
    except Exception:
        return to_hira(jp)


def to_int_number(s: str) -> Optional[int]:
    m = NUM_RE.fullmatch(s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    if KANJI_NUM_RE.fullmatch(s):
        val = 0
        for ch in s:
            if ch == "十":
                val = (val or 1) * 10
            elif ch in KANJI_NUM_MAP:
                val += KANJI_NUM_MAP[ch]
            else:
                return None
        return val or None
    return None


def best_gloss(word: str, prefer_counter: bool = False) -> Optional[str]:
    """Get best English gloss — user dict first, then JMdict."""
    ov = _user_overrides.get(word)
    if ov and ov[1]:
        return ov[1]
    return jmdict.best_gloss(word) or None


def compound_reading(tokens) -> str:
    kana = []
    for m in tokens:
        k = get_reading(m)
        kana.append(k if k else m.surface)
    return to_hira("".join(kana))


def maybe_merge_compound(tokens, i, max_len=3):
    for L in (3, 2):
        if i + L <= len(tokens):
            seg = tokens[i:i+L]
            if all(getattr(t.feature, "pos1", "") == "名詞" for t in seg):
                surf = "".join(t.surface for t in seg)
                if KANJI_RE.search(surf) and jmdict.has_entry(surf):
                    ov = _user_overrides.get(surf)
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
    """Annotate Japanese text with readings and glosses.

    Returns string like: 小学校(しょうがっこう、elementary school)の時(とき、time)
    """
    if tagger is None:
        return text

    toks = list(tagger(text))
    out = []
    i = 0
    seen_lemmas: set = set()

    while i < len(toks):
        m = toks[i]
        surf = m.surface
        pos1 = getattr(m.feature, "pos1", "")
        lemma = getattr(m.feature, "lemma", None) or surf

        # Compound merge
        merged = maybe_merge_compound(toks, i)
        if merged:
            anno, L = merged
            out.append(anno)
            i += L
            continue

        # Skip punctuation/particles/aux
        if pos1 in PUNCT_POS or pos1 in SKIP_POS:
            out.append(surf)
            i += 1
            continue

        # Counter patterns
        if len(surf) >= 2 and surf[-1] in COUNTER_READ:
            n = to_int_number(surf[:-1])
            if n and 1 <= n <= 10:
                reading = COUNTER_READ[surf[-1]][n - 1]
                gloss = best_gloss(surf[-1], prefer_counter=True)
                out.append(f"{surf}({reading}" + (f"、{gloss})" if gloss else ")"))
                i += 1
                continue
        if surf in COUNTER_READ and i > 0:
            n = to_int_number(toks[i - 1].surface)
            if n and 1 <= n <= 10:
                reading = COUNTER_READ[surf][n - 1]
                gloss = best_gloss(surf, prefer_counter=True)
                out.append(f"{surf}({reading}" + (f"、{gloss})" if gloss else ")"))
                i += 1
                continue

        # Only annotate kanji/katakana tokens
        annotatable = bool(KANJI_RE.search(surf) or has_kata_letter(surf))
        if not annotatable:
            out.append(surf)
            i += 1
            continue

        # User overrides
        if surf in _user_overrides or lemma in _user_overrides:
            rd, en = _user_overrides.get(surf, _user_overrides.get(lemma))
            reading = rd if rd else (to_hira(get_reading(m)) if get_reading(m) else to_hira(surf))
            if en:
                out.append(f"{surf}({reading}、{en})")
            else:
                out.append(f"{surf}({reading})")
            i += 1
            continue

        # Reading
        k = get_reading(m)
        reading = to_hira(k) if k else to_hira(surf)

        # Gloss (de-duped per utterance)
        gloss = None
        if lemma not in seen_lemmas:
            gloss = best_gloss(lemma)
            seen_lemmas.add(lemma)
        if gloss is None:
            gloss = best_gloss(surf)

        out.append(f"{surf}({reading}" + (f"、{gloss})" if gloss else ")"))
        i += 1

    return "".join(out)


def extract_vocabulary(text: str, max_items: int = 10) -> List[Dict[str, str]]:
    """Extract vocabulary items from Japanese text.

    Returns list of {ja, reading, en} dicts.
    """
    if tagger is None:
        return []

    try:
        toks = list(tagger(text))
    except Exception:
        return []

    items: List[Dict[str, str]] = []
    seen: set = set()

    for m in toks:
        surf = m.surface
        pos1 = getattr(m.feature, "pos1", "")
        if pos1 not in {"名詞", "動詞", "形容詞"}:
            continue
        if not (KANJI_RE.search(surf) or has_kata_letter(surf)):
            continue
        if surf in seen:
            continue
        seen.add(surf)

        rd = get_reading(m) or surf
        rd = to_hira(rd)
        en = best_gloss(surf) or ""

        items.append({"ja": surf, "reading": rd, "en": en})
        if len(items) >= max_items:
            break

    return items
