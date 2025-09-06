"""
Local JP semantic suggestions provider.

Implement similar_words(bases, top_k) to return a list of single-word
suggestions semantically related to the given bases.

Each item should be a dict: { 'ja': str, 'read': str, 'en': str }

This default implementation uses Jisho see_also as a lightweight proxy,
and fills reading with fugashi/pykakasi. Replace with your internal
embedding/knowledge-based system when available.
"""
from typing import List, Dict, Optional
import re
import sqlite3, json

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from fugashi import Tagger
    from pykakasi import kakasi
    _tagger = Tagger()
    _kks = kakasi(); _kks.setMode("J","H"); _conv = _kks.getConverter()
except Exception:  # pragma: no cover
    _tagger = None
    _conv = None

_KANJI = re.compile(r"[一-龯々〆ヵヶ]")
_KATA  = re.compile(r"[ァ-ヴ]")

def _to_hira(s: str) -> str:
    if _conv is None: return s
    try: return "".join(p["hira"] for p in _conv.convert(s))
    except Exception: return s

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

def _has_kana_or_kanji(s: str) -> bool:
    return bool(_KANJI.search(s) or _KATA.search(s))

def _jisho_best_gloss(word: str) -> str:
    if not requests: return ''
    try:
        url = f"https://jisho.org/api/v1/search/words?keyword={requests.utils.quote(word)}"
        r = requests.get(url, timeout=3)
        j = r.json(); data = j.get('data') or []
        if not data: return ''
        senses = data[0].get('senses') or []
        for s in senses:
            ed = s.get('english_definitions') or []
            if ed: return ed[0]
    except Exception:
        return ''
    return ''

def _similar_words_jisho(base: str) -> List[str]:
    if not requests: return []
    try:
        url = f"https://jisho.org/api/v1/search/words?keyword={requests.utils.quote(base)}"
        r = requests.get(url, timeout=3)
        j = r.json(); data = j.get('data') or []
        if not data: return []
        out: List[str] = []
        for sense in (data[0].get('senses') or []):
            for sa in (sense.get('see_also') or []):
                w = str(sa).split(' ')[0].strip()
                if _has_kana_or_kanji(w): out.append(w)
        return out
    except Exception:
        return []

def _load_cached_jisho(word: str) -> Optional[dict]:
    try:
        conn = sqlite3.connect('dict_cache.sqlite')
        try:
            row = conn.execute('SELECT v FROM jisho WHERE k=?', (word,)).fetchone()
            if not row: return None
            return json.loads(row[0])
        finally:
            conn.close()
    except Exception:
        return None

def _candidate_gloss(candidate: str) -> Optional[Dict[str,str]]:
    # Try cache first for reading + gloss
    j = _load_cached_jisho(candidate)
    ja = candidate
    rd = ''
    en = ''
    try:
        if j:
            data = j.get('data') or []
            if data:
                jp = data[0].get('japanese') or []
                if jp:
                    rd = jp[0].get('reading') or ''
                senses = data[0].get('senses') or []
                for s in senses:
                    ed = s.get('english_definitions') or []
                    if ed:
                        en = ed[0]
                        break
    except Exception:
        pass
    if not rd:
        rd = _reading_for(candidate)
    if not en:
        en = _jisho_best_gloss(candidate)
    # Filter junk: ensure english meaningful; katakana len >= 3; english >= 3 letters
    if _KATA.search(candidate):
        if len(candidate) < 3:
            return None
        if not en or len(en.strip()) < 3:
            return None
    if not en:
        return None
    return { 'ja': ja, 'read': rd, 'en': en }

def similar_words(bases: List[str], top_k: int = 30) -> List[Dict[str, str]]:
    """
    Return up to top_k items like { 'ja', 'read', 'en' } semantically similar
    to bases. Uses cached Jisho see_also relations when available, with
    strict filters to ensure meaningful single-word suggestions.
    """
    out: List[Dict[str,str]] = []
    seen = set()
    for b in bases:
        if len(out) >= top_k: break
        # Prefer cached entry for base, then see_also
        sims: List[str] = []
        j = _load_cached_jisho(b)
        try:
            if j:
                data = j.get('data') or []
                if data:
                    for sense in (data[0].get('senses') or []):
                        for sa in (sense.get('see_also') or []):
                            w = str(sa).split(' ')[0].strip()
                            if _has_kana_or_kanji(w):
                                sims.append(w)
        except Exception:
            sims = []
        if not sims:
            sims = _similar_words_jisho(b)
        for w in sims:
            if len(out) >= top_k: break
            if w in seen: continue
            seen.add(w)
            if len(w) > 14: continue
            if not _has_kana_or_kanji(w): continue
            cand = _candidate_gloss(w)
            if not cand: continue
            out.append(cand)
    return out[:top_k]
