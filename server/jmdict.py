"""JMdict SQLite lookup — replaces Jisho API calls.

Expects a pre-built DB from scriptin/jmdict-simplified with schema:
  entries(id, kanji, reading, sense)
  — kanji/reading are JSON arrays, sense is JSON array of {gloss, pos, ...}

Falls back gracefully if DB is not present (returns empty results).
"""

from __future__ import annotations

import json
import sqlite3
import functools
from pathlib import Path
from typing import Optional

from server.config import JMDICT_PATH

_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> Optional[sqlite3.Connection]:
    global _conn
    if _conn is not None:
        return _conn
    if not JMDICT_PATH.exists():
        return None
    try:
        _conn = sqlite3.connect(str(JMDICT_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        return _conn
    except Exception:
        return None


@functools.lru_cache(maxsize=4096)
def lookup(word: str) -> Optional[dict]:
    """Look up a word in JMdict. Returns first matching entry or None.

    Return shape: {
        "kanji": ["漢字", ...],
        "reading": ["かんじ", ...],
        "senses": [{"glosses": ["Chinese characters", ...], "pos": ["Noun"], ...}]
    }
    """
    conn = _get_conn()
    if conn is None:
        return None

    try:
        # Try exact kanji match first
        row = conn.execute(
            "SELECT * FROM entries WHERE kanji LIKE ? LIMIT 1",
            (f'%"{word}"%',),
        ).fetchone()

        # Fallback: reading match
        if row is None:
            row = conn.execute(
                "SELECT * FROM entries WHERE reading LIKE ? LIMIT 1",
                (f'%"{word}"%',),
            ).fetchone()

        if row is None:
            return None

        return {
            "kanji": json.loads(row["kanji"]) if row["kanji"] else [],
            "reading": json.loads(row["reading"]) if row["reading"] else [],
            "senses": json.loads(row["sense"]) if row["sense"] else [],
        }
    except Exception:
        return None


def best_gloss(word: str) -> str:
    """Get the best English gloss for a Japanese word."""
    entry = lookup(word)
    if not entry:
        return ""
    for sense in entry.get("senses", []):
        glosses = sense.get("glosses") or sense.get("gloss") or []
        if isinstance(glosses, list) and glosses:
            # glosses may be strings or dicts with "text" key
            g = glosses[0]
            if isinstance(g, dict):
                return g.get("text", "")
            return str(g)
        if isinstance(glosses, str):
            return glosses
    return ""


def has_entry(word: str) -> bool:
    """Check if a word exists in JMdict."""
    return lookup(word) is not None
