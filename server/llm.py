"""LLM integration — GPT formatting and grammar suggestions.

Uses OpenAI chat completions for:
1. Transcript formatting (JP verbatim + EN translation + suggestions)
2. Grammar suggestions (pattern/template/words)
"""

from __future__ import annotations

import json
import re
import time
import requests
from typing import Optional, Dict, List

from server.config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_CHAT_URL,
    USE_GPT_FORMATTER, GPT_MAX_VOCAB, GPT_MAX_CHARS,
    GPT_CLEAN_REPEATS, GPT_SKIP_ON_REPEAT, GPT_REPEAT_SKIP_THRESHOLD,
    GPT_COMPRESS_ELONG, LLM_GRAMMAR_TIMEOUT_S, LLM_CONTEXT_UTTERANCES,
)

_HTTP = requests.Session()


# --- Text sanitization ---

def _collapse_long_runs(s: str, max_run: int = 12) -> str:
    return re.sub(r"(.)\1{" + str(max_run) + r",}", lambda m: m.group(1) * max_run, s)


def _collapse_repeated_phrases(s: str, min_len: int = 2, max_len: int = 8, keep: int = 3) -> str:
    out = s
    try:
        for n in range(max_len, min_len - 1, -1):
            pattern = re.compile(r"((?:.{" + str(n) + r"}))\1{3,}")
            out = pattern.sub(lambda m: m.group(1) * keep, out)
    except Exception:
        return s
    return out


def _repetition_score(s: str) -> float:
    if not s:
        return 0.0
    cleaned = _collapse_repeated_phrases(_collapse_long_runs(s), 2, 8, 1)
    return len(s) / max(1, len(cleaned))


def sanitize_for_gpt(s: str) -> Optional[str]:
    """Clean text before sending to GPT. Returns None if too repetitive."""
    if not s:
        return s
    t = s
    if GPT_CLEAN_REPEATS:
        t = _collapse_long_runs(t)
        t = _collapse_repeated_phrases(t)
    if len(t) > GPT_MAX_CHARS:
        t = t[:GPT_MAX_CHARS]
    if GPT_SKIP_ON_REPEAT and _repetition_score(s) >= GPT_REPEAT_SKIP_THRESHOLD:
        return None
    return t


# --- GPT transcript formatting ---

def format_transcript(jp_text: str) -> Optional[Dict]:
    """Call GPT to format a Japanese utterance.

    Returns dict with keys: jp_html, en_html, suggestions
    or None on failure/skip.
    """
    if not USE_GPT_FORMATTER or not OPENAI_API_KEY:
        return None

    payload_text = sanitize_for_gpt(jp_text)
    if payload_text is None:
        return None

    system = (
        "You are a precise Japanese translator. "
        "Given ONE Japanese utterance, respond with STRICT JSON only: { jp_html, en_html, suggestions }. "
        "jp_html: EXACT verbatim of the input (no paraphrase/normalization). Preserve all characters and whitespace. No added readings, no extra symbols. "
        "en_html: Faithful, natural English translation of the entire utterance. "
        "suggestions: 3-8 short, on-topic words or brief replies in Japanese (single terms/very short phrases), each as { jp, reading_kana, en, hint }. 'hint' gives a brief note on how to build or use the word or phrase (particles, conjugation, etc.). Keep concise; avoid duplicates. "
        "Constraints: VALID JSON only (no markdown). If no Japanese present, suggestions may be empty."
    )
    user = {
        "utterance": payload_text,
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
        r = _HTTP.post(
            OPENAI_CHAT_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=60,
        )
        r.raise_for_status()
        j = r.json()
        content = j["choices"][0]["message"]["content"].strip()
        return json.loads(content)
    except Exception:
        return None


# --- Grammar suggestions ---

_GRAMMAR_SYSTEM = (
    "You are a Japanese grammar tutor. Given recent conversation utterances, "
    "suggest ONE grammar pattern the learner could practice. "
    "Respond with STRICT JSON: { pattern, template, words, hint }. "
    "pattern: the grammar point name (e.g. 'ても', '〜たら'). "
    "template: a fill-in-the-blank sentence using the pattern. "
    "words: 2-4 words the learner can use to fill the template, each as { jp, reading, en }. "
    "hint: one-line usage note. "
    "VALID JSON only, no markdown."
)


def grammar_suggestion(recent_utterances: List[str]) -> Optional[Dict]:
    """Generate a grammar suggestion based on recent conversation.

    Uses structured output to enforce JSON schema.
    """
    if not OPENAI_API_KEY:
        return None

    context = recent_utterances[-LLM_CONTEXT_UTTERANCES:]
    if not context:
        return None

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _GRAMMAR_SYSTEM},
            {"role": "user", "content": json.dumps(
                {"utterances": context}, ensure_ascii=False
            )},
        ],
    }

    try:
        r = _HTTP.post(
            OPENAI_CHAT_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=LLM_GRAMMAR_TIMEOUT_S,
        )
        r.raise_for_status()
        j = r.json()
        content = j["choices"][0]["message"]["content"].strip()
        result = json.loads(content)
        # Validate required fields
        if not all(k in result for k in ("pattern", "template", "words")):
            return None
        return result
    except Exception:
        return None
