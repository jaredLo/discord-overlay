#!/usr/bin/env python3
"""
ChatGPT-powered vocabulary analysis for Japanese text.
Extracts vocabulary, kanji, and katakana with readings and English meanings.
"""

import os
import time
import json
import threading
from typing import List, Dict, Optional, Tuple, Any, Set
import re
import requests
from pathlib import Path
import sqlite3
import hashlib

try:
    from fugashi import Tagger as _FallbackTagger
    from pykakasi import kakasi as _kakasi
    _fb_tagger = _FallbackTagger()
    _fb_kks = _kakasi(); _fb_kks.setMode("J","H"); _fb_conv = _fb_kks.getConverter()
except Exception:
    _fb_tagger = None
    _fb_conv = None

# Load environment variables (similar to listener.py)
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
                k = k.strip()
                v = v.strip()
                # Remove comments from values
                if "#" in v:
                    v = v.split("#")[0].strip()
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

_load_env_file()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano-2025-04-14")
USE_GPT_VOCAB_ANALYSIS = os.getenv("USE_GPT_VOCAB_ANALYSIS", "true").lower() in {"1", "true", "yes", "y"}
GPT_VOCAB_RATE_LIMIT_MS = int(os.getenv("GPT_VOCAB_RATE_LIMIT_MS", "1000"))
GPT_VOCAB_BATCH_SIZE = int(os.getenv("GPT_VOCAB_BATCH_SIZE", "8"))
GPT_VOCAB_WINDOW = int(os.getenv("GPT_VOCAB_WINDOW", "1000"))

# Caching
_CACHE_DB = "vocab_cache.sqlite"
_cache_lock = threading.Lock()

def _init_cache_db():
    """Initialize the vocabulary cache database."""
    try:
        conn = sqlite3.connect(_CACHE_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gpt_vocab (
                text_hash TEXT PRIMARY KEY,
                response TEXT,
                timestamp REAL
            )
        """)
        conn.commit()
        conn.close()
    except Exception:
        pass

def _sha1(s: str) -> str:
    """Generate SHA1 hash for caching."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _get_cached_analysis(text: str) -> Optional[Dict]:
    """Get cached vocabulary analysis."""
    if not USE_GPT_VOCAB_ANALYSIS:
        return None
    
    try:
        text_hash = _sha1(text)
        with _cache_lock:
            conn = sqlite3.connect(_CACHE_DB)
            row = conn.execute("SELECT response FROM gpt_vocab WHERE text_hash=?", (text_hash,)).fetchone()
            conn.close()
            
            if row:
                return json.loads(row[0])
    except Exception:
        pass
    return None

def _cache_analysis(text: str, response: Dict):
    """Cache vocabulary analysis."""
    if not USE_GPT_VOCAB_ANALYSIS:
        return
    
    try:
        text_hash = _sha1(text)
        response_json = json.dumps(response, ensure_ascii=False)
        
        with _cache_lock:
            conn = sqlite3.connect(_CACHE_DB)
            conn.execute(
                "INSERT OR REPLACE INTO gpt_vocab (text_hash, response, timestamp) VALUES (?, ?, ?)",
                (text_hash, response_json, time.time())
            )
            conn.commit()
            conn.close()
    except Exception:
        pass

def _to_hira_fb(s: str) -> str:
    if _fb_conv is None:
        return s
    try:
        return "".join(p["hira"] for p in _fb_conv.convert(s))
    except Exception:
        return s

def _fallback_basic_vocab(text: str) -> Dict[str, List[Dict]]:
    """Simple fallback analysis using fugashi if ChatGPT is unavailable or returns empty."""
    if _fb_tagger is None:
        return {"vocabulary": [], "kanji_only": [], "katakana_words": []}
    entries: List[Dict] = []
    try:
        for token in _fb_tagger(text):
            pos = getattr(token.feature, "pos1", "")
            if pos not in ("名詞", "動詞", "形容詞"):
                continue
            reading = getattr(token.feature, "pron", None) or getattr(token.feature, "kana", None) or ""
            entries.append({
                "surface": token.surface,
                "reading_hiragana": _to_hira_fb(reading),
                "meaning_en": "",
                "word_type": pos,
            })
    except Exception:
        return {"vocabulary": [], "kanji_only": [], "katakana_words": []}
    return {"vocabulary": entries, "kanji_only": [], "katakana_words": []}

# Rate limiting
_last_request_time = {"t": 0.0}
_rate_limit_lock = threading.Lock()

def _rate_limit():
    """Apply rate limiting to ChatGPT requests."""
    with _rate_limit_lock:
        elapsed = (time.time() - _last_request_time["t"]) * 1000
        if elapsed < GPT_VOCAB_RATE_LIMIT_MS:
            sleep_time = (GPT_VOCAB_RATE_LIMIT_MS - elapsed) / 1000
            time.sleep(sleep_time)
        _last_request_time["t"] = time.time()

def _chatgpt_vocab_request(japanese_text: str) -> Optional[Dict]:
    """Make a ChatGPT request for vocabulary analysis."""
    if not USE_GPT_VOCAB_ANALYSIS or not OPENAI_API_KEY:
        return None
    
    url = os.getenv("OPENAI_CHAT_URL", "https://api.openai.com/v1/chat/completions")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    system_prompt = """You are a Japanese vocabulary analyzer. Given Japanese text, extract vocabulary words and analyze them.

Return STRICT JSON only with this structure:
{
  "vocabulary": [
    {
      "surface": "Japanese word",
      "reading_hiragana": "hiragana reading",
      "reading_katakana": "katakana reading (if applicable)",
      "meaning_en": "English meaning",
      "word_type": "noun/verb/adjective/etc",
      "kanji_breakdown": [
        {"kanji": "漢", "reading": "かん", "meaning": "Chinese character"}
      ],
      "usage_notes": "brief usage explanation",
      "examples": ["example sentence 1", "example sentence 2"]
    }
  ],
  "kanji_only": [
    {
      "kanji": "漢",
      "readings_on": ["カン", "ガン"],
      "readings_kun": ["から"],
      "meaning_en": "Chinese character",
      "stroke_count": 13
    }
  ],
  "katakana_words": [
    {
      "surface": "カタカナ",
      "reading_hiragana": "かたかな", 
      "meaning_en": "katakana",
      "origin": "native/foreign"
    }
  ]
}

Identify nouns, verbs, adjectives, and notable expressions. Include both common and advanced terms and do not omit polite or infrequent words. Exclude particles and conjunctions."""

    user_content = f"Analyze this Japanese text for vocabulary:\n\n{japanese_text}"
    
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    
    try:
        _rate_limit()
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        return json.loads(content)
    except Exception as e:
        print(f"ChatGPT vocab analysis error: {e}")
        return None

class VocabularyAnalyzer:
    """Main vocabulary analyzer using ChatGPT."""

    def __init__(self):
        self.session = requests.Session()
        _init_cache_db()

    def analyze_text(self, japanese_text: str) -> Optional[Dict[str, List[Dict]]]:
        """
        Analyze Japanese text and return vocabulary analysis.

        Returns:
            Dict with keys: 'vocabulary', 'kanji_only', 'katakana_words'
            or None if the request failed.
        """
        if not japanese_text.strip():
            return {"vocabulary": [], "kanji_only": [], "katakana_words": []}

        # Check cache first
        cached = _get_cached_analysis(japanese_text)
        if cached:
            return cached

        # Make ChatGPT request
        result = _chatgpt_vocab_request(japanese_text)
        if result is None:
            return None

        # Normalize response structure
        normalized = {
            "vocabulary": result.get("vocabulary", []),
            "kanji_only": result.get("kanji_only", []),
            "katakana_words": result.get("katakana_words", [])
        }

        # Cache the result
        _cache_analysis(japanese_text, normalized)

        return normalized
    
    def analyze_batch(self, text_segments: List[str]) -> List[Optional[Dict[str, List[Dict]]]]:
        """
        Analyze multiple text segments in batch.

        Args:
            text_segments: List of Japanese text segments to analyze

        Returns:
            List of analysis results, one per segment (None on failure)
        """
        results: List[Optional[Dict[str, List[Dict]]]] = []

        # Process in batches to respect rate limits
        batch_size = GPT_VOCAB_BATCH_SIZE
        for i in range(0, len(text_segments), batch_size):
            batch = text_segments[i:i + batch_size]

            for text in batch:
                result = self.analyze_text(text)
                results.append(result)

                # Rate limit between requests in batch
                if len(batch) > 1 and text != batch[-1]:
                    time.sleep(GPT_VOCAB_RATE_LIMIT_MS / 1000 / 2)

        return results

def extract_japanese_from_transcript(transcript_text: str, max_segment_chars: int = 200) -> List[str]:
    """
    Extract Japanese sentences/segments from transcript text.
    Filters out vocab lines and English translations and splits long text into
    smaller chunks so more vocabulary can be captured.
    """
    if not transcript_text:
        return []

    lines = transcript_text.strip().split('\n')
    japanese_segments: List[str] = []

    # Japanese character pattern
    jp_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF々〆ヵヶ]')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip vocab lines
        if line.startswith('Vocab:') or '📚' in line:
            continue

        # Skip lines that look like English translations (start with lowercase or common EN words)
        if re.match(r'^[a-z]', line) or line.startswith(('The ', 'A ', 'An ', 'I ', 'You ', 'He ', 'She ', 'It ', 'We ', 'They ')):
            continue

        # Remove HTML tags
        clean_line = re.sub(r'<[^>]+>', '', line)

        # Check if line contains Japanese characters
        if not jp_pattern.search(clean_line):
            continue

        # Split into sentences and chunk by max_segment_chars
        sentences = re.split(r'(?<=[。！？!?])', clean_line)
        buf = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            buf += sent
            if len(buf) >= max_segment_chars or buf.endswith(('。', '！', '!', '？', '?')):
                japanese_segments.append(buf.strip())
                buf = ""
        if buf:
            japanese_segments.append(buf.strip())

    return japanese_segments

# Initialize global analyzer instance
_analyzer = None
_analyzed_hashes: Set[str] = set()

# Aggregated results across processed segments
_vocab_counts: Dict[str, int] = {}
_aggregated_vocab: Dict[str, Dict] = {}
_aggregated_kanji: Dict[str, Dict] = {}
_aggregated_katakana: Dict[str, Dict] = {}

def get_analyzer() -> VocabularyAnalyzer:
    """Get the global vocabulary analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = VocabularyAnalyzer()
    return _analyzer

# Convenience functions for integration
def analyze_transcript_vocab(transcript_text: str) -> Dict[str, List[Dict]]:
    """
    Analyze vocabulary from transcript text.

    Args:
        transcript_text: Raw transcript text containing Japanese

    Returns:
        Aggregated vocabulary analysis across all processed segments. Results
        accumulate over time and include usage counts for each vocabulary word.
    """
    if not USE_GPT_VOCAB_ANALYSIS:
        return {"vocabulary": [], "kanji_only": [], "katakana_words": [], "vocab_counts": {}}

    window_text = transcript_text[-GPT_VOCAB_WINDOW:] if GPT_VOCAB_WINDOW > 0 else transcript_text
    japanese_segments = extract_japanese_from_transcript(window_text)
    if not japanese_segments:
        return {
            "vocabulary": list(_aggregated_vocab.values()),
            "kanji_only": list(_aggregated_kanji.values()),
            "katakana_words": list(_aggregated_katakana.values()),
            "vocab_counts": dict(_vocab_counts),
        }

    analyzer = get_analyzer()

    for seg in japanese_segments:
        seg_hash = _sha1(seg)
        if seg_hash in _analyzed_hashes:
            continue

        result = _get_cached_analysis(seg)
        if result is None:
            result = analyzer.analyze_text(seg)
        if not result or not result.get("vocabulary"):
            result = _fallback_basic_vocab(seg)
        if not result or not result.get("vocabulary"):
            continue

        # Aggregate vocabulary entries and counts
        for vocab in result.get("vocabulary", []):
            surf = vocab.get("surface")
            if not surf:
                continue
            _vocab_counts[surf] = _vocab_counts.get(surf, 0) + 1
            if surf not in _aggregated_vocab:
                _aggregated_vocab[surf] = vocab

        for kanji in result.get("kanji_only", []):
            kan = kanji.get("kanji")
            if kan and kan not in _aggregated_kanji:
                _aggregated_kanji[kan] = kanji

        for kata in result.get("katakana_words", []):
            surf = kata.get("surface")
            if surf and surf not in _aggregated_katakana:
                _aggregated_katakana[surf] = kata

        _analyzed_hashes.add(seg_hash)

    return {
        "vocabulary": list(_aggregated_vocab.values()),
        "kanji_only": list(_aggregated_kanji.values()),
        "katakana_words": list(_aggregated_katakana.values()),
        "vocab_counts": dict(_vocab_counts),
    }

if __name__ == "__main__":
    sample = "聞いてないよ!分かったんじゃんかい! 負け!"
    print("Testing ChatGPT vocabulary analyzer...")
    result = analyze_transcript_vocab(sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))
