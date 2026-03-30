"""Centralized configuration from environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path):
    try:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if "#" in v:
                v = v.split("#")[0].strip()
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


_load_env_file(ROOT / ".env")


def _bool(key: str, default: str = "false") -> bool:
    return os.getenv(key, default).lower() in {"1", "true", "yes", "y"}


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


# --- ASR ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "gpt-4o-mini-transcribe")
OPENAI_AUDIO_URL = os.getenv(
    "OPENAI_AUDIO_URL", "https://api.openai.com/v1/audio/transcriptions"
)

# --- LLM ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
OPENAI_CHAT_URL = os.getenv(
    "OPENAI_CHAT_URL", "https://api.openai.com/v1/chat/completions"
)
USE_GPT_FORMATTER = _bool("USE_GPT_FORMATTER", "true") and bool(OPENAI_API_KEY)
GPT_MAX_VOCAB = _int("GPT_MAX_VOCAB", 10)
GPT_MAX_CHARS = _int("GPT_MAX_CHARS", 280)
GPT_RATE_LIMIT_MS = _int("GPT_RATE_LIMIT_MS", 600)
GPT_CLEAN_REPEATS = _bool("GPT_CLEAN_REPEATS", "true")
GPT_SKIP_ON_REPEAT = _bool("GPT_SKIP_ON_REPEAT", "true")
GPT_REPEAT_SKIP_THRESHOLD = float(os.getenv("GPT_REPEAT_SKIP_THRESHOLD", "2.0"))
GPT_COMPRESS_ELONG = _bool("GPT_COMPRESS_ELONGATIONS")

# --- LLM grammar suggestions ---
LLM_GRAMMAR_TIMEOUT_S = _int("LLM_GRAMMAR_TIMEOUT_S", 10)
LLM_CONTEXT_UTTERANCES = _int("LLM_CONTEXT_UTTERANCES", 5)
SUGGESTION_QUEUE_CAP = _int("SUGGESTION_QUEUE_CAP", 5)

# --- Auth ---
API_KEY = os.getenv("KOTOFLOAT_API_KEY", "")

# --- Session ---
SESSION_ORPHAN_TIMEOUT_S = _int("SESSION_ORPHAN_TIMEOUT_S", 90)
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))

# --- JMdict ---
JMDICT_PATH = Path(
    os.getenv("JMDICT_PATH", str(ROOT / "static" / "jmdict.db"))
)

# --- Server ---
SAMPLE_RATE = 16000
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _int("PORT", 8201)
