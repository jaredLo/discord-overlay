"""ASR via OpenAI Whisper API.

Receives raw PCM bytes, wraps in WAV container, sends to Whisper.
"""

from __future__ import annotations

import io
import wave
import struct
import numpy as np
import requests
from typing import Optional

from server.config import OPENAI_API_KEY, WHISPER_MODEL, OPENAI_AUDIO_URL, SAMPLE_RATE

_HTTP = requests.Session()


def _wav_bytes_from_pcm(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM int16 bytes in a WAV container.

    Whisper API rejects raw PCM — must have proper RIFF/fmt/data headers.
    Size fields are computed from actual byte count, not templated.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def transcribe(audio_pcm: bytes, sample_rate: int = SAMPLE_RATE) -> Optional[str]:
    """Transcribe PCM int16 audio bytes via OpenAI Whisper API.

    Args:
        audio_pcm: Raw PCM int16 LE bytes (mono, 16kHz)
        sample_rate: Sample rate of the audio

    Returns:
        Transcribed Japanese text, or None on failure.
    """
    if not OPENAI_API_KEY:
        return None

    wav_bytes = _wav_bytes_from_pcm(audio_pcm, sample_rate)

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {
        "model": WHISPER_MODEL,
        "language": "ja",
        "temperature": "0.0",
        "response_format": "json",
    }

    try:
        r = _HTTP.post(OPENAI_AUDIO_URL, headers=headers, files=files, data=data, timeout=30)
        r.raise_for_status()
        j = r.json()
        return (j.get("text") or "").strip() or None
    except Exception:
        return None
