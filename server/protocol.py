"""WebSocket protocol message types and builders.

Client → Server (text frames):
  start_session: { type, session_id?, audio_format }
  end_session:   { type }

Client → Server (binary frames):
  Raw PCM audio bytes (16kHz, mono, int16)

Server → Client (text frames):
  transcript:  { type, seq, text, annotated, reading }
  vocabulary:  { type, seq, mode, items }
  suggestion:  { type, seq, items }
  error:       { type, code, message }
  session_ack: { type, session_id }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


# --- Inbound message parsing ---

def parse_client_message(raw: str) -> dict:
    """Parse a JSON text frame from the client."""
    msg = json.loads(raw)
    if not isinstance(msg, dict):
        raise ValueError("Expected JSON object, got " + type(msg).__name__)
    if "type" not in msg:
        raise ValueError("Missing 'type' field")
    return msg


# --- Outbound message builders ---

def session_ack(session_id: str) -> str:
    return json.dumps({"type": "session_ack", "session_id": session_id})


def transcript_msg(seq: int, text: str, annotated: str, reading: str) -> str:
    return json.dumps({
        "type": "transcript",
        "seq": seq,
        "text": text,
        "annotated": annotated,
        "reading": reading,
    })


def vocabulary_msg(seq: int, mode: str, items: list[dict]) -> str:
    """mode is 'snapshot' (first after start_session) or 'delta'."""
    return json.dumps({
        "type": "vocabulary",
        "seq": seq,
        "mode": mode,
        "items": items,
    })


def suggestion_msg(seq: int, items: list[dict]) -> str:
    return json.dumps({
        "type": "suggestion",
        "seq": seq,
        "items": items,
    })


def transcript_update_msg(seq: int, annotated: str, en: str) -> str:
    """GPT-enhanced update for an already-sent transcript."""
    return json.dumps({
        "type": "transcript_update",
        "seq": seq,
        "annotated": annotated,
        "en": en,
    })


def error_msg(code: str, message: str) -> str:
    return json.dumps({"type": "error", "code": code, "message": message})


# --- Audio format validation ---

@dataclass
class AudioFormat:
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "pcm_s16le"

    @classmethod
    def from_dict(cls, d: dict) -> "AudioFormat":
        return cls(
            sample_rate=d.get("sample_rate", 16000),
            channels=d.get("channels", 1),
            encoding=d.get("encoding", "pcm_s16le"),
        )

    def validate(self) -> str | None:
        """Return error message if invalid, else None."""
        if self.sample_rate != 16000:
            return f"Unsupported sample_rate {self.sample_rate}, must be 16000"
        if self.channels != 1:
            return f"Unsupported channels {self.channels}, must be 1"
        if self.encoding != "pcm_s16le":
            return f"Unsupported encoding {self.encoding}, must be pcm_s16le"
        return None
