"""KotoFloat Cloud Backend — FastAPI + WebSocket streaming.

Replaces file-based IPC with WebSocket protocol:

  - Client sends binary audio frames (PCM int16, 16kHz mono)
  - Server returns structured JSON: transcript, vocabulary, suggestions
  - Session state persisted to SQLite after each transcript event

Note: The old REST endpoints (/api/overlay/*) are intentionally removed.
The desktop client uses the local `make dev` server (old architecture).
This cloud backend serves the Android app exclusively via WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from server.config import SAMPLE_RATE, OPENAI_API_KEY
from server.auth import verify_api_key
from server.protocol import (
    parse_client_message,
    AudioFormat,
    session_ack,
    transcript_msg,
    transcript_update_msg,
    vocabulary_msg,
    suggestion_msg,
    error_msg,
)
from server.session import store, SessionState
from server import asr, nlp, llm


# Separate thread pools to prevent LLM background work from starving ASR.
# ASR is the fast path — must never queue behind slow GPT/grammar calls.
_asr_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr")
_llm_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="llm")

# Track active websocket per session for eviction on resume.
_active_ws: Dict[str, WebSocket] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _asr_executor.shutdown(wait=False, cancel_futures=True)
    _llm_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan)


# --- Health check ---

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "asr_configured": bool(OPENAI_API_KEY),
        "jmdict_available": nlp.tagger is not None,
    }


# --- WebSocket session handler ---

@app.websocket("/ws/session")
async def ws_session(ws: WebSocket):
    if not verify_api_key(ws):
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()

    session: SessionState | None = None
    audio_format: AudioFormat | None = None
    loop = asyncio.get_event_loop()
    background_tasks: Set[asyncio.Task] = set()

    async def _safe_send(text: str):
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_text(text)
        except Exception:
            pass

    def _cancel_background():
        for t in background_tasks:
            t.cancel()
        background_tasks.clear()

    try:
        while True:
            message = await ws.receive()

            # --- Binary frame: audio data ---
            if "bytes" in message and message["bytes"]:
                if session is None:
                    await _safe_send(error_msg("no_session", "Send start_session first"))
                    continue
                if audio_format is None:
                    await _safe_send(error_msg("no_format", "Audio format not set"))
                    continue

                pcm_bytes = message["bytes"]

                # ASR + NLP on dedicated pool (never blocked by LLM work).
                # Does NOT mutate session — mutations after eviction check.
                result = await loop.run_in_executor(
                    _asr_executor, _process_audio_core, pcm_bytes
                )

                # Check if we were evicted during processing (reconnect race).
                if _active_ws.get(session.session_id) is not ws:
                    break

                if result is None:
                    continue

                # Safe to mutate session now — we're still the active handler
                session.add_utterance(result["text"])
                seq = session.next_seq()

                # Send transcript with local NLP results IMMEDIATELY
                await _safe_send(transcript_msg(
                    seq=seq,
                    text=result["text"],
                    annotated=result["annotated"],
                    reading=result["reading"],
                ))

                # Send vocabulary delta
                if result["vocabulary"]:
                    session.add_vocabulary(result["vocabulary"])
                    await _safe_send(vocabulary_msg(
                        seq=seq,
                        mode="delta",
                        items=result["vocabulary"],
                    ))

                # Persist after transcript (don't wait for LLM)
                store.persist(session.session_id)

                # Fire LLM enhancement on separate pool (non-blocking).
                # Uses transcript_update message type — distinct from transcript.
                task = asyncio.create_task(
                    _enhance_with_llm(
                        session, seq, result["text"], ws, _safe_send, loop
                    )
                )
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)

                # Fire grammar suggestion on separate pool
                if len(session.utterances) >= 2:
                    utterances = list(session.utterances)
                    gtask = asyncio.create_task(
                        _run_grammar(
                            session, seq, ws, _safe_send, loop, utterances
                        )
                    )
                    background_tasks.add(gtask)
                    gtask.add_done_callback(background_tasks.discard)

            # --- Text frame: control message ---
            elif "text" in message and message["text"]:
                try:
                    msg = parse_client_message(message["text"])
                except (json.JSONDecodeError, ValueError) as e:
                    await _safe_send(error_msg("parse_error", str(e)))
                    continue

                msg_type = msg["type"]

                if msg_type == "start_session":
                    sid = msg.get("session_id")

                    # Validate audio format BEFORE evicting old session.
                    fmt_dict = msg.get("audio_format", {})
                    new_format = AudioFormat.from_dict(fmt_dict)
                    fmt_err = new_format.validate()
                    if fmt_err:
                        await _safe_send(error_msg("audio_format_mismatch", fmt_err))
                        await ws.close(code=4002, reason=fmt_err)
                        return

                    # Now safe to evict old websocket
                    if sid and sid in _active_ws:
                        old_ws = _active_ws.pop(sid, None)
                        if old_ws is not None:
                            try:
                                await old_ws.close(
                                    code=4003,
                                    reason="Session resumed on new connection",
                                )
                            except Exception:
                                pass

                    session = store.create_session(sid)
                    audio_format = new_format
                    _active_ws[session.session_id] = ws

                    await _safe_send(session_ack(session.session_id))

                    if session.vocabulary:
                        await _safe_send(vocabulary_msg(
                            seq=0,
                            mode="snapshot",
                            items=session.vocabulary,
                        ))

                elif msg_type == "end_session":
                    if session:
                        _cancel_background()
                        if _active_ws.get(session.session_id) is ws:
                            _active_ws.pop(session.session_id, None)
                        store.end_session(session.session_id)
                        session = None
                    audio_format = None

                elif msg_type == "ping":
                    await _safe_send(json.dumps({"type": "pong"}))

                else:
                    await _safe_send(error_msg(
                        "unknown_type", f"Unknown message type: {msg_type}"
                    ))

            # --- WebSocket disconnect ---
            elif message.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        _cancel_background()
        if session:
            if _active_ws.get(session.session_id) is ws:
                _active_ws.pop(session.session_id, None)
                await store.mark_disconnected(session.session_id)


def _process_audio_core(pcm_bytes: bytes) -> dict | None:
    """Process audio through ASR + NLP pipeline (fast path).

    Runs on _asr_executor. Does NOT mutate SessionState —
    mutations happen in the main loop after eviction check.
    """
    text = asr.transcribe(pcm_bytes)
    if not text or len(text.strip()) < 2:
        return None

    text = text.strip()

    annotated = nlp.annotate_text(text)
    reading = nlp.reading_for(text)
    vocabulary = nlp.extract_vocabulary(text)

    return {
        "text": text,
        "annotated": annotated,
        "reading": reading,
        "vocabulary": vocabulary,
    }


async def _enhance_with_llm(
    session: SessionState,
    seq: int,
    text: str,
    ws: WebSocket,
    safe_send,
    loop,
):
    """Background: GPT formatting. Sends transcript_update (not transcript)."""
    try:
        gpt_result = await loop.run_in_executor(
            _llm_executor, llm.format_transcript, text
        )
        if not gpt_result:
            return

        # Check we're still the active handler before sending
        if _active_ws.get(session.session_id) is not ws:
            return

        annotated = gpt_result.get("jp_html", text)
        en = gpt_result.get("en_html", "")

        # Send as transcript_update — client overlays on existing seq
        await safe_send(transcript_update_msg(
            seq=seq,
            annotated=annotated,
            en=en,
        ))

        # Re-check after yield — reconnect may have happened during send
        if _active_ws.get(session.session_id) is not ws:
            return

        # Send GPT-derived suggestions as vocabulary delta
        extra_vocab = []
        for s in gpt_result.get("suggestions", []):
            jp = s.get("jp", "").strip()
            if jp:
                extra_vocab.append({
                    "ja": jp,
                    "reading": s.get("reading_kana", ""),
                    "en": s.get("en", ""),
                    "hint": s.get("hint", ""),
                })
        if extra_vocab:
            # Final identity check before mutating session state
            if _active_ws.get(session.session_id) is not ws:
                return
            session.add_vocabulary(extra_vocab)
            await safe_send(vocabulary_msg(seq=seq, mode="delta", items=extra_vocab))
            store.persist(session.session_id)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _run_grammar(
    session: SessionState,
    seq: int,
    ws: WebSocket,
    safe_send,
    loop,
    utterances: List[str],
):
    """Background: grammar suggestion from utterance snapshot."""
    try:
        result = await loop.run_in_executor(
            _llm_executor, llm.grammar_suggestion, utterances
        )
        if not result:
            return

        # Check we're still the active handler
        if _active_ws.get(session.session_id) is not ws:
            return

        session.add_suggestion(result)
        await safe_send(suggestion_msg(seq=seq, items=[result]))
        store.persist(session.session_id)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass
