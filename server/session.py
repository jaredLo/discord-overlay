"""Session state management with SQLite persistence.

Each WebSocket connection creates a session. State is persisted to SQLite
after each transcript event (survives VM restarts, OOM kills).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from server.config import DATA_DIR, SESSION_ORPHAN_TIMEOUT_S


@dataclass
class SessionState:
    session_id: str
    last_seq: int = 0
    utterances: list = field(default_factory=list)  # recent raw transcripts
    vocabulary: list = field(default_factory=list)   # accumulated vocab items
    suggestions: list = field(default_factory=list)  # grammar suggestions
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def next_seq(self) -> int:
        self.last_seq += 1
        self.updated_at = time.time()
        return self.last_seq

    def add_utterance(self, text: str):
        self.utterances.append(text)
        # Keep only last 5 for LLM context window
        if len(self.utterances) > 5:
            self.utterances = self.utterances[-5:]
        self.updated_at = time.time()

    def add_vocabulary(self, items: list):
        seen = {v["ja"] for v in self.vocabulary}
        for item in items:
            if item.get("ja") and item["ja"] not in seen:
                self.vocabulary.append(item)
                seen.add(item["ja"])
        self.updated_at = time.time()

    def add_suggestion(self, item: dict):
        self.suggestions.append(item)
        # Cap at 5, drop oldest
        if len(self.suggestions) > 5:
            self.suggestions = self.suggestions[-5:]
        self.updated_at = time.time()


class SessionStore:
    """Manages active sessions with SQLite-backed persistence."""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._orphan_tasks: Dict[str, asyncio.Task] = {}
        self._db_path = DATA_DIR / "sessions.db"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                last_seq INTEGER DEFAULT 0,
                utterances TEXT DEFAULT '[]',
                vocabulary TEXT DEFAULT '[]',
                suggestions TEXT DEFAULT '[]',
                created_at REAL,
                updated_at REAL
            )
        """)
        conn.commit()
        conn.close()

    def create_session(self, session_id: Optional[str] = None) -> SessionState:
        sid = session_id or str(uuid.uuid4())

        # Cancel orphan cleanup if reconnecting to same session
        if sid in self._orphan_tasks:
            self._orphan_tasks[sid].cancel()
            del self._orphan_tasks[sid]

        # Try to restore from DB
        if sid in self._sessions:
            return self._sessions[sid]

        state = self._load_from_db(sid)
        if state is None:
            state = SessionState(session_id=sid)
        self._sessions[sid] = state
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def persist(self, session_id: str):
        """Write session state to SQLite. Called after each transcript event."""
        state = self._sessions.get(session_id)
        if state is None:
            return
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                    (session_id, last_seq, utterances, vocabulary, suggestions, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                state.session_id,
                state.last_seq,
                json.dumps(state.utterances, ensure_ascii=False),
                json.dumps(state.vocabulary, ensure_ascii=False),
                json.dumps(state.suggestions, ensure_ascii=False),
                state.created_at,
                state.updated_at,
            ))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _load_from_db(self, session_id: str) -> Optional[SessionState]:
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            conn.close()
            if row is None:
                return None
            return SessionState(
                session_id=row["session_id"],
                last_seq=row["last_seq"],
                utterances=json.loads(row["utterances"]),
                vocabulary=json.loads(row["vocabulary"]),
                suggestions=json.loads(row["suggestions"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        except Exception:
            return None

    async def mark_disconnected(self, session_id: str):
        """Start orphan cleanup timer. Persist state, cancel pending work after timeout."""
        # Persist immediately on disconnect
        self.persist(session_id)

        async def _orphan_cleanup():
            await asyncio.sleep(SESSION_ORPHAN_TIMEOUT_S)
            # Session wasn't reclaimed — clean up
            self.persist(session_id)
            self._sessions.pop(session_id, None)

        task = asyncio.create_task(_orphan_cleanup())
        self._orphan_tasks[session_id] = task

    def end_session(self, session_id: str):
        """Explicitly end a session. Persist and remove."""
        self.persist(session_id)
        self._sessions.pop(session_id, None)
        if session_id in self._orphan_tasks:
            self._orphan_tasks[session_id].cancel()
            del self._orphan_tasks[session_id]


# Singleton
store = SessionStore()
