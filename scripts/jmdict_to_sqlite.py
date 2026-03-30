#!/usr/bin/env python3
"""Convert jmdict-simplified JSON to SQLite for fast lookups.

Usage: python jmdict_to_sqlite.py input.json output.db

Input: jmdict-eng-X.X.X.json from scriptin/jmdict-simplified
Output: SQLite DB with entries table
"""

import json
import sqlite3
import sys


def convert(input_path: str, output_path: str):
    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words", [])
    print(f"Found {len(words)} entries")

    conn = sqlite3.connect(output_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY,
            kanji TEXT,
            reading TEXT,
            sense TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kanji ON entries(kanji)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reading ON entries(reading)")

    batch = []
    for entry in words:
        eid = entry.get("id", "")

        # Extract kanji forms
        kanji_forms = [k.get("text", "") for k in entry.get("kanji", [])]

        # Extract reading forms
        reading_forms = [r.get("text", "") for r in entry.get("kana", [])]

        # Extract senses with glosses and POS
        senses = []
        for sense in entry.get("sense", []):
            glosses = [g.get("text", "") for g in sense.get("gloss", [])]
            pos = sense.get("partOfSpeech", [])
            senses.append({"glosses": glosses, "pos": pos})

        batch.append((
            eid,
            json.dumps(kanji_forms, ensure_ascii=False),
            json.dumps(reading_forms, ensure_ascii=False),
            json.dumps(senses, ensure_ascii=False),
        ))

        if len(batch) >= 5000:
            conn.executemany(
                "INSERT OR REPLACE INTO entries (id, kanji, reading, sense) VALUES (?, ?, ?, ?)",
                batch,
            )
            batch = []

    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO entries (id, kanji, reading, sense) VALUES (?, ?, ?, ?)",
            batch,
        )

    conn.commit()

    # Build FTS for fast text search
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
            kanji, reading, content=entries, content_rowid=id
        )
    """)
    conn.execute("""
        INSERT INTO entries_fts(entries_fts) VALUES('rebuild')
    """)
    conn.commit()
    conn.close()

    print(f"Written {len(words)} entries to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.json output.db")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
