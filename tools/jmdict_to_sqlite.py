#!/usr/bin/env python3
"""
Convert a compact JMdict JSON index (term -> [glosses]) into a SQLite DB.

Input:  jmdict.min.json (as produced by tools/jmdict_build.py)
Output: jmdict.sqlite with table:
  entries(term TEXT PRIMARY KEY, gloss TEXT)  -- gloss is JSON array string

Usage:
  python tools/jmdict_to_sqlite.py jmdict.min.json jmdict.sqlite
"""
import sys, json, sqlite3

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/jmdict_to_sqlite.py jmdict.min.json jmdict.sqlite")
        sys.exit(2)
    src, out = sys.argv[1], sys.argv[2]
    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)

    con = sqlite3.connect(out)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("DROP TABLE IF EXISTS entries;")
    cur.execute("CREATE TABLE entries(term TEXT PRIMARY KEY, gloss TEXT);")

    batch = []
    for k, v in data.items():
        if not k:
            continue
        if isinstance(v, list):
            gloss_json = json.dumps(v, ensure_ascii=False)
        else:
            gloss_json = json.dumps([v], ensure_ascii=False)
        batch.append((k, gloss_json))
        if len(batch) >= 5000:
            cur.executemany("INSERT OR REPLACE INTO entries(term, gloss) VALUES(?, ?)", batch)
            batch.clear()
    if batch:
        cur.executemany("INSERT OR REPLACE INTO entries(term, gloss) VALUES(?, ?)", batch)
    con.commit()
    con.close()
    print(f"Built {out}")

if __name__ == '__main__':
    main()

