JMdict Offline Glossing

Overview
- The overlay can annotate Japanese with readings + English glosses offline.
- It looks up glosses in this order:
  1) `keywords.json` (exact surface match)
  2) `glossary.json` (exact surface/lemma match)
  3) `jmdict.sqlite` (SQLite index; fastest for large dictionaries)
  4) `jmdict.min.json` (compact JSON index)
  5) Optional online fallback (DeepL/Jisho) if `ALLOW_ONLINE_GLOSS=1`

Quick Start (SQLite recommended)
- Get JMdict_e (English) XML from the JMdict project (EDRDG). File may be `JMdict_e.gz`.
- Build a compact JSON index:
  - `python tools/jmdict_build.py /path/to/JMdict_e.gz jmdict.min.json`
- Convert JSON index to SQLite:
  - `python tools/jmdict_to_sqlite.py jmdict.min.json jmdict.sqlite`
- Run overlay (auto-detects files in repo root):
  - `JMDICT_DB=jmdict.sqlite OUTPUT_MODE=annotate python overlay.py`

File Formats
- `jmdict.min.json`: `{ "単語": ["english gloss 1", "gloss 2"], ... }`
  - Keys include both kanji (`keb`) and reading (`reb`) forms.
  - Values are a short list of English gloss strings; the overlay uses the first.
- `jmdict.sqlite`: SQLite database with table `entries(term TEXT PRIMARY KEY, gloss TEXT)`
  - `term`: surface form (kanji or reading)
  - `gloss`: JSON array string (e.g., `["record", "minutes"]`)

Environment Variables
- `JMDICT_DB`: Path to SQLite index (default `jmdict.sqlite`)
- `JMDICT_JSON`: Path to JSON index (default `jmdict.min.json`)
- `ALLOW_ONLINE_GLOSS`: `0` (default, offline-only) or `1` to allow brief online fallback
- `ASR_BEAM_SIZE`: Beam size for Whisper (default `1` for speed)
- `MODEL_SIZE`, `COMPUTE_TYPE`: Faster-whisper knobs (e.g., `tiny`, `int8`)

Tips
- Keep `keywords.json` for custom names or specific translations you prefer.
- Use `glossary.json` for quick local additions without rebuilding JMdict.
- SQLite avoids large JSON load time and is recommended for big dictionaries.

