# Discord Overlay – Tauri Client + FastAPI (KotoFloat - 言葉 (kotoba/words) + float)

I am passionate on learning Japanese, so I made this tool, a desktop overlay that listens to Japanese conversation and extracts keywords in real-time, helping learners follow along during voice calls(mainly on Japanese-English language exchange servers on Discord) without pausing to look things up.

## Prereqs
- Python (Conda env): `conda activate discord-overlay`
- Node.js: `nvm use v24`
- Rust toolchain (for Tauri): `rustup` with stable toolchain installed

## Easiest: Makefile (API + Tauri together)
```
make dev
```

This starts FastAPI on :8201 (spawns listener.py) and then launches Tauri dev. When you close Tauri, the API stops automatically.

## Manual: start backend then Tauri
```
uvicorn server.main:app --reload --port 8201
cd desktop && npm install && npm run tauri:dev
```

The Tauri client polls:
- `GET /api/overlay/transcript` for the annotated transcript HTML
- `GET /api/overlay/waveform` for rolling amplitude data

Listener lifecycle: When the FastAPI app starts, it launches `listener.py` using the current Python (`sys.executable`). On shutdown, it terminates it.

## Notes
- You can change the API URL in the app header; it defaults to `http://127.0.0.1:8201` and persists in `localStorage`.
- The transcript view follows the tail unless you scroll up or select text.
- The waveform renders from values in `waveform.json` (range -100..100).
