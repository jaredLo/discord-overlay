# Environment Keys Reference

This project uses environment variables to configure audio capture, ASR backends, GPT formatting, concurrency, and UI/debug features. Below is a concise reference for each key: what it does, where it is used in code, how it works, and whether it is deprecated.

Note: File paths below are relative to repo root. Primary runtime is `listener.py` (recorder + ASR + formatter). The desktop app reads only a few flags via the API `/api/health`.

## ASR Backend

- `ASR_BACKEND`
  - Purpose: Select the transcription backend: `openai` | `remote`.
  - Used by: `listener.py` in `transcribe_iter`, `transcribe_chunk` and backend selection near model init.
  - How: Chooses code path for `_openai_transcribe` or `_remote_transcribe`.
  - Status: Active.

- `WHISPER_MODEL`
  - Purpose: When using OpenAI audio (`openai`), set model name (e.g., `whisper-1`).
  - Used by: `listener.py` in `_openai_transcribe` payload.
  - How: Included in the form/body for OpenAI audio transcriptions.
  - Status: Active. (Ignored when `ASR_BACKEND != openai`).

- `REMOTE_ASR_URL`
  - Purpose: Endpoint for self‑hosted Whisper server used by `ASR_BACKEND=remote`.
  - Used by: `listener.py` `_remote_transcribe`.
  - How: Posts WAV bytes; expects JSON with `transcript` or `text`.
  - Status: Active.

## Local Faster‑Whisper (CPU) Tuning (Removed)

- `FASTER_WHISPER_SIZE`
  - Purpose: (Deprecated) Local Faster‑Whisper size.
  - Used by: Previously in `listener.py` local model; now removed.
  - Status: Deprecated — local backend removed.

- `FASTER_WHISPER_COMPUTE`
  - Purpose: (Deprecated) Local compute type.
  - Status: Deprecated — local backend removed.

- `CPU_THREADS`, `NUM_WORKERS`
  - Purpose: (Deprecated) Local model threading.
  - Status: Deprecated — local backend removed.

## Remote/OpenAI API

- `OPENAI_API_KEY`
  - Purpose: Enable OpenAI (audio + chat) calls when configured.
  - Used by: `listener.py` `_openai_transcribe`, `_gpt_request`.
  - How: Bearer token header; when not set, OpenAI paths are skipped.
  - Status: Active.

- `OPENAI_AUDIO_URL` (optional)
  - Purpose: Override OpenAI audio transcription endpoint.
  - Used by: `listener.py` `_openai_transcribe`.
  - How: Replaces default URL.
  - Status: Active (optional).

- `OPENAI_CHAT_URL` (optional)
  - Purpose: Override OpenAI chat/completions endpoint.
  - Used by: `listener.py` `_gpt_request`.
  - How: Replaces default URL.
  - Status: Active (optional).

- `REMOTE_ASR_TIMEOUT_MS`
  - Purpose: Timeout for remote ASR HTTP posts.
  - Used by: `listener.py` `_remote_transcribe` and OpenAI requests.
  - How: Milliseconds to wait on the HTTP client.
  - Status: Active.

## Audio Capture + VAD + Chunking

- `TARGET_DEVICE`
  - Purpose: Substring to select input audio device (e.g., `BlackHole`).
  - Used by: `listener.py` `pick_input`.
  - How: Device name must contain this substring.
  - Status: Active.

- `VAD_LEVEL`
  - Purpose: WebRTC VAD aggressiveness (0..3).
  - Used by: `listener.py` in `chunker_thread` and `utterance_chunks`.
  - How: Higher means more aggressive silence detection.
  - Status: Active.

- `SILENCE_HANG_MS`
  - Purpose: Hangover before finalizing a chunk after speech ends.
  - Used by: `listener.py` chunkers.
  - How: Milliseconds worth of non‑speech frames required to end an utterance.
  - Status: Active.

- `MIN_CHUNK_SEC`, `MAX_CHUNK_SEC`
  - Purpose: Lower/upper bounds for chunk duration.
  - Used by: `listener.py` chunkers.
  - How: Prevents sub‑min and too‑long chunks; shorter tails may be carried to next.
  - Status: Active.

- `CHUNK_OVERLAP_MS`, `DISABLE_OVERLAP`
  - Purpose: Overlap tail padding between chunks; allow disabling.
  - Used by: `listener.py` chunkers.
  - How: Tail frames appended to next utterance to reduce boundary clipping.
  - Status: Active.

- `DISABLE_CARRY`
  - Purpose: Disable carrying short segments across silence to next chunk.
  - Used by: `listener.py` chunkers.
  - How: When true, drops too‑short tails instead of accumulating.
  - Status: Active.

- `CAPTURE_ALL`
  - Purpose: Relax quality gates (keep short/noisy outputs) and use lower beam size.
  - Used by: `listener.py` `good_seg`, guards in workers.
  - How: When true, suppresses some filters that drop short/noisy lines.
  - Status: Active.

## GPT Formatter (two‑line JP/EN + Suggestions)

- `USE_GPT_FORMATTER`
  - Purpose: Enable GPT formatting (jp_html + en_html + suggestions JSON).
  - Used by: `listener.py` `_gpt_request` and `formatter_worker`.
  - How: When false, GPT paths are skipped; fallback still emits JP/EN and local Vocab.
  - Status: Active.

- `OPENAI_MODEL`
  - Purpose: Chat/completions model for `_gpt_request`.
  - Used by: `listener.py` payload.
  - Status: Active.

- `GPT_COMPRESS_ELONGATIONS`
  - Purpose: Allow compressing extreme elongations before GPT.
  - Used by: `listener.py` `_sanitize_for_gpt`.
  - Status: Active.

- `GPT_MAX_VOCAB`
  - Purpose: Cap the number of vocab items shown/processed.
  - Used by: `listener.py` local Vocab builder.
  - Status: Active.

- `GPT_MAX_CHARS`
  - Purpose: Limit utterance characters sent to GPT.
  - Used by: `listener.py` `_sanitize_for_gpt`.
  - Status: Active.

- `GPT_RATE_LIMIT_MS`
  - Purpose: Minimum milliseconds between GPT calls (global rate limit in process).
  - Used by: `listener.py` `formatter_worker` with `gpt_lock`.
  - Status: Active.

- `GPT_CLEAN_REPEATS`
  - Purpose: Collapse repeated sequences before sending to GPT.
  - Used by: `listener.py` `_sanitize_for_gpt`.
  - Status: Active.

- `GPT_SKIP_ON_REPEAT`
  - Purpose: Skip GPT entirely when repetition score is high.
  - Used by: `listener.py` `_sanitize_for_gpt`.
  - Status: Active.

- `GPT_REPEAT_SKIP_THRESHOLD`
  - Purpose: Repetition score threshold to skip GPT (lower = more aggressive).
  - Used by: `listener.py` `_sanitize_for_gpt`.
  - Status: Active.

- `LINE4_TABLE_MODE`
  - Purpose: Old table layout for “details” line (markers, context, etc.).
  - Used by: (historical) `listener.py` `_render_four_lines`.
  - How: This path has been removed from output.
  - Status: Deprecated — not emitted; kept only as vestige.

- `JP_VERBATIM`
  - Purpose: Historical toggle for verbatim JP handling.
  - Used by: (legacy) GPT payload; not needed now.
  - Status: Deprecated — no effect with the current lean prompt.

## Concurrency / Workers

- `ASR_WORKERS`
  - Purpose: Number of ASR worker threads consuming chunk queue.
  - Used by: `listener.py` worker startup.
  - Status: Active.

- `FORMAT_WORKERS`
  - Purpose: Number of formatter threads (JP/EN + suggestions) consuming text queue.
  - Used by: `listener.py` worker startup.
  - Status: Active.

## Optional Translation

- `DEEPL_API_KEY`
  - Purpose: Enable DeepL per‑utterance translation.
  - Used by: (legacy path) not active in current code.
  - Status: Deprecated — not used; translation handled by GPT `en_html` when enabled.

## Output / UI / Debug Flags

- `SHOW_VOCAB_LINE`
  - Purpose: Whether to include a “Vocab:” line in transcript.
  - Used by: `listener.py` (previously gated); now always appended.
  - Status: Deprecated — code always emits Vocab to feed the left sidebar.

- `SHOW_DETAILS_LINE`
  - Purpose: Old “details” line enable (markers/context/personal/etc.).
  - Used by: (legacy); current UI ignores this and builds Suggestions separately.
  - Status: Deprecated — UI no longer shows a details line. Suggestions are built via server.

- `SHOW_ASR_DEBUG`
  - Purpose: Enable ASR debug compare (OpenAI, remote, local) sidebar.
  - Used by: `server/main.py` `/api/health` flag; desktop checks to show debug; `listener.py` to spawn debug threads.
  - Status: Active. Set to `false` to disable debug threads and sidebar.

- `SIM_API_URL`
  - Purpose: Optional HTTP endpoint for internal JP similarity suggestions.
  - Used by: `server/main.py` `_internal_suggestions`.
  - How: If set, server POSTs `{ bases, top_k }` and expects `{ items: [{ ja, read, en }] }`.
  - Status: Active (optional). Alternatively, the local `jp_internal.py` provider is used.

## Quick Recipes

- Use home Whisper server only
  - `ASR_BACKEND=remote`
  - `REMOTE_ASR_URL=http://127.0.0.1:8585/api/transcribe`
  - `USE_GPT_FORMATTER=false`
  - `SHOW_ASR_DEBUG=false`
  - Leave `OPENAI_API_KEY` unset.


- Token‑lean with GPT suggestions
  - `USE_GPT_FORMATTER=true`
  - `GPT_REPEAT_SKIP_THRESHOLD=2.0` (skip repetitive)
  - Keep other GPT_* conservative.

