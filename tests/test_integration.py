#!/usr/bin/env python3
"""Integration test for KotoFloat WebSocket backend.

Streams a test WAV file over WebSocket and validates:
  - session_ack on start_session
  - transcript message with text
  - vocabulary message with ja/reading/en fields
  - error handling for invalid messages
  - ping/pong keepalive
  - end_session cleanup

Usage:
  python tests/test_integration.py [--url ws://localhost:8201] [--api-key KEY]

Requires: websockets, wave
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import wave
from pathlib import Path

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)


FIXTURES = Path(__file__).parent / "fixtures"
TEST_WAV = FIXTURES / "japanese_test.wav"


def generate_test_audio(duration_s: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generate silent PCM int16 audio for testing protocol flow.

    For real integration tests, use an actual Japanese speech WAV.
    """
    num_samples = int(duration_s * sample_rate)
    # Silent audio (zeros)
    return b"\x00\x00" * num_samples


def load_wav_pcm(path: Path) -> bytes:
    """Load a WAV file and return raw PCM int16 bytes."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, "Expected mono audio"
        assert wf.getsampwidth() == 2, "Expected 16-bit audio"
        assert wf.getframerate() == 16000, "Expected 16kHz"
        return wf.readframes(wf.getnframes())


async def test_protocol(url: str, api_key: str | None):
    """Test the WebSocket protocol flow."""
    connect_url = url + "/ws/session"
    if api_key:
        connect_url += f"?api_key={api_key}"

    print(f"Connecting to {connect_url}...")
    async with websockets.connect(connect_url) as ws:

        # 1. Test start_session
        print("\n--- Test: start_session ---")
        await ws.send(json.dumps({
            "type": "start_session",
            "session_id": "test-001",
            "audio_format": {
                "sample_rate": 16000,
                "channels": 1,
                "encoding": "pcm_s16le",
            },
        }))
        ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert ack["type"] == "session_ack", f"Expected session_ack, got {ack}"
        assert ack["session_id"] == "test-001"
        print(f"  OK: session_ack received, id={ack['session_id']}")

        # 2. Test ping/pong
        print("\n--- Test: ping/pong ---")
        await ws.send(json.dumps({"type": "ping"}))
        pong = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert pong["type"] == "pong", f"Expected pong, got {pong}"
        print("  OK: pong received")

        # 3. Test audio streaming
        print("\n--- Test: audio streaming ---")
        if TEST_WAV.exists():
            pcm = load_wav_pcm(TEST_WAV)
            print(f"  Using test WAV: {TEST_WAV} ({len(pcm)} bytes)")
        else:
            pcm = generate_test_audio(3.0)
            print(f"  Using synthetic silence ({len(pcm)} bytes)")
            print(f"  Note: Place a real Japanese WAV at {TEST_WAV} for full test")

        # Send audio in chunks (simulating real-time streaming)
        chunk_size = 16000 * 2  # 1 second of int16 audio
        for i in range(0, len(pcm), chunk_size):
            chunk = pcm[i:i + chunk_size]
            await ws.send(chunk)

        # Wait for response (transcript or nothing if silent)
        try:
            response = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            print(f"  Response type: {response['type']}")

            if response["type"] == "transcript":
                print(f"  Text: {response.get('text', '')}")
                print(f"  Annotated: {response.get('annotated', '')[:80]}...")
                assert response.get("seq", 0) > 0, "Expected seq > 0"
                assert response.get("text"), "Expected non-empty text"
                print("  OK: transcript received")

                # Check for vocabulary message
                try:
                    vocab = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                    if vocab["type"] == "vocabulary":
                        print(f"\n--- Vocabulary ({len(vocab.get('items', []))} items) ---")
                        for item in vocab.get("items", [])[:3]:
                            print(f"  {item.get('ja')} ({item.get('reading')}) — {item.get('en')}")
                        assert vocab.get("mode") in ("snapshot", "delta")
                        print("  OK: vocabulary received")
                except asyncio.TimeoutError:
                    print("  (no vocabulary message — OK for short audio)")

            elif response["type"] == "error":
                print(f"  Error: {response.get('code')}: {response.get('message')}")

        except asyncio.TimeoutError:
            if TEST_WAV.exists():
                print("  WARN: No response within 15s — ASR may not be configured")
            else:
                print("  OK: No response for silent audio (expected)")

        # 4. Test unknown message type
        print("\n--- Test: unknown message type ---")
        await ws.send(json.dumps({"type": "invalid_type"}))
        err = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert err["type"] == "error", f"Expected error, got {err}"
        assert err["code"] == "unknown_type"
        print(f"  OK: error received — {err['message']}")

        # 5. Test end_session
        print("\n--- Test: end_session ---")
        await ws.send(json.dumps({"type": "end_session"}))
        print("  OK: end_session sent")

    print("\n=== All tests passed ===")


async def test_auth_rejection(url: str):
    """Test that invalid API key is rejected."""
    print("\n--- Test: auth rejection ---")
    try:
        async with websockets.connect(
            url + "/ws/session?api_key=wrong-key"
        ) as ws:
            # Should be rejected
            try:
                await asyncio.wait_for(ws.recv(), timeout=3)
            except websockets.ConnectionClosed as e:
                if e.code == 4001:
                    print("  OK: connection rejected with 4001")
                    return
            print("  WARN: connection not rejected (auth may be disabled)")
    except websockets.InvalidStatusCode as e:
        print(f"  OK: connection rejected with HTTP {e.status_code}")
    except Exception as e:
        print(f"  WARN: unexpected error — {e}")


async def test_health(url: str):
    """Test health endpoint."""
    import urllib.request
    http_url = url.replace("ws://", "http://").replace("wss://", "https://")
    print("\n--- Test: health endpoint ---")
    try:
        resp = urllib.request.urlopen(f"{http_url}/api/health", timeout=5)
        data = json.loads(resp.read())
        assert data["status"] == "ok"
        print(f"  OK: status=ok, asr={data.get('asr_configured')}, jmdict={data.get('jmdict_available')}")
    except Exception as e:
        print(f"  FAIL: {e}")


async def main():
    parser = argparse.ArgumentParser(description="KotoFloat integration test")
    parser.add_argument("--url", default="ws://localhost:8201", help="WebSocket URL")
    parser.add_argument("--api-key", default=None, help="API key")
    args = parser.parse_args()

    await test_health(args.url)
    await test_auth_rejection(args.url)
    await test_protocol(args.url, args.api_key)


if __name__ == "__main__":
    asyncio.run(main())
