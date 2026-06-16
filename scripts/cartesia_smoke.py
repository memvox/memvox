"""Isolated Cartesia Sonic smoke test — no mic / ASR / LLM / Ollama.

Synthesizes one sentence through the real CartesiaTTS backend and plays it,
reporting time-to-first-chunk and total audio duration. Use this to validate
your key + voice UUID before running a full `python -m memvox` session.

    export CARTESIA_API_KEY=sk_car_...
    export CARTESIA_VOICE_ID=<voice-uuid>     # from https://play.cartesia.ai/
    python scripts/cartesia_smoke.py "안녕하세요, 오늘 기분이 어때요?"
"""

import asyncio
import os
import sys
import time

import numpy as np
import sounddevice as sd

from memvox.voice.tts_base import SAMPLE_RATE
from memvox.voice.tts_cartesia import CartesiaTTS


async def _tokens(text: str):
    """Fake an LLM token stream so synthesize() has something to chew on."""
    for word in text.split():
        yield word + " "


async def main() -> None:
    # Default is a mixed sentence so you hear the effect: English at normal
    # speed, the embedded Korean help phrase rendered slowly.
    text = sys.argv[1] if len(sys.argv) > 1 else "To say hello in Korean, say 안녕하세요."
    voice_id = os.environ.get("CARTESIA_VOICE_ID", "")
    if not voice_id:
        sys.exit("Set CARTESIA_VOICE_ID (a voice UUID from play.cartesia.ai).")

    tts = CartesiaTTS(voice_id=voice_id, lang_code="ko", model="sonic-2")
    await tts.initialize()
    print(f"[smoke] synthesizing: {text!r}")

    t0 = time.monotonic()
    first_ms = None
    pcm = bytearray()
    async for chunk in tts.synthesize(_tokens(text)):
        if chunk.is_final:
            continue
        if first_ms is None:
            first_ms = (time.monotonic() - t0) * 1000
            print(f"[smoke] first audio chunk: {first_ms:.0f} ms")
        pcm.extend(chunk.pcm_bytes)
    await tts.close()

    audio = np.frombuffer(bytes(pcm), dtype="<f4")
    dur = len(audio) / SAMPLE_RATE
    print(f"[smoke] total: {len(audio)} samples ({dur:.2f}s) @ {SAMPLE_RATE} Hz")
    if len(audio):
        print("[smoke] playing…")
        sd.play(audio, SAMPLE_RATE)
        sd.wait()
    print("[smoke] done.")


if __name__ == "__main__":
    asyncio.run(main())
