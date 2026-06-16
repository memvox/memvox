"""Cartesia Sonic TTS backend (bring-your-own-key).

Streams audio over Cartesia's TTS websocket for low first-byte latency
(~75–90 ms). Requires a Cartesia API key — set CARTESIA_API_KEY in the
environment (or pass api_key=). Install the optional dependency with:

    pip install -e ".[cartesia]"

Output is requested as raw little-endian float32 PCM at 24 kHz, which is the
exact AudioChunk convention the Rust egress expects — no conversion needed.

NOTE: the cartesia-python SDK surface drifts between releases (see NOTES.md).
This backend reads audio defensively (attribute *or* dict access) so a minor
SDK bump doesn't break playback.
"""

import asyncio
import os
import time
from typing import Any, AsyncGenerator, AsyncIterator

from memvox.observability import metrics
from memvox.voice.tts_base import (
    SAMPLE_RATE,
    new_accumulator,
    segment_for_tts,
)
from memvox.voice.types import AudioChunk

_DEFAULT_MODEL = "sonic-2"


def _extract_audio(item: Any) -> bytes | None:
    """Pull the raw PCM bytes out of one websocket output, SDK-version-agnostic."""
    audio = getattr(item, "audio", None)
    if audio is None and isinstance(item, dict):
        audio = item.get("audio")
    if audio is None:
        return None
    return bytes(audio)


class CartesiaTTS:
    """Token stream → AudioChunk stream via Cartesia Sonic.

    Mirrors TTSEngine's interface: initialize() then synthesize(tokens).
    Sentences are accumulated from the token stream (same boundary logic as
    XTTS) and synthesised one at a time so audio starts at the first sentence.
    """

    def __init__(
        self,
        voice_id: str,
        lang_code: str = "en",
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        flush_tokens: int = 30,
        korean_help_speed: str = "slow",
        _client=None,           # inject a fake AsyncCartesia for tests
    ) -> None:
        if not voice_id:
            raise ValueError(
                "CartesiaTTS requires a voice_id (a Cartesia voice UUID). "
                "Set cartesia_voice_id in the skin's SessionConfig."
            )
        self._voice_id = voice_id
        self._lang_code = lang_code
        self._model = model
        self._api_key = api_key or os.environ.get("CARTESIA_API_KEY")
        self._flush_tokens = flush_tokens
        self._korean_help_speed = korean_help_speed
        self._client = _client
        self._ws = None

    async def initialize(self) -> None:
        """Create the async client and open the TTS websocket."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "CARTESIA_API_KEY is not set. Export it, or switch the skin "
                    "to a local TTS backend (tts_backend='xtts')."
                )
            try:
                from cartesia import AsyncCartesia
            except ImportError as e:
                raise RuntimeError(
                    "The Cartesia backend needs the `cartesia` package. "
                    'Install it with: pip install -e ".[cartesia]"'
                ) from e
            self._client = AsyncCartesia(api_key=self._api_key)

        self._ws = await self._client.tts.websocket()

    async def synthesize(
        self, tokens: AsyncIterator[str]
    ) -> AsyncGenerator[AudioChunk, None]:
        """Consume a token stream, yield AudioChunks as sentences complete."""
        if self._ws is None:
            raise RuntimeError("call initialize() before synthesize()")

        acc = new_accumulator(self._flush_tokens)

        async for token in tokens:
            sentence = acc.push(token)
            if sentence:
                async for chunk in self._synthesize_sentence(sentence):
                    yield chunk

        remainder = acc.drain()
        if remainder:
            async for chunk in self._synthesize_sentence(remainder):
                yield chunk

        yield AudioChunk(
            pcm_bytes=b"",
            sample_rate=SAMPLE_RATE,
            is_final=True,
            sentence_text="",
            chunk_latency_ms=0.0,
        )

    async def _synthesize_sentence(
        self, sentence: str
    ) -> AsyncGenerator[AudioChunk, None]:
        # A sentence may interleave scripts ("the word is 안녕하세요"): split it
        # into runs and synthesize each in its own language, slowing embedded
        # Korean help phrases. Runs stream in order over the same websocket.
        t0 = time.monotonic()
        first = True

        for run_text, language, speed in segment_for_tts(
            sentence, self._lang_code, self._korean_help_speed
        ):
            output = await self._ws.send(
                model_id=self._model,
                transcript=run_text,
                voice={"mode": "id", "id": self._voice_id},
                language=language,
                speed=speed,
                output_format={
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate": SAMPLE_RATE,
                },
                stream=True,
            )

            async for item in output:
                pcm = _extract_audio(item)
                if not pcm:
                    continue

                chunk_latency_ms = (time.monotonic() - t0) * 1000
                if first:
                    metrics.event(metrics.TTS_FIRST_CHUNK, latency_ms=chunk_latency_ms)
                    first = False

                yield AudioChunk(
                    pcm_bytes=pcm,
                    sample_rate=SAMPLE_RATE,
                    is_final=False,
                    sentence_text=run_text,
                    chunk_latency_ms=chunk_latency_ms,
                )

    async def close(self) -> None:
        """Close the websocket and underlying client (best-effort)."""
        for target in (self._ws, self._client):
            close = getattr(target, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass
        self._ws = None
