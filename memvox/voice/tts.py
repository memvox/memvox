import asyncio
import time
from typing import AsyncGenerator, AsyncIterator

import numpy as np

from memvox.observability import metrics
from memvox.voice.types import AudioChunk

_SAMPLE_RATE = 24_000

# Use the compiled Rust extension when available; fall back to pure Python.
try:
    from memvox._rust import SentenceAccumulator as _RustSentenceAccumulator
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class _PySentenceAccumulator:
    """Pure-Python fallback for memvox._rust.SentenceAccumulator.

    Identical behaviour to the Rust implementation — used when the extension
    has not been compiled yet (pre-Phase 4). Switch to the Rust version via:
        maturin develop --manifest-path memvox-rs/Cargo.toml
    """

    _ENDINGS = frozenset(".!?。！？")

    def __init__(self, flush_tokens: int = 30) -> None:
        self._buf = ""
        self._token_count = 0
        self._flush_tokens = flush_tokens

    def push(self, token: str) -> str | None:
        self._buf += token
        self._token_count += 1
        stripped = self._buf.rstrip()
        if (stripped and stripped[-1] in self._ENDINGS) or (
            self._token_count >= self._flush_tokens
        ):
            return self._take()
        return None

    def drain(self) -> str | None:
        s = self._buf.strip()
        self._buf = ""
        self._token_count = 0
        return s or None

    def _take(self) -> str:
        s = self._buf.strip()
        self._buf = ""
        self._token_count = 0
        return s


def _new_accumulator(flush_tokens: int = 30):
    if _HAS_RUST:
        return _RustSentenceAccumulator(flush_tokens=flush_tokens)
    return _PySentenceAccumulator(flush_tokens=flush_tokens)


class TTSEngine:
    """Token stream → AudioChunk stream.

    First audio begins as soon as the first sentence is complete — TTS
    synthesis and LLM generation overlap at the sentence boundary.

    Kokoro is synchronous PyTorch; each sentence is synthesised in a thread-
    pool executor.  Chunks are posted back to the event loop via a queue so
    the generator can yield them as they arrive without buffering the whole
    sentence.
    """

    def __init__(
        self,
        voice: str = "af_heart",
        lang_code: str = "a",        # 'a' = American English, 'k' = Korean
        flush_tokens: int = 30,
        _accumulator=None,           # inject a pre-built accumulator for tests
        _pipeline=None,              # inject a mock pipeline callable for tests
    ) -> None:
        self._voice = voice
        self._lang_code = lang_code
        self._flush_tokens = flush_tokens
        self._accumulator = _accumulator
        self._pipeline = _pipeline

    async def initialize(self) -> None:
        """Load the Kokoro model (runs in a thread — blocks until ready)."""
        if self._pipeline is not None:
            return
        from kokoro import KPipeline
        self._pipeline = await asyncio.to_thread(KPipeline, self._lang_code)

    async def synthesize(
        self, tokens: AsyncIterator[str]
    ) -> AsyncGenerator[AudioChunk, None]:
        """Consume a token stream, yield AudioChunks as sentences are ready."""
        if self._pipeline is None:
            raise RuntimeError("call initialize() before synthesize()")

        acc = self._accumulator if self._accumulator is not None else _new_accumulator(
            self._flush_tokens
        )

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
            sample_rate=_SAMPLE_RATE,
            is_final=True,
            sentence_text="",
            chunk_latency_ms=0.0,
        )

    async def _synthesize_sentence(
        self, sentence: str
    ) -> AsyncGenerator[AudioChunk, None]:
        """Run Kokoro in a thread-pool executor, streaming chunks back via queue."""
        t0 = time.monotonic()
        first = True
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        def _produce() -> None:
            try:
                for _gs, _ps, audio in self._pipeline(sentence, voice=self._voice):
                    loop.call_soon_threadsafe(queue.put_nowait, audio)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        fut = loop.run_in_executor(None, _produce)

        while True:
            audio = await queue.get()
            if audio is None:
                break

            chunk_latency_ms = (time.monotonic() - t0) * 1000
            if first:
                metrics.event(metrics.TTS_FIRST_CHUNK, latency_ms=chunk_latency_ms)
                first = False

            yield AudioChunk(
                pcm_bytes=audio.astype(np.float32).tobytes(),
                sample_rate=_SAMPLE_RATE,
                is_final=False,
                sentence_text=sentence,
                chunk_latency_ms=chunk_latency_ms,
            )

        await fut  # re-raise any exception from the producer thread
