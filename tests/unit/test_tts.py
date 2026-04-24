import asyncio

import numpy as np
import pytest

from memvox.observability import metrics
from memvox.voice.tts import TTSEngine, _PySentenceAccumulator
from memvox.voice.types import AudioChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pcm(seconds: float = 0.1) -> np.ndarray:
    """Fake 24kHz float32 audio."""
    return np.zeros(int(24_000 * seconds), dtype=np.float32)


def _pipeline(*sentences_audio: list[np.ndarray]):
    """Mock KPipeline: maps call index to a list of audio arrays to yield."""
    calls = iter(sentences_audio)

    def pipeline(text, voice=None, **_):
        chunks = next(calls, [_pcm()])
        for audio in chunks:
            yield None, None, audio

    return pipeline


async def _tokens(*words: str) -> list[str]:
    """Async iterator over a fixed list of tokens."""
    for w in words:
        yield w


async def _collect(engine: TTSEngine, *tokens: str) -> list[AudioChunk]:
    return [c async for c in engine.synthesize(_tokens(*tokens))]


def _content(chunks: list[AudioChunk]) -> list[AudioChunk]:
    return [c for c in chunks if not c.is_final]


# ── _PySentenceAccumulator ────────────────────────────────────────────────────

class TestPySentenceAccumulator:
    def test_no_flush_mid_sentence(self):
        acc = _PySentenceAccumulator()
        assert acc.push("Hello") is None
        assert acc.push(" world") is None

    def test_flush_on_period(self):
        acc = _PySentenceAccumulator()
        acc.push("Hello")
        result = acc.push(" world.")
        assert result == "Hello world."

    def test_flush_on_question_mark(self):
        acc = _PySentenceAccumulator()
        acc.push("Really")
        result = acc.push("?")
        assert result is not None

    def test_flush_on_korean_punctuation(self):
        acc = _PySentenceAccumulator()
        acc.push("안녕하세요")
        result = acc.push("。")
        assert result is not None

    def test_flush_on_token_limit(self):
        acc = _PySentenceAccumulator(flush_tokens=3)
        acc.push("one")
        acc.push("two")
        result = acc.push("three")
        assert result is not None

    def test_drain_returns_remainder(self):
        acc = _PySentenceAccumulator()
        acc.push("hello")
        result = acc.drain()
        assert result == "hello"

    def test_drain_empty_returns_none(self):
        acc = _PySentenceAccumulator()
        assert acc.drain() is None

    def test_buffer_clears_after_flush(self):
        acc = _PySentenceAccumulator()
        acc.push("Hi.")
        acc.push("Next")
        drained = acc.drain()
        assert drained == "Next"


# ── TTSEngine.synthesize ──────────────────────────────────────────────────────

class TestTTSEngineSynthesize:
    def _engine(self, *sentences_audio):
        acc = _PySentenceAccumulator()
        return TTSEngine(_accumulator=acc, _pipeline=_pipeline(*sentences_audio))

    async def test_yields_audio_for_complete_sentence(self):
        engine = self._engine([_pcm()])
        chunks = await _collect(engine, "Hello", " world", ".")
        assert any(len(c.pcm_bytes) > 0 for c in _content(chunks))

    async def test_final_chunk_emitted(self):
        engine = self._engine([_pcm()])
        chunks = await _collect(engine, "Hi.")
        assert chunks[-1].is_final
        assert chunks[-1].pcm_bytes == b""

    async def test_sentence_text_carried_on_chunks(self):
        engine = self._engine([_pcm()])
        chunks = await _collect(engine, "Hello", " world", ".")
        content = _content(chunks)
        assert all(c.sentence_text != "" for c in content)

    async def test_sample_rate_is_24k(self):
        engine = self._engine([_pcm()])
        chunks = await _collect(engine, "Hi.")
        assert all(c.sample_rate == 24_000 for c in chunks)

    async def test_remainder_drained_at_end(self):
        # No sentence-ending punctuation — content should still be synthesised
        engine = self._engine([_pcm()])
        chunks = await _collect(engine, "no", " punctuation")
        assert any(len(c.pcm_bytes) > 0 for c in _content(chunks))

    async def test_multiple_sentences(self):
        engine = self._engine([_pcm()], [_pcm()])
        chunks = await _collect(engine, "First.", " Second.")
        assert len(_content(chunks)) >= 2

    async def test_multiple_kokoro_chunks_per_sentence(self):
        engine = self._engine([_pcm(0.05), _pcm(0.05), _pcm(0.05)])
        chunks = await _collect(engine, "Long sentence.")
        assert len(_content(chunks)) == 3

    async def test_pcm_bytes_are_float32(self):
        audio = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        engine = self._engine([audio])
        chunks = await _collect(engine, "Test.")
        content = _content(chunks)
        recovered = np.frombuffer(content[0].pcm_bytes, dtype=np.float32)
        np.testing.assert_array_almost_equal(recovered, audio)

    async def test_raises_without_initialize(self):
        engine = TTSEngine()  # no _pipeline injected
        with pytest.raises(RuntimeError, match="initialize"):
            async for _ in engine.synthesize(_tokens("hi")):
                pass


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestTTSMetrics:
    async def test_emits_tts_first_chunk_event(self):
        acc = _PySentenceAccumulator()
        engine = TTSEngine(_accumulator=acc, _pipeline=_pipeline([_pcm()]))
        with metrics.override() as sink:
            await _collect(engine, "Hello.")
        events = [e for e in sink.events if e.name == metrics.TTS_FIRST_CHUNK]
        assert len(events) == 1
        assert events[0].latency_ms is not None
        assert events[0].latency_ms >= 0

    async def test_emits_one_event_per_sentence(self):
        acc = _PySentenceAccumulator()
        engine = TTSEngine(_accumulator=acc, _pipeline=_pipeline([_pcm()], [_pcm()]))
        with metrics.override() as sink:
            await _collect(engine, "First.", " Second.")
        events = [e for e in sink.events if e.name == metrics.TTS_FIRST_CHUNK]
        assert len(events) == 2

    async def test_chunk_latency_ms_is_positive(self):
        acc = _PySentenceAccumulator()
        engine = TTSEngine(_accumulator=acc, _pipeline=_pipeline([_pcm()]))
        chunks = await _collect(engine, "Hello.")
        assert all(c.chunk_latency_ms >= 0 for c in _content(chunks))
