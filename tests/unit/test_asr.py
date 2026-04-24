import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from memvox.observability import metrics
from memvox.voice.asr import ASREngine
from memvox.voice.types import SpeechSegment, Transcript


def _make_segment(duration_ms: float = 2000.0) -> SpeechSegment:
    samples = int(16000 * duration_ms / 1000)
    audio = (np.zeros(samples, dtype=np.int16)).tobytes()
    return SpeechSegment(
        audio=audio,
        speech_prob=0.95,
        duration_ms=duration_ms,
        timestamp_start=0.0,
    )


def _mock_model(text: str = "안녕하세요", no_speech_prob: float = 0.05, language: str = "ko"):
    seg = MagicMock()
    seg.text = text
    info = MagicMock()
    info.language = language
    info.no_speech_prob = no_speech_prob
    model = MagicMock()
    model.transcribe.return_value = ([seg], info)
    return model


async def _run_one(engine: ASREngine, segment: SpeechSegment) -> Transcript | None:
    engine._input_q.put_nowait(segment)
    engine._input_q.put_nowait(None)  # shutdown sentinel
    await engine.run()
    return engine._output_q.get_nowait() if not engine._output_q.empty() else None


@pytest.fixture
def queues():
    return asyncio.Queue(), asyncio.Queue()


# ── happy path ────────────────────────────────────────────────────────────────

async def test_transcribes_segment(queues):
    input_q, output_q = queues
    engine = ASREngine(input_q, output_q)
    engine._model = _mock_model(text="안녕하세요", no_speech_prob=0.05, language="ko")

    result = await _run_one(engine, _make_segment())

    assert result is not None
    assert result.text == "안녕하세요"
    assert result.language == "ko"
    assert result.confidence == pytest.approx(0.95)
    assert result.latency_ms > 0


async def test_latency_ms_is_set(queues):
    input_q, output_q = queues
    engine = ASREngine(input_q, output_q)
    engine._model = _mock_model()

    result = await _run_one(engine, _make_segment())

    assert result is not None
    assert result.latency_ms >= 0.0


# ── drop conditions ───────────────────────────────────────────────────────────

async def test_drops_high_no_speech_prob(queues):
    input_q, output_q = queues
    engine = ASREngine(input_q, output_q)
    engine._model = _mock_model(text="some text", no_speech_prob=0.8)

    with metrics.override() as sink:
        result = await _run_one(engine, _make_segment())

    assert result is None
    assert any(e.name == metrics.ASR_DROP for e in sink.events)


async def test_drops_empty_text(queues):
    input_q, output_q = queues
    engine = ASREngine(input_q, output_q)
    engine._model = _mock_model(text="   ", no_speech_prob=0.05)

    with metrics.override() as sink:
        result = await _run_one(engine, _make_segment())

    assert result is None
    assert any(e.name == metrics.ASR_DROP for e in sink.events)


async def test_drops_filler_brackets(queues):
    input_q, output_q = queues
    engine = ASREngine(input_q, output_q)
    engine._model = _mock_model(text="[BLANK_AUDIO]", no_speech_prob=0.05)

    with metrics.override() as sink:
        result = await _run_one(engine, _make_segment())

    assert result is None
    assert any(e.name == metrics.ASR_DROP for e in sink.events)


# ── metrics ───────────────────────────────────────────────────────────────────

async def test_records_asr_span(queues):
    input_q, output_q = queues
    engine = ASREngine(input_q, output_q)
    engine._model = _mock_model()

    with metrics.override() as sink:
        await _run_one(engine, _make_segment())

    assert any(s.name == "asr.transcribe" for s in sink.spans)


# ── multi-segment ─────────────────────────────────────────────────────────────

async def test_processes_multiple_segments(queues):
    input_q, output_q = queues
    engine = ASREngine(input_q, output_q)
    engine._model = _mock_model(text="hello")

    for _ in range(3):
        input_q.put_nowait(_make_segment())
    input_q.put_nowait(None)
    await engine.run()

    results = []
    while not output_q.empty():
        results.append(output_q.get_nowait())
    assert len(results) == 3
