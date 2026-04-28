import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from memvox.observability import metrics
from memvox.session.orchestrator import SessionOrchestrator
from memvox.session.types import SessionConfig
from memvox.voice.types import AudioChunk, SpeechSegment, SpeechStarted, Transcript, TokenChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _config(**overrides) -> SessionConfig:
    defaults = dict(
        system_prompt="You are a Korean tutor.",
        language="ko",
        voice="Ana Florence",
    )
    defaults.update(overrides)
    return SessionConfig(**defaults)


def _segment() -> SpeechSegment:
    audio = np.zeros(16_000, dtype=np.int16).tobytes()
    return SpeechSegment(audio=audio, speech_prob=0.95, duration_ms=1000.0, timestamp_start=0.0)


def _transcript(text: str = "안녕하세요") -> Transcript:
    return Transcript(
        text=text,
        language="ko",
        confidence=0.95,
        no_speech_prob=0.05,
        latency_ms=50.0,
        source_segment=_segment(),
    )


def _audio_chunk(is_final: bool = False) -> AudioChunk:
    return AudioChunk(
        pcm_bytes=np.zeros(100, dtype=np.float32).tobytes(),
        sample_rate=24_000,
        is_final=is_final,
        sentence_text="안녕하세요",
        chunk_latency_ms=10.0,
    )


async def _token_stream(*texts: str):
    for t in texts:
        yield TokenChunk(text=t, is_thinking=False, is_final=False, turn_id="t1")
    yield TokenChunk(text="", is_thinking=False, is_final=True, turn_id="t1")


async def _audio_stream(*content_chunks: AudioChunk):
    for c in content_chunks:
        yield c
    yield AudioChunk(pcm_bytes=b"", sample_rate=24_000, is_final=True, sentence_text="", chunk_latency_ms=0.0)


def _make_orchestrator(config: SessionConfig | None = None):
    config = config or _config()

    asr = MagicMock()
    asr.initialize = AsyncMock()
    asr.transcribe  = AsyncMock(return_value=_transcript())

    llm = MagicMock()
    llm.generate = MagicMock(side_effect=lambda *_: _token_stream("안녕하세요!"))

    tts = MagicMock()
    tts.initialize = AsyncMock()
    tts.synthesize  = MagicMock(side_effect=lambda *_: _audio_stream(_audio_chunk()))

    wiki = MagicMock()
    wiki.search = AsyncMock(return_value=[])

    ingress = MagicMock()
    ingress.connect = AsyncMock()
    ingress.recv    = AsyncMock(return_value=None)
    ingress.close   = AsyncMock()

    egress = MagicMock()
    egress.connect = AsyncMock()
    egress.send    = AsyncMock()
    egress.close   = AsyncMock()

    orch = SessionOrchestrator(config, asr, llm, tts, wiki, ingress, egress)
    return orch, asr, llm, tts, wiki, ingress, egress


# ── Lifecycle ─────────────────────────────────────────────────────────────────

class TestSessionLifecycle:
    async def test_start_emits_session_start(self):
        orch, _, _, _, _, _, _ = _make_orchestrator()
        with metrics.override() as sink:
            await orch.start()
        assert any(e.name == metrics.SESSION_START for e in sink.events)

    async def test_stop_emits_session_end(self):
        orch, _, _, _, _, _, _ = _make_orchestrator()
        await orch.start()
        await asyncio.sleep(0)
        with metrics.override() as sink:
            await orch.stop()
        assert any(e.name == metrics.SESSION_END for e in sink.events)

    async def test_start_initializes_asr_and_tts(self):
        orch, asr, _, tts, _, _, _ = _make_orchestrator()
        await orch.start()
        await orch.stop()
        asr.initialize.assert_called_once()
        tts.initialize.assert_called_once()

    async def test_start_connects_ingress_and_egress(self):
        orch, _, _, _, _, ingress, egress = _make_orchestrator()
        await orch.start()
        await orch.stop()
        ingress.connect.assert_called_once()
        egress.connect.assert_called_once()

    async def test_stop_closes_ingress_and_egress(self):
        orch, _, _, _, _, ingress, egress = _make_orchestrator()
        await orch.start()
        await asyncio.sleep(0)
        await orch.stop()
        ingress.close.assert_called_once()
        egress.close.assert_called_once()


# ── _run loop ─────────────────────────────────────────────────────────────────

class TestRunLoop:
    async def test_speech_started_does_not_crash(self):
        orch, _, _, _, _, ingress, _ = _make_orchestrator()
        ingress.recv = AsyncMock(side_effect=[SpeechStarted(timestamp=0.0), None])
        await orch.start()
        await orch._task
        await orch.stop()

    async def test_speech_segment_triggers_asr(self):
        orch, asr, _, _, _, ingress, _ = _make_orchestrator()
        ingress.recv = AsyncMock(side_effect=[_segment(), None])
        await orch.start()
        await orch._task
        await orch.stop()
        asr.transcribe.assert_called_once()


# ── _process_segment ──────────────────────────────────────────────────────────

class TestProcessSegment:
    async def test_asr_drop_skips_everything(self):
        orch, asr, _, _, _, _, egress = _make_orchestrator()
        asr.transcribe = AsyncMock(return_value=None)
        await orch._process_segment(_segment())
        egress.send.assert_not_called()

    async def test_non_final_chunk_sent_to_egress(self):
        orch, _, _, _, _, _, egress = _make_orchestrator()
        await orch._process_segment(_segment())
        egress.send.assert_called_once()
        sent = egress.send.call_args[0][0]
        assert not sent.is_final

    async def test_final_chunk_not_sent_to_egress(self):
        orch, _, _, tts, _, _, egress = _make_orchestrator()
        tts.synthesize = MagicMock(
            side_effect=lambda *_: _audio_stream(_audio_chunk(is_final=False))
        )
        await orch._process_segment(_segment())
        for call in egress.send.call_args_list:
            assert not call[0][0].is_final

    async def test_multiple_audio_chunks_all_sent(self):
        orch, _, _, tts, _, _, egress = _make_orchestrator()
        tts.synthesize = MagicMock(
            side_effect=lambda *_: _audio_stream(_audio_chunk(), _audio_chunk())
        )
        await orch._process_segment(_segment())
        assert egress.send.call_count == 2

    async def test_wiki_search_called_with_transcript_text(self):
        orch, _, _, _, wiki, _, _ = _make_orchestrator()
        await orch._process_segment(_segment())
        wiki.search.assert_called_once_with("안녕하세요", top_k=5)

    async def test_llm_fully_consumed_before_tts_receives_tokens(self):
        """Phase 1 invariant: TTS only starts after all LLM tokens are collected."""
        calls: list[str] = []

        async def _tracking_llm(_):
            calls.append("llm_start")
            yield TokenChunk(text="hello", is_thinking=False, is_final=False, turn_id="t1")
            yield TokenChunk(text="", is_thinking=False, is_final=True, turn_id="t1")
            calls.append("llm_end")

        async def _tracking_tts(tokens):
            async for _ in tokens:
                pass
            calls.append("tts_tokens_consumed")
            yield _audio_chunk()
            yield _audio_chunk(is_final=True)

        orch, _, llm, tts, _, _, _ = _make_orchestrator()
        llm.generate  = MagicMock(side_effect=_tracking_llm)
        tts.synthesize = MagicMock(side_effect=_tracking_tts)

        await orch._process_segment(_segment())

        assert "llm_end" in calls
        assert "tts_tokens_consumed" in calls
        assert calls.index("llm_end") < calls.index("tts_tokens_consumed")

    async def test_thinking_chunks_excluded_from_history(self):
        orch, _, llm, tts, _, _, _ = _make_orchestrator()

        async def _mixed_stream(_):
            yield TokenChunk(text="<thought>", is_thinking=True,  is_final=False, turn_id="t1")
            yield TokenChunk(text="reply",     is_thinking=False, is_final=False, turn_id="t1")
            yield TokenChunk(text="",          is_thinking=False, is_final=True,  turn_id="t1")

        async def _consuming_tts(tokens):
            async for _ in tokens:  # must consume tokens so content_parts is populated
                pass
            yield _audio_chunk()
            yield _audio_chunk(is_final=True)

        llm.generate  = MagicMock(side_effect=_mixed_stream)
        tts.synthesize = MagicMock(side_effect=_consuming_tts)
        await orch._process_segment(_segment())
        assert orch._history[-1].content == "reply"


# ── History management ────────────────────────────────────────────────────────

class TestHistory:
    async def test_history_empty_before_any_turn(self):
        orch, _, _, _, _, _, _ = _make_orchestrator()
        assert orch._history == []

    async def test_history_grows_two_messages_per_turn(self):
        orch, _, _, _, _, _, _ = _make_orchestrator()
        await orch._process_segment(_segment())
        assert len(orch._history) == 2

    async def test_history_roles(self):
        orch, _, _, _, _, _, _ = _make_orchestrator()
        await orch._process_segment(_segment())
        assert orch._history[0].role == "user"
        assert orch._history[1].role == "assistant"

    async def test_history_capped_at_max_turns(self):
        config = _config(history_max_turns=2)
        orch, _, _, _, _, _, _ = _make_orchestrator(config=config)
        for _ in range(5):
            await orch._process_segment(_segment())
        assert len(orch._history) == 4  # 2 turns × 2 messages

    async def test_turns_list_grows_with_each_turn(self):
        orch, _, _, _, _, _, _ = _make_orchestrator()
        for _ in range(3):
            await orch._process_segment(_segment())
        assert len(orch._turns) == 3

    async def test_turn_records_correct_user_text(self):
        orch, asr, _, _, _, _, _ = _make_orchestrator()
        asr.transcribe = AsyncMock(return_value=_transcript(text="한국어"))
        await orch._process_segment(_segment())
        assert orch._turns[0].user_message.content == "한국어"

    async def test_history_not_grown_on_asr_drop(self):
        orch, asr, _, _, _, _, _ = _make_orchestrator()
        asr.transcribe = AsyncMock(return_value=None)
        await orch._process_segment(_segment())
        assert orch._history == []


# ── Wiki metrics ──────────────────────────────────────────────────────────────

class TestMetrics:
    async def test_wiki_query_span_emitted(self):
        orch, _, _, _, _, _, _ = _make_orchestrator()
        with metrics.override() as sink:
            await orch._process_segment(_segment())
        assert any(s.name == metrics.WIKI_QUERY for s in sink.spans)
