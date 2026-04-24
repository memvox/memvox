from unittest.mock import AsyncMock, MagicMock

import pytest

from memvox.observability import metrics
from memvox.voice.llm import LLMEngine, _ThinkingParser, _build_messages
from memvox.voice.types import GenerationRequest, TokenChunk
from memvox.wiki.types import ChatMessage


# ── Helpers ───────────────────────────────────────────────────────────────────

def _request(
    messages: list[ChatMessage] | None = None,
    context_snippets: list[str] | None = None,
    thinking_enabled: bool = False,
) -> GenerationRequest:
    return GenerationRequest(
        messages=messages or [
            ChatMessage(role="system", content="You are a Korean tutor."),
            ChatMessage(role="user", content="안녕하세요"),
        ],
        context_snippets=context_snippets or [],
        session_id="s1",
        turn_id="t1",
        thinking_enabled=thinking_enabled,
    )


def _fake_stream(*chunks: str) -> MagicMock:
    """Build a mock AsyncOpenAI stream from a list of content strings."""

    async def _aiter():
        for content in chunks:
            chunk = MagicMock()
            chunk.choices[0].delta.content = content
            yield chunk

    stream = MagicMock()
    stream.__aiter__ = lambda _: _aiter()
    stream.close = AsyncMock()
    return stream


def _engine(*chunks: str) -> LLMEngine:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_stream(*chunks))
    return LLMEngine(_client=client)


async def _collect(engine: LLMEngine, request: GenerationRequest) -> list[TokenChunk]:
    return [chunk async for chunk in engine.generate(request)]


# ── Basic streaming ───────────────────────────────────────────────────────────

async def test_yields_token_chunks():
    chunks = await _collect(_engine("Hello", " world", "!"), _request())
    texts = [c.text for c in chunks if not c.is_final]
    assert texts == ["Hello", " world", "!"]


async def test_final_chunk_is_last():
    chunks = await _collect(_engine("hi"), _request())
    assert chunks[-1].is_final
    assert chunks[-1].text == ""


async def test_turn_id_propagated():
    chunks = await _collect(_engine("hi"), _request())
    assert all(c.turn_id == "t1" for c in chunks)


async def test_empty_delta_skipped():
    chunks = await _collect(_engine("", "hello", ""), _request())
    texts = [c.text for c in chunks if not c.is_final]
    assert texts == ["hello"]


# ── TTFT recording ────────────────────────────────────────────────────────────

async def test_ttft_set_on_first_content_chunk():
    chunks = await _collect(_engine("Hello", " world"), _request())
    content_chunks = [c for c in chunks if not c.is_final and not c.is_thinking]
    assert content_chunks[0].ttft_ms is not None
    assert content_chunks[0].ttft_ms >= 0
    # Only the first chunk carries ttft_ms
    assert all(c.ttft_ms is None for c in content_chunks[1:])


async def test_ttft_emits_metric():
    with metrics.override() as sink:
        await _collect(_engine("hello"), _request())
    ttft_events = [e for e in sink.events if e.name == metrics.LLM_TTFT]
    assert len(ttft_events) == 1
    assert ttft_events[0].latency_ms is not None


# ── Thinking tag parsing ──────────────────────────────────────────────────────

async def test_thinking_chunks_marked():
    chunks = await _collect(
        _engine("<think>hmm</think>answer"),
        _request(thinking_enabled=True),
    )
    thinking = [c for c in chunks if c.is_thinking]
    content = [c for c in chunks if not c.is_thinking and not c.is_final]
    assert any("hmm" in c.text for c in thinking)
    assert any("answer" in c.text for c in content)


async def test_thinking_not_counted_for_ttft():
    chunks = await _collect(
        _engine("<think>deliberating</think>response"),
        _request(thinking_enabled=True),
    )
    # ttft_ms must be None on thinking chunks
    thinking = [c for c in chunks if c.is_thinking]
    assert all(c.ttft_ms is None for c in thinking)
    # ttft_ms must be set on the first content chunk
    content = [c for c in chunks if not c.is_thinking and not c.is_final]
    assert content[0].ttft_ms is not None


async def test_thinking_tag_spanning_chunks():
    # "<think>" split across two chunks
    chunks = await _collect(_engine("<thi", "nk>thought</think>hi"), _request())
    thinking = [c for c in chunks if c.is_thinking]
    assert any("thought" in c.text for c in thinking)


# ── Context injection ─────────────────────────────────────────────────────────

async def test_context_appended_to_system_message():
    req = _request(context_snippets=["vocab: 안녕 = hello", "grammar: subject + 은/는"])
    msgs = _build_messages(req)
    system = next(m for m in msgs if m["role"] == "system")
    assert "Relevant knowledge" in system["content"]
    assert "안녕 = hello" in system["content"]
    assert "grammar" in system["content"]


async def test_context_prepended_when_no_system_message():
    req = GenerationRequest(
        messages=[ChatMessage(role="user", content="hi")],
        context_snippets=["fact: Korea is in East Asia"],
        session_id="s1",
        turn_id="t1",
    )
    msgs = _build_messages(req)
    assert msgs[0]["role"] == "system"
    assert "Korea" in msgs[0]["content"]


async def test_no_context_leaves_messages_unchanged():
    req = _request(context_snippets=[])
    msgs = _build_messages(req)
    assert msgs[0]["content"] == "You are a Korean tutor."


# ── Cancellation ─────────────────────────────────────────────────────────────

async def test_cancellation_closes_stream():
    stream = _fake_stream("tok1", "tok2", "tok3", "tok4", "tok5")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=stream)
    engine = LLMEngine(_client=client)

    gen = engine.generate(_request())
    await gen.__anext__()  # consume one chunk
    await gen.aclose()     # cancel mid-stream

    stream.close.assert_awaited_once()


# ── _ThinkingParser unit tests ────────────────────────────────────────────────

def test_parser_passthrough():
    p = _ThinkingParser()
    assert p.feed("hello world") == [("hello world", False)]


def test_parser_full_think_block():
    p = _ThinkingParser()
    out = p.feed("<think>reason</think>answer")
    assert ("reason", True) in out
    assert ("answer", False) in out


def test_parser_split_open_tag():
    p = _ThinkingParser()
    out1 = p.feed("<thi")
    out2 = p.feed("nk>thought</think>done")
    combined = out1 + out2
    assert any(t == "thought" and th is True for t, th in combined)
    assert any(t == "done" and th is False for t, th in combined)


def test_parser_flush_emits_remainder():
    p = _ThinkingParser()
    p.feed("<thi")           # partial tag held in buffer
    result = p.flush()
    assert result == [("<thi", False)]
