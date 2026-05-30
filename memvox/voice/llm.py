import time
from typing import AsyncGenerator

from openai import AsyncOpenAI

from memvox.observability import metrics
from memvox.voice.types import GenerationRequest, TokenChunk

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


class _ThinkingParser:
    """Stream-safe parser for Qwen3 <think>...</think> blocks.

    Holds back only the minimum suffix needed to catch tags that span chunk
    boundaries — typically 0 characters for normal content tokens.
    """

    def __init__(self) -> None:
        self._thinking = False
        self._buf = ""

    def feed(self, text: str) -> list[tuple[str, bool]]:
        combined = self._buf + text
        segments: list[tuple[str, bool]] = []

        while combined:
            tag = _CLOSE_TAG if self._thinking else _OPEN_TAG
            idx = combined.find(tag)

            if idx == -1:
                # Hold back only if the tail of combined looks like a tag prefix
                for hold in range(len(tag) - 1, 0, -1):
                    if combined.endswith(tag[:hold]):
                        if combined[:-hold]:
                            segments.append((combined[:-hold], self._thinking))
                        self._buf = combined[-hold:]
                        break
                else:
                    segments.append((combined, self._thinking))
                    self._buf = ""
                break

            if idx > 0:
                segments.append((combined[:idx], self._thinking))
            self._thinking = not self._thinking
            combined = combined[idx + len(tag):]

        return [(t, th) for t, th in segments if t]

    def flush(self) -> list[tuple[str, bool]]:
        """Emit any held-back bytes at end of stream."""
        result = [(self._buf, self._thinking)] if self._buf else []
        self._buf = ""
        return [(t, th) for t, th in result if t]


def _append_no_think(messages: list[dict]) -> list[dict]:
    """Append Qwen3's `/no_think` soft-switch to the last user message.

    Qwen3 honors `/no_think` at the end of a user turn to disable thinking
    for that response. Works on vLLM and Ollama identically.
    """
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i]["content"] = f"{out[i]['content']}\n\n/no_think"
            break
    return out


def _build_messages(request: GenerationRequest) -> list[dict]:
    """Merge history + wiki context into the OpenAI messages list.

    Context snippets are appended to the existing system message, or prepended
    as a new system message if none is present.
    """
    if not request.context_snippets:
        return [{"role": m.role, "content": m.content} for m in request.messages]

    context_block = "\n\n## Relevant knowledge\n\n" + "\n\n---\n\n".join(
        request.context_snippets
    )
    msgs: list[dict] = []
    injected = False

    for m in request.messages:
        content = m.content
        if m.role == "system" and not injected:
            content += context_block
            injected = True
        msgs.append({"role": m.role, "content": content})

    if not injected:
        msgs.insert(0, {"role": "system", "content": context_block.lstrip()})

    return msgs


class LLMEngine:
    """Messages + wiki context → streaming TokenChunk async generator.

    Stateless: caller (SessionOrchestrator) passes the full message history
    on every call. Cancellable: call aclose() on the returned generator.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-8B-Instruct",
        api_key: str = "not-needed",
        _client: AsyncOpenAI | None = None,  # injectable for tests
    ) -> None:
        self._client = _client or AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def generate(
        self, request: GenerationRequest
    ) -> AsyncGenerator[TokenChunk, None]:
        messages = _build_messages(request)
        t_start = time.monotonic()
        ttft_recorded = False
        parser = _ThinkingParser()

        # Two ways to control Qwen3 thinking, sent together so the same code
        # works against vLLM and Ollama:
        #   1. `chat_template_kwargs.enable_thinking` — vLLM honors this via
        #      the Jinja chat template. Ollama ignores unknown extras.
        #   2. `/no_think` inline marker on the last user message — Qwen3's
        #      documented soft switch, works regardless of serving stack.
        # Without an explicit opt-out, Qwen3 defaults to thinking ON; the
        # think tokens fill `max_tokens` before any content appears and our
        # parser drops them, leaving nothing for TTS.
        extra_body = {"chat_template_kwargs": {"enable_thinking": request.thinking_enabled}}
        if not request.thinking_enabled:
            messages = _append_no_think(messages)

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            extra_body=extra_body,
            max_tokens=120,    # hard cap; keeps replies short for live conversation
        )

        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue

                for text, is_thinking in parser.feed(delta):
                    ttft_ms = None
                    if not is_thinking and not ttft_recorded:
                        ttft_ms = (time.monotonic() - t_start) * 1000
                        ttft_recorded = True
                        metrics.event(
                            metrics.LLM_TTFT,
                            latency_ms=ttft_ms,
                            turn_id=request.turn_id,
                        )
                    yield TokenChunk(
                        text=text,
                        is_thinking=is_thinking,
                        is_final=False,
                        turn_id=request.turn_id,
                        ttft_ms=ttft_ms,
                    )

            for text, is_thinking in parser.flush():
                yield TokenChunk(
                    text=text,
                    is_thinking=is_thinking,
                    is_final=False,
                    turn_id=request.turn_id,
                    ttft_ms=None,
                )

            yield TokenChunk(
                text="", is_thinking=False, is_final=True,
                turn_id=request.turn_id, ttft_ms=None,
            )
        finally:
            await stream.close()
