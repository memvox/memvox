import asyncio
import time
import traceback
import uuid
from typing import AsyncIterator


def _log_task_exception(task: asyncio.Task) -> None:
    """Surface exceptions from the background _run task; otherwise they vanish."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        print(f"[orchestrator] _run task crashed: {exc!r}")
        traceback.print_exception(type(exc), exc, exc.__traceback__)

from memvox.observability import metrics
from memvox.session.types import SessionConfig
from memvox.voice.asr import ASREngine
from memvox.voice.egress import AudioEgressClient
from memvox.voice.ingress import AudioIngressClient
from memvox.voice.llm import LLMEngine
from memvox.voice.tts_base import TTSBackend
from memvox.voice.types import AudioChunk, CancelPlayback, SpeechSegment, SpeechStarted
from memvox.wiki.store import WikiStore
from memvox.wiki.types import ChatMessage, CompileRequest, ConversationTurn


# ASR language code → human name for the per-turn "reply in <lang>" directive.
# Only languages we explicitly support are steered; anything else lets the
# system prompt decide (avoids forcing a bad reply on a mis-detection).
_LANG_NAMES = {"en": "English", "ko": "Korean"}


class SessionOrchestrator:
    """Wire all components into a live turn-taking loop.

    Phase 1 (config.overlapping = False):
      Each turn runs sequentially: ASR → wiki search → LLM (collect all)
      → TTS → egress. The next segment is not read until the turn completes.

    Phase 2 (config.overlapping = True):
      TODO: concurrent _ingest_loop / _turn_loop / _tts_loop / _playback_loop
      with barge-in monitor. Toggled by the same flag, same class.
    """

    def __init__(
        self,
        config: SessionConfig,
        asr: ASREngine,
        llm: LLMEngine,
        tts: TTSBackend,
        wiki: WikiStore,
        ingress: AudioIngressClient,
        egress: AudioEgressClient,
        ui_bridge=None,
    ) -> None:
        self._config = config
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._wiki = wiki
        self._ingress = ingress
        self._egress = egress
        self._ui = ui_bridge

        self._history: list[ChatMessage] = []
        self._turns: list[ConversationTurn] = []
        self._session_id: str = ""
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self._asr.initialize()
        await self._tts.initialize()
        await self._ingress.connect()
        await self._egress.connect()

        self._session_id = uuid.uuid4().hex
        self._stop_event.clear()
        metrics.event(metrics.SESSION_START, session_id=self._session_id)
        self._emit_ui({"type": "hello", "session_id": self._session_id})

        self._task = asyncio.create_task(self._run())
        self._task.add_done_callback(_log_task_exception)

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        await self._ingress.close()
        await self._egress.close()

        metrics.event(metrics.SESSION_END, session_id=self._session_id)
        self._emit_ui({"type": "session_end", "session_id": self._session_id})

        # Print latency summary if the active sink is the in-memory one.
        report = metrics.summary()
        if report:
            print(report)

        # Fire WikiCompiler in background — does not block session teardown.
        # TODO Phase 3: wire in WikiCompiler
        # existing_slugs = [a.slug for a in await self._wiki.list_articles()]
        # req = CompileRequest(self._session_id, self._turns, existing_slugs)
        # asyncio.create_task(wiki_compiler.compile(req))

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            msg = await self._ingress.recv()
            if msg is None:
                break
            if isinstance(msg, SpeechStarted):
                # Phase 2: trigger LLM KV-cache pre-warm here.
                pass
            elif isinstance(msg, SpeechSegment):
                await self._process_segment(msg)

    # ── Turn pipeline ─────────────────────────────────────────────────────────

    async def _process_segment(self, segment: SpeechSegment) -> None:
        # mouth_to_ear stopwatch starts at end-of-utterance — i.e., the moment
        # the VAD has decided the user finished speaking and handed us the
        # SpeechSegment. We stop it when the first non-final AudioChunk leaves
        # egress (first audio that the user could hear).
        t_segment_received = time.monotonic()
        print(f"[turn] SpeechSegment received: {segment.duration_ms:.0f}ms")

        transcript = await self._asr.transcribe(segment)
        if transcript is None:
            print("[turn] ASR returned None — dropped (likely no-speech or low confidence)")
            return
        print(f"[turn] ASR transcript: {transcript.text!r}")

        turn_id = uuid.uuid4().hex[:8]
        self._emit_ui({
            "type": "user_final",
            "turn_id": turn_id,
            "text": transcript.text,
            "language": transcript.language,
        })

        # Wiki retrieval (target: <50ms)
        async with metrics.span(metrics.WIKI_QUERY, turn_id=turn_id):
            results = await self._wiki.search(transcript.text, top_k=5)
        snippets = [chunk for r in results for chunk in r.matched_chunks[:1]]
        print(f"[turn] wiki: {len(snippets)} snippet(s)")

        # Build message list — system prompt + bounded history + new user turn.
        # Steer the reply language from ASR's detection instead of letting the
        # model guess from bare text (a one-word "hello" otherwise falls to the
        # prompt's "Korean is default when ambiguous" rule). The directive is
        # transient — placed last for recency, and NOT stored in history.
        messages = [
            ChatMessage(role="system", content=self._config.system_prompt),
            *self._history,
        ]
        lang_name = _LANG_NAMES.get(transcript.language)
        if lang_name:
            messages.append(
                ChatMessage(
                    role="system",
                    content=f"(Detected input language: {lang_name}. "
                            f"Apply your current-mode language rules.)",
                )
            )
        messages.append(ChatMessage(role="user", content=transcript.text))

        from memvox.voice.types import GenerationRequest
        request = GenerationRequest(
            messages=messages,
            context_snippets=snippets,
            session_id=self._session_id,
            turn_id=turn_id,
            thinking_enabled=self._config.thinking_enabled,
        )

        # Phase 1: collect full LLM response before TTS starts
        content_parts: list[str] = []

        async def _sequential_tokens() -> AsyncIterator[str]:
            async for chunk in self._llm.generate(request):
                if not chunk.is_thinking and not chunk.is_final:
                    content_parts.append(chunk.text)
            assistant_text_local = "".join(content_parts)
            print(f"[turn] LLM reply: {assistant_text_local!r}")
            for part in content_parts:
                yield part

        # TTS synthesis → egress
        first_audio_recorded = False
        tts_chunks = 0
        last_ui_sentence = ""
        async for audio_chunk in self._tts.synthesize(_sequential_tokens()):
            if not audio_chunk.is_final:
                tts_chunks += 1
                await self._egress.send(audio_chunk)
                if not first_audio_recorded:
                    first_audio_recorded = True
                    metrics.event(
                        metrics.MOUTH_TO_EAR,
                        latency_ms=(time.monotonic() - t_segment_received) * 1000,
                        turn_id=turn_id,
                    )
                # Mirror the agent's words to the web UI as each sentence (or
                # script run) starts playing. Chunks repeat sentence_text, so
                # only emit on change.
                sentence = " ".join(audio_chunk.sentence_text.split())
                if sentence and sentence != last_ui_sentence:
                    last_ui_sentence = sentence
                    self._emit_ui({
                        "type": "assistant_sentence",
                        "turn_id": turn_id,
                        "text": sentence,
                    })
        print(f"[turn] TTS produced {tts_chunks} chunks → egress")

        # Record turn in history
        assistant_text = "".join(content_parts)
        self._emit_ui({
            "type": "assistant_final",
            "turn_id": turn_id,
            "text": " ".join(assistant_text.split()),
        })
        self._append_history(transcript.text, assistant_text, turn_id)

    def _emit_ui(self, event: dict) -> None:
        """Forward an event to the web UI bridge, if one is attached."""
        if self._ui is not None:
            self._ui.emit(event)

    # ── History management ────────────────────────────────────────────────────

    def _append_history(
        self, user_text: str, assistant_text: str, turn_id: str
    ) -> None:
        from datetime import datetime, timezone

        user_msg      = ChatMessage(role="user",      content=user_text)
        assistant_msg = ChatMessage(role="assistant", content=assistant_text)

        self._history.append(user_msg)
        self._history.append(assistant_msg)

        self._turns.append(ConversationTurn(
            turn_id=turn_id,
            user_message=user_msg,
            assistant_message=assistant_msg,
            timestamp=datetime.now(timezone.utc),
        ))

        # Cap at history_max_turns full turns (2 messages per turn)
        max_msgs = self._config.history_max_turns * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]
