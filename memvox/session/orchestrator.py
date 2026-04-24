import asyncio
import uuid
from typing import AsyncIterator

from memvox.observability import metrics
from memvox.session.types import SessionConfig
from memvox.voice.asr import ASREngine
from memvox.voice.egress import AudioEgressClient
from memvox.voice.ingress import AudioIngressClient
from memvox.voice.llm import LLMEngine
from memvox.voice.tts import TTSEngine
from memvox.voice.types import AudioChunk, CancelPlayback, SpeechSegment, SpeechStarted
from memvox.wiki.store import WikiStore
from memvox.wiki.types import ChatMessage, CompileRequest, ConversationTurn


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
        tts: TTSEngine,
        wiki: WikiStore,
        ingress: AudioIngressClient,
        egress: AudioEgressClient,
    ) -> None:
        self._config = config
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._wiki = wiki
        self._ingress = ingress
        self._egress = egress

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

        self._task = asyncio.create_task(self._run())

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
        transcript = await self._asr.transcribe(segment)
        if transcript is None:
            return

        turn_id = uuid.uuid4().hex[:8]

        # Wiki retrieval (target: <50ms)
        async with metrics.span(metrics.WIKI_QUERY, turn_id=turn_id):
            results = await self._wiki.search(transcript.text, top_k=5)
        snippets = [chunk for r in results for chunk in r.matched_chunks[:1]]

        # Build message list — system prompt + bounded history + new user turn
        messages = [
            ChatMessage(role="system", content=self._config.system_prompt),
            *self._history,
            ChatMessage(role="user", content=transcript.text),
        ]

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
            for part in content_parts:
                yield part

        # TTS synthesis → egress
        async for audio_chunk in self._tts.synthesize(_sequential_tokens()):
            if not audio_chunk.is_final:
                await self._egress.send(audio_chunk)

        # Record turn in history
        assistant_text = "".join(content_parts)
        self._append_history(transcript.text, assistant_text, turn_id)

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
