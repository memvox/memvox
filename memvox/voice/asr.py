import asyncio
import time

from faster_whisper import WhisperModel

from memvox.observability import metrics
from memvox.voice.types import SpeechSegment, Transcript

_NO_SPEECH_THRESHOLD = 0.6
_FILLER_PREFIXES = ("...", "…", "[", "(")


def _is_filler(text: str) -> bool:
    t = text.strip()
    return not t or any(t.startswith(p) for p in _FILLER_PREFIXES)


class ASREngine:
    """SpeechSegment → Transcript via faster-whisper.

    Owns a single WhisperModel instance. Reads from `input_q`, writes to
    `output_q`. Drops segments where no_speech_prob > 0.6 or text is empty.
    """

    def __init__(
        self,
        input_q: asyncio.Queue[SpeechSegment],
        output_q: asyncio.Queue[Transcript],
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self._input_q = input_q
        self._output_q = output_q
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._model: WhisperModel | None = None

    async def initialize(self) -> None:
        """Load the Whisper model (runs in a thread — blocks until ready)."""
        self._model = await asyncio.to_thread(
            WhisperModel, self._model_name, device=self._device, compute_type=self._compute_type
        )

    async def transcribe(self, segment: SpeechSegment) -> Transcript | None:
        """Transcribe one segment directly. Returns None if dropped.

        Used by SessionOrchestrator in sequential (Phase 1) mode.
        """
        if self._model is None:
            raise RuntimeError("call initialize() before transcribe()")
        async with metrics.span("asr.transcribe"):
            t0 = time.monotonic()
            result = await asyncio.to_thread(self._transcribe, segment)
            latency_ms = (time.monotonic() - t0) * 1000

        if result is None:
            metrics.event(metrics.ASR_DROP)
            return None

        result.latency_ms = latency_ms
        return result

    async def run(self) -> None:
        """Queue-based loop — consumes SpeechSegments until None sentinel."""
        if self._model is None:
            raise RuntimeError("call initialize() before run()")
        while True:
            segment = await self._input_q.get()
            if segment is None:
                break
            transcript = await self.transcribe(segment)
            if transcript is not None:
                await self._output_q.put(transcript)

    def _transcribe(self, segment: SpeechSegment) -> Transcript | None:
        """Blocking transcription — called via asyncio.to_thread."""
        import numpy as np

        # faster-whisper expects float32 numpy array in [-1, 1]
        audio = np.frombuffer(segment.audio, dtype=np.int16).astype(np.float32) / 32768.0

        segs, info = self._model.transcribe(
            audio,
            language=None,          # auto-detect
            beam_size=5,
            vad_filter=False,       # VAD already handled by AudioIngress
        )
        # faster-whisper returns a generator; consume it
        text = " ".join(s.text for s in segs).strip()

        if info.no_speech_prob > _NO_SPEECH_THRESHOLD or _is_filler(text):
            return None

        return Transcript(
            text=text,
            language=info.language,
            confidence=1.0 - info.no_speech_prob,
            no_speech_prob=info.no_speech_prob,
            latency_ms=0.0,         # filled in by _process
            source_segment=segment,
        )
