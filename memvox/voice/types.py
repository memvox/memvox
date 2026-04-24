from dataclasses import dataclass, field

from memvox.wiki.types import ChatMessage


# ── ASR pipeline ──────────────────────────────────────────────────────────────

@dataclass
class SpeechSegment:
    audio: bytes           # PCM 16 kHz mono int16
    speech_prob: float
    duration_ms: float
    timestamp_start: float


@dataclass
class Transcript:
    text: str
    language: str          # ISO 639-1
    confidence: float
    no_speech_prob: float
    latency_ms: float
    source_segment: SpeechSegment


# ── LLM pipeline ──────────────────────────────────────────────────────────────

@dataclass
class GenerationRequest:
    messages: list[ChatMessage]
    context_snippets: list[str]
    session_id: str
    turn_id: str
    thinking_enabled: bool = False


@dataclass
class TokenChunk:
    text: str
    is_thinking: bool
    is_final: bool
    turn_id: str
    ttft_ms: float | None = None  # set on first non-thinking token


# ── TTS pipeline ──────────────────────────────────────────────────────────────

@dataclass
class AudioChunk:
    pcm_bytes: bytes       # 24 kHz float32 little-endian
    sample_rate: int
    is_final: bool
    sentence_text: str
    chunk_latency_ms: float


# ── VAD signals ───────────────────────────────────────────────────────────────

@dataclass
class SpeechStarted:
    timestamp: float


@dataclass
class SpeechEnded:
    timestamp: float


# ── AudioEgress control ───────────────────────────────────────────────────────

@dataclass
class CancelPlayback:
    pass


@dataclass
class PausePlayback:
    pass


@dataclass
class ResumePlayback:
    pass


# ── AudioEgress output signals ────────────────────────────────────────────────

@dataclass
class PlaybackStarted:
    timestamp: float  # first sample emitted — used for end-to-end TTFA


@dataclass
class PlaybackFinished:
    pass


@dataclass
class PlaybackCancelled:
    pass
