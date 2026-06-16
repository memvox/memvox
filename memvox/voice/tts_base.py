"""Shared TTS plumbing: the backend Protocol, sentence accumulation, and
per-sentence language resolution.

A TTS backend turns a stream of LLM tokens into a stream of AudioChunks. The
orchestrator depends only on this Protocol, so swapping XTTS ↔ Cartesia ↔ any
future backend is a one-line factory change (see tts_factory.build_tts).
"""

from typing import AsyncGenerator, AsyncIterator, Protocol, runtime_checkable

from memvox.voice.types import AudioChunk

# Every backend emits float32 little-endian PCM; the Rust egress resamples to
# the hardware rate. 24 kHz matches XTTS-v2 and is a clean Cartesia output rate.
SAMPLE_RATE = 24_000


@runtime_checkable
class TTSBackend(Protocol):
    """Token stream → AudioChunk stream.

    Implementations must be safe to call once `initialize()` has completed.
    `synthesize` yields non-final AudioChunks as audio becomes available and a
    single final (empty, is_final=True) chunk to mark end-of-utterance.
    """

    async def initialize(self) -> None:
        ...

    def synthesize(
        self, tokens: AsyncIterator[str]
    ) -> AsyncGenerator[AudioChunk, None]:
        ...


# ── Sentence accumulation ───────────────────────────────────────────────────
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


def new_accumulator(flush_tokens: int = 30):
    if _HAS_RUST:
        return _RustSentenceAccumulator(flush_tokens=flush_tokens)
    return _PySentenceAccumulator(flush_tokens=flush_tokens)


# ── Language resolution ─────────────────────────────────────────────────────

def resolve_language(sentence: str, lang_code: str) -> str:
    """Pick the synthesis language for one sentence.

    For a Korean session the user may code-switch into English (e.g. when stuck).
    XTTS and Cartesia both mispronounce ASCII text read as Korean, so when the
    configured language is Korean we route pure-ASCII sentences to English and
    anything containing Hangul (or no Latin letters) to Korean. Non-Korean
    sessions always use the configured code unchanged.
    """
    if lang_code != "ko":
        return lang_code

    has_hangul = any("가" <= ch <= "힣" for ch in sentence)
    if has_hangul:
        return "ko"

    has_ascii_letter = any(ch.isascii() and ch.isalpha() for ch in sentence)
    return "en" if has_ascii_letter else "ko"
