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

    has_hangul = any(_is_hangul(ch) for ch in sentence)
    if has_hangul:
        return "ko"

    has_ascii_letter = any(ch.isascii() and ch.isalpha() for ch in sentence)
    return "en" if has_ascii_letter else "ko"


# ── Mixed-script segmentation ────────────────────────────────────────────────
# A single sentence may interleave English and Korean ("The word for hello is
# 안녕하세요"). Cartesia synthesizes one language + one speed per request, so we
# split such sentences into runs of like script, render each in its own
# language, and — for Korean phrases embedded in an English sentence (a "help
# phrase") — at a slower speed so the learner can repeat it. Pure-Korean
# sentences (immersion mode) stay at normal speed.

SPEED_NORMAL = "normal"


def _is_hangul(ch: str) -> bool:
    # Syllables (가–힣) plus the compatibility Jamo block (ㄱ–ㅣ) so isolated
    # consonants/vowels a tutor might say aloud still count as Korean.
    return "가" <= ch <= "힣" or "ㄱ" <= ch <= "ㆎ"


def _classify(ch: str) -> str | None:
    """'ko' for Hangul, 'en' for ASCII letters, None for neutral (punctuation,
    digits, whitespace) — neutral chars never force a language boundary."""
    if _is_hangul(ch):
        return "ko"
    if ch.isascii() and ch.isalpha():
        return "en"
    return None


def split_script_runs(sentence: str) -> list[str]:
    """Split a sentence into consecutive same-script runs.

    A boundary is introduced only when a Korean letter meets a Latin letter (or
    vice-versa); punctuation, digits, and spaces attach to the current run so
    runs stay prosodically whole. Leading neutral text merges into the run that
    follows it.
    """
    runs: list[str] = []
    buf = ""
    script: str | None = None

    for ch in sentence:
        ch_script = _classify(ch)
        if ch_script is not None and script is not None and ch_script != script:
            runs.append(buf)
            buf = ch
            script = ch_script
        else:
            buf += ch
            if script is None and ch_script is not None:
                script = ch_script
    if buf:
        runs.append(buf)

    # A purely-neutral leading run (no letters of its own) belongs with the next.
    merged: list[str] = []
    for run in runs:
        if merged and not any(_classify(c) for c in merged[-1]):
            merged[-1] += run
        else:
            merged.append(run)
    return merged


def segment_for_tts(
    sentence: str, lang_code: str, korean_help_speed: str = "slow"
) -> list[tuple[str, str, str]]:
    """Break a sentence into (text, language, speed) synthesis units.

    Korean runs inside a *mixed* (English + Korean) sentence are treated as help
    phrases and get `korean_help_speed`; everything else is normal speed.
    """
    runs = [r for r in split_script_runs(sentence) if r.strip()]
    if not runs:
        return []

    languages = [resolve_language(r, lang_code) for r in runs]

    # A Korean run is a "help phrase" only when Korean is the *minority* script
    # of the sentence — i.e. a short Korean phrase embedded in English. That
    # keeps immersion sentences (Korean-primary, maybe one English loanword) at
    # normal speed while slowing genuine teaching phrases.
    ko_letters = sum(1 for ch in sentence if _is_hangul(ch))
    en_letters = sum(1 for ch in sentence if ch.isascii() and ch.isalpha())
    korean_is_help = "en" in languages and 0 < ko_letters < en_letters

    units: list[tuple[str, str, str]] = []
    for run, language in zip(runs, languages):
        speed = korean_help_speed if (language == "ko" and korean_is_help) else SPEED_NORMAL
        units.append((run, language, speed))
    return units
