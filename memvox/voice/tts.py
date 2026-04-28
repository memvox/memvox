import asyncio
from collections.abc import Iterable
import importlib.util
from pathlib import Path
import time
from typing import Any, AsyncGenerator, AsyncIterator

import numpy as np

from memvox.observability import metrics
from memvox.voice.types import AudioChunk

_SAMPLE_RATE = 24_000
_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
_XTTS_LANG_CODES = frozenset(
    {
        "ar",
        "cs",
        "de",
        "en",
        "es",
        "fr",
        "hi",
        "hu",
        "it",
        "ja",
        "ko",
        "nl",
        "pl",
        "pt",
        "ru",
        "tr",
        "zh-cn",
    }
)

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


def _new_accumulator(flush_tokens: int = 30):
    if _HAS_RUST:
        return _RustSentenceAccumulator(flush_tokens=flush_tokens)
    return _PySentenceAccumulator(flush_tokens=flush_tokens)


class TTSEngine:
    """Token stream → AudioChunk stream.

    First audio begins as soon as the first sentence is complete — TTS
    synthesis and LLM generation overlap at the sentence boundary.

    Coqui XTTS-v2 is synchronous PyTorch; each sentence is synthesised in a
    thread-pool executor. Chunks are posted back to the event loop via a queue
    so the generator can yield them without blocking token generation.
    """

    def __init__(
        self,
        voice: str | list[str] | tuple[str, ...] = "Ana Florence",
        lang_code: str = "en",
        model_name: str = _XTTS_MODEL,
        device: str | None = None,
        flush_tokens: int = 30,
        _accumulator=None,           # inject a pre-built accumulator for tests
        _pipeline=None,              # inject a mock pipeline callable for tests
    ) -> None:
        if lang_code not in _XTTS_LANG_CODES:
            raise ValueError(
                f"unsupported XTTS language code {lang_code!r}; expected one of "
                f"{', '.join(sorted(_XTTS_LANG_CODES))}"
            )
        self._voice = voice
        self._lang_code = lang_code
        self._model_name = model_name
        self._device = device
        self._flush_tokens = flush_tokens
        self._accumulator = _accumulator
        self._pipeline = _pipeline

    async def initialize(self) -> None:
        """Load XTTS-v2 (runs in a thread because model setup blocks)."""
        if self._pipeline is not None:
            return

        if importlib.util.find_spec("TTS") is None:
            raise RuntimeError(
                "Coqui TTS is required for TTSEngine. Install the project with "
                "`pip install -e .` or install `coqui-tts`."
            )

        from TTS.api import TTS

        pipeline = await asyncio.to_thread(TTS, self._model_name)
        device = self._device or await asyncio.to_thread(self._default_device)
        to_device = getattr(pipeline, "to", None)
        if callable(to_device):
            self._pipeline = await asyncio.to_thread(to_device, device)
        else:
            self._pipeline = pipeline

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    async def synthesize(
        self, tokens: AsyncIterator[str]
    ) -> AsyncGenerator[AudioChunk, None]:
        """Consume a token stream, yield AudioChunks as sentences are ready."""
        if self._pipeline is None:
            raise RuntimeError("call initialize() before synthesize()")

        acc = self._accumulator if self._accumulator is not None else _new_accumulator(
            self._flush_tokens
        )

        async for token in tokens:
            sentence = acc.push(token)
            if sentence:
                async for chunk in self._synthesize_sentence(sentence):
                    yield chunk

        remainder = acc.drain()
        if remainder:
            async for chunk in self._synthesize_sentence(remainder):
                yield chunk

        yield AudioChunk(
            pcm_bytes=b"",
            sample_rate=_SAMPLE_RATE,
            is_final=True,
            sentence_text="",
            chunk_latency_ms=0.0,
        )

    async def _synthesize_sentence(
        self, sentence: str
    ) -> AsyncGenerator[AudioChunk, None]:
        """Run XTTS in a thread-pool executor, streaming chunks back via queue."""
        t0 = time.monotonic()
        first = True
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any | None] = asyncio.Queue()

        def _produce() -> None:
            try:
                for audio in self._iter_pipeline_audio(sentence):
                    loop.call_soon_threadsafe(queue.put_nowait, audio)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        fut = loop.run_in_executor(None, _produce)

        while True:
            audio = await queue.get()
            if audio is None:
                break

            chunk_latency_ms = (time.monotonic() - t0) * 1000
            if first:
                metrics.event(metrics.TTS_FIRST_CHUNK, latency_ms=chunk_latency_ms)
                first = False

            if hasattr(audio, "detach"):
                audio_np = audio.detach().cpu().numpy()
            elif hasattr(audio, "numpy"):
                audio_np = audio.numpy()
            else:
                audio_np = audio

            pcm = np.asarray(audio_np, dtype=np.float32)

            yield AudioChunk(
                pcm_bytes=pcm.tobytes(),
                sample_rate=_SAMPLE_RATE,
                is_final=False,
                sentence_text=sentence,
                chunk_latency_ms=chunk_latency_ms,
            )

        await fut  # re-raise any exception from the producer thread

    def _iter_pipeline_audio(self, sentence: str) -> Iterable[Any]:
        """Return one or more model audio chunks for a sentence.

        Runtime uses Coqui's `TTS` API. The callable branch preserves the small
        streaming test doubles used by the unit tests.
        """
        if self._pipeline is None:
            raise RuntimeError("call initialize() before synthesize()")

        tts = getattr(self._pipeline, "tts", None)
        if callable(tts):
            yield tts(
                text=sentence,
                language=self._language_for_sentence(sentence),
                split_sentences=False,
                **self._voice_kwargs(),
            )
            return

        result = self._pipeline(sentence, voice=self._voice)
        yield from self._coerce_audio_result(result)

    def _voice_kwargs(self) -> dict[str, Any]:
        if isinstance(self._voice, (list, tuple)):
            return {"speaker_wav": list(self._voice)}

        voice_path = Path(self._voice).expanduser()
        if voice_path.exists():
            return {"speaker_wav": [str(voice_path)]}

        return {"speaker": self._voice}

    def _language_for_sentence(self, sentence: str) -> str:
        if self._lang_code != "ko":
            return self._lang_code

        has_hangul = any("\uac00" <= char <= "\ud7a3" for char in sentence)
        if has_hangul:
            return "ko"

        has_ascii_letter = any(
            char.isascii() and char.isalpha() for char in sentence
        )
        return "en" if has_ascii_letter else "ko"

    @staticmethod
    def _coerce_audio_result(result: Any) -> Iterable[Any]:
        if result is None:
            return

        if (
            isinstance(result, np.ndarray)
            or hasattr(result, "detach")
            or hasattr(result, "numpy")
        ):
            yield result
            return

        if isinstance(result, dict) and "wav" in result:
            yield result["wav"]
            return

        if isinstance(result, (list, tuple)) and (
            not result or np.isscalar(result[0])
        ):
            yield result
            return

        for item in result:
            if isinstance(item, tuple) and len(item) >= 3:
                yield item[2]
            elif isinstance(item, dict) and "wav" in item:
                yield item["wav"]
            else:
                yield item
