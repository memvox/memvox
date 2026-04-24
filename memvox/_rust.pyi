# Type stubs for memvox._rust (PyO3 extension, compiled by maturin)
#
# Build:  maturin develop --manifest-path memvox-rs/Cargo.toml
# Output: memvox/_rust.so  (Linux) / memvox/_rust.pyd  (Windows)
#
# Import: from memvox._rust import SentenceAccumulator

class SentenceAccumulator:
    """Punctuation-boundary detection and N-token flush for the TTS pipeline.

    Called on every LLM token. Returns a complete sentence string when a
    flush is triggered (punctuation boundary or token-count limit).
    """

    def __init__(self, flush_tokens: int = 30) -> None: ...

    def push(self, token: str) -> str | None:
        """Accumulate one token. Returns a flushed sentence or None."""
        ...

    def drain(self) -> str | None:
        """Flush remaining buffer at turn end. Returns None if empty."""
        ...

    def peek(self) -> str:
        """Return current buffer contents without flushing."""
        ...
