//! SentenceAccumulator — punctuation-boundary detection + N-token flush.
//!
//! Called synchronously on every LLM token (~100+ per turn). Pure string
//! processing; no blocking, no I/O. PyO3 FFI overhead is ~0, which is why
//! this lives here rather than behind a Unix socket.
//!
//! TTSEngine calls it as:
//!   acc = SentenceAccumulator(flush_tokens=30)
//!   if sentence := acc.push(token):
//!       tts_queue.put_nowait(sentence)
//!   if remainder := acc.drain():
//!       tts_queue.put_nowait(remainder)

use pyo3::prelude::*;

const SENTENCE_ENDINGS: &[char] = &['.', '!', '?', '。', '！', '？'];

#[pyclass]
pub struct SentenceAccumulator {
    buffer: String,
    token_count: usize,
    flush_tokens: usize,
}

#[pymethods]
impl SentenceAccumulator {
    #[new]
    #[pyo3(signature = (flush_tokens = 30))]
    pub fn new(flush_tokens: usize) -> Self {
        Self {
            buffer: String::new(),
            token_count: 0,
            flush_tokens,
        }
    }

    /// Push one LLM token. Returns a flushed sentence string when either:
    ///   - the buffer ends with sentence-ending punctuation, or
    ///   - `flush_tokens` tokens have accumulated without a boundary.
    pub fn push(&mut self, token: &str) -> Option<String> {
        self.buffer.push_str(token);
        self.token_count += 1;

        let ends_on_boundary = self
            .buffer
            .trim_end()
            .ends_with(SENTENCE_ENDINGS);
        let over_flush_limit = self.token_count >= self.flush_tokens;

        if ends_on_boundary || over_flush_limit {
            Some(self.take())
        } else {
            None
        }
    }

    /// Drain whatever remains in the buffer at turn end.
    pub fn drain(&mut self) -> Option<String> {
        let s = self.buffer.trim().to_string();
        self.buffer.clear();
        self.token_count = 0;
        if s.is_empty() { None } else { Some(s) }
    }

    /// Current buffer contents without flushing (useful for tests).
    pub fn peek(&self) -> &str {
        &self.buffer
    }

    fn take(&mut self) -> String {
        let s = self.buffer.trim().to_string();
        self.buffer.clear();
        self.token_count = 0;
        s
    }
}
