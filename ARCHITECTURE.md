# Memvox Architecture

Memvox is a low-latency streaming voice agent with persistent, evolving memory. This document describes the nine components that make up the system, the design decisions behind each one, and how they are composed to achieve sub-perceptual-threshold response times.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Map](#component-map)
3. [Components](#components)
   - [1. AudioIngress (Rust)](#1-audioingress--rust)
   - [2. ASREngine](#2-asrengine)
   - [3. LLMEngine](#3-llmengine)
   - [4. TTSEngine](#4-ttsengine)
   - [5. AudioEgress (Rust)](#5-audioegress--rust)
   - [6. WikiStore](#6-wikistore)
   - [7. WikiCompiler](#7-wikicompiler)
   - [8. SessionOrchestrator](#8-sessionorchestrator)
   - [9. MetricsCollector](#9-metricscollector)
4. [Low-Latency Design](#low-latency-design)
   - [The Latency Budget](#the-latency-budget)
   - [Form 1: LLM → TTS Overlap](#form-1-llm--tts-overlap)
   - [Form 2: VAD Pre-warm](#form-2-vad-pre-warm)
   - [Form 3: Barge-in Cancellation](#form-3-barge-in-cancellation)
5. [Data Flow](#data-flow)
6. [Repository Layout](#repository-layout)

---

## System Overview

A voice turn moves through four stages: the user's speech is captured and segmented, transcribed to text, reasoned over by an LLM (with context retrieved from the wiki), and synthesized back to speech. In a naive sequential pipeline each stage waits for the previous to complete. Memvox eliminates these wait times through overlapping execution — TTS synthesis begins on the first sentence while the LLM is still generating the second, and audio playback begins while synthesis is still running.

The wiki layer compounds knowledge across sessions. After every conversation a background process extracts new facts and updates a user-owned collection of Markdown files. These are indexed for hybrid semantic + keyword search and injected as context into every subsequent turn.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         memvox-audio  (Rust)                        │
│                                                                     │
│   Microphone ──► AudioIngress ──────────────────► AudioEgress ──► Speaker
│                  cpal + Silero VAD                cpal + rubato    │
│                       │  ▲                            ▲            │
│                SpeechStarted                    CancelPlayback     │
│                SpeechSegment                    AudioChunk PCM     │
└───────────────────────┼────────────────────────────────────────────┘
                        │ Unix socket (bincode)          ▲
┌───────────────────────▼────────────────────────────────┼────────────┐
│                  SessionOrchestrator  (Python)          │            │
│                                                                     │
│  ingress shim ──► ASREngine ──► LLMEngine ──► TTSEngine ──► egress shim
│                  faster-      vLLM HTTP     XTTS-v2 +     Unix     │
│                  whisper      stream        SentenceAcc   socket   │
│                       │            │                               │
│                  WikiStore.search()│                               │
│                       ▲            ▼                               │
│                  WikiStore    WikiCompiler  (post-session, async)  │
│                  LanceDB      vLLM HTTP                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Map

```
Layer               Component              Language    Location
──────────────────────────────────────────────────────────────────────
Hardware Audio      AudioIngress           Rust        memvox-audio/src/ingress.rs
                    AudioEgress            Rust        memvox-audio/src/egress.rs
                    (Barge-in signal)      Rust        memvox-audio/src/barge_in.rs
                    (IPC framing)          Rust        memvox-audio/src/ipc.rs

Voice Pipeline      ASREngine              Python      memvox/voice/asr.py
                    LLMEngine              Python      memvox/voice/llm.py
                    TTSEngine              Python      memvox/voice/tts.py
                    SentenceAccumulator    Rust/PyO3   memvox-rs/src/sentence_acc.rs

Memory              WikiStore              Python      memvox/wiki/store.py
                    WikiCompiler           Python      memvox/wiki/compiler.py

Orchestration       SessionOrchestrator    Python      memvox/session/orchestrator.py

Observability       MetricsCollector       Python      memvox/observability/metrics.py
```

---

## Components

### 1. AudioIngress — Rust

**Purpose:** Capture microphone audio and detect speech boundaries using the Silero VAD model, producing a complete `SpeechSegment` per utterance and an early-onset signal before the utterance ends.

**Interface**

```
Outbound (binary → Python, via Unix socket):
  SpeechStarted { timestamp: f64 }
    — emitted at speech onset, before utterance is complete
    — used to pre-warm the LLM's KV cache speculatively

  SpeechSegment { audio: bytes, speech_prob: f32, duration_ms: f32, timestamp_start: f64 }
    — emitted at confirmed end-of-utterance (VAD trailing silence)
    — audio is 16kHz mono int16 PCM
```

**State Machine**

```
                    prob > 0.5
                   for N frames
  SILENT ─────────────────────► SPEAKING
    ▲                                │
    │   N frames below 0.35          │ prob drops below 0.35
    └────────────────────────── TRAILING ──► emit SpeechSegment
                                            (then → SILENT)
```

**Key libraries**
- `cpal` — cross-platform audio I/O. Opens the microphone on a dedicated real-time OS thread via callback. Gives direct access to the OS audio subsystem with no Python GIL involvement.
- `ort` — ONNX Runtime for Rust. Loads the Silero VAD `.onnx` model natively. No PyTorch, no Python interpreter, no CUDA required (VAD runs on CPU).

**Why Rust and not Python:** `cpal` drives audio on a real-time callback thread. Managing the GIL across a blocking real-time thread from PyO3 requires `py.allow_threads()` scaffolding and creates subtle correctness risks. A standalone binary owns its thread cleanly. The more important reason: AudioIngress and AudioEgress live in the same binary so barge-in cancellation happens via an internal Rust channel — no Python, no asyncio, no IPC.

---

### 2. ASREngine

**Purpose:** Transcribe a `SpeechSegment` to text with language detection.

**Interface**

```python
Input:  SpeechSegment
Output: Transcript(
    text: str,
    language: str,          # ISO 639-1, auto-detected
    confidence: float,
    no_speech_prob: float,
    latency_ms: float,
    source_segment: SpeechSegment
)
```

**Filtering:** Segments with `no_speech_prob > 0.6` or empty text after stripping are silently dropped (a `asr.drop` metric event is emitted).

**Key library: `faster-whisper`**
A CTranslate2-backed reimplementation of OpenAI's Whisper. Runs 4× faster than the reference implementation at the same accuracy. The Python layer is a thin wrapper — all inference happens in C++ and CUDA. `large-v3` gives best accuracy; `distil-large-v3` cuts latency by ~40% with a small quality trade.

Note: Whisper is not a streaming transcription model. It transcribes complete utterances. The real-time feel comes from the VAD's early `SpeechStarted` signal allowing downstream stages to begin work while the user is still speaking.

---

### 3. LLMEngine

**Purpose:** Generate a streaming response given conversation history and wiki context snippets.

**Interface**

```python
Input:
  GenerationRequest(
      messages: list[ChatMessage],     # full history — caller owns history
      context_snippets: list[str],     # injected wiki excerpts
      session_id: str,
      turn_id: str,
      thinking_enabled: bool = False   # off by default — adds 300–2000ms
  )

Output: AsyncGenerator[TokenChunk]
  TokenChunk(
      text: str,
      is_thinking: bool,   # True inside <think>...</think> blocks
      is_final: bool,
      ttft_ms: float | None,  # set on first non-thinking chunk
      turn_id: str
  )
```

**Key library: vLLM (via OpenAI-compatible HTTP)**

vLLM is the highest-throughput open-source LLM inference server. It uses PagedAttention for efficient KV cache management and continuous batching to minimise per-token latency. The `openai.AsyncOpenAI` client communicates with vLLM's OpenAI-compatible endpoint at `http://localhost:8000/v1`.

Using the HTTP API rather than `AsyncLLMEngine` directly means:
- Unit tests mock the HTTP layer — no GPU required
- Model swaps are a config change (`llm_model`, `llm_base_url`)
- vLLM can run on a different machine in a cloud deployment

**Model:** Qwen3-8B-Instruct by default (excellent Korean + English). For English-only use, Llama 3.1 8B Instruct is a strong alternative.

**Thinking mode** is always `False` for voice. The `<think>` tokens are streamed through with `is_thinking=True` so the orchestrator can log them for observability without sending them to TTS.

---

### 4. TTSEngine

**Purpose:** Convert a stream of text tokens into a stream of audio chunks. First audio begins before the full response is known.

**Interface**

```python
Input:  AsyncIterator[str]           # token-by-token text from LLMEngine
Output: AsyncGenerator[AudioChunk]
  AudioChunk(
      pcm_bytes: bytes,              # 24kHz float32 little-endian
      sample_rate: int,              # always 24000 for XTTS-v2
      is_final: bool,
      sentence_text: str,
      chunk_latency_ms: float
  )
```

**Internal pipeline**

```
Token stream
    │
    ▼
SentenceAccumulator (Rust/PyO3)
    │  push(token) → None or sentence str
    │
    ▼  on sentence ready
asyncio.to_thread(TTS.tts(sentence))      ← non-blocking: XTTS runs in thread pool
    │
    ▼  yields 24 kHz float32 audio
AudioChunk emitted per model chunk
```

**SentenceAccumulator** is a PyO3 Rust extension (`memvox._rust.SentenceAccumulator`). It detects sentence boundaries from punctuation (`. ! ?`) or flushes after 30 tokens. It is called on every LLM token — potentially 100+ times per turn — so the cost of a Python loop here is measurable. The Rust FFI call costs ~0.

**Key library: Coqui XTTS-v2**
Multilingual TTS model with English (`en`) and Korean (`ko`) support, 24 kHz output, and optional voice cloning from speaker WAV references.

`asyncio.to_thread` keeps the event loop free while XTTS synthesizes — this is what allows LLM token generation and TTS synthesis to run concurrently.

---

### 5. AudioEgress — Rust

**Purpose:** Receive `AudioChunk` PCM from Python, resample to device native rate, and play back via the speaker. Supports instantaneous barge-in cancellation.

**Interface**

```
Inbound (Python → binary, via Unix socket):
  AudioChunk { pcm_bytes: bytes, sample_rate: u32, is_final: bool, ... }
  CancelPlayback {}
  PausePlayback {} / ResumePlayback {}

Outbound (binary → Python):
  PlaybackStarted { timestamp: f64 }   ← timestamp of first sample written to device
  PlaybackFinished {}
  PlaybackCancelled {}
```

**Key libraries**
- `cpal` — playback on a dedicated real-time thread, same as ingress.
- `rubato` — purpose-built Rust resampling library. Converts TTS 24kHz output to device native rate (typically 48kHz). Reinitialises automatically if `sample_rate` changes between chunks, enabling seamless TTS model swaps. Eliminates scipy from the playback hot path.

**Barge-in cancel path**

```
memvox-audio process (Rust)

  Ingress thread                        Egress thread
  ──────────────                        ─────────────
  VAD: speech_prob > 0.5
  → watch::Sender::send(Cancel) ──────► watch::Receiver detects Cancel
                                        → cpal device.stop()
                                        → send PlaybackCancelled over Unix socket
                                        (< 1ms total)
```

No Python is involved in stopping audio. This is the reason AudioIngress and AudioEgress are in the same binary.

---

### 6. WikiStore

**Purpose:** Persist user knowledge as Markdown files and serve fast hybrid search during live conversations.

**Interface**

```python
# Write
await store.upsert_article(WikiArticle)
await store.delete_article(slug: str)

# Read
await store.search(query: str, top_k: int = 5) -> list[SearchResult]
await store.get_article(slug: str) -> WikiArticle | None
await store.list_articles() -> list[WikiArticle]

SearchResult(
    article: WikiArticle,
    score: float,           # RRF-fused rank score
    matched_chunks: list[str]  # top-2 relevant excerpts, not full body
)
```

**Storage model**

```
wiki/
  korean-grammar-particles.md     ← source of truth (plain Markdown)
  cafe-vocabulary.md
  ...

LanceDB table: chunks
  slug | chunk_text | embedding: vector(384) | tags | updated_at
  ─────────────────────────────────────────────────────────────
  Articles are chunked into 300-token windows (50-token overlap)
  One row per chunk. Index is always rebuildable from Markdown.
```

**Hybrid search with RRF**

```
query
  │
  ├──► vector search  (semantic similarity)  ──► ranked list A
  │    all-MiniLM-L6-v2 embeddings
  │
  └──► BM25 FTS       (keyword matching)     ──► ranked list B

  Reciprocal Rank Fusion (k=60):
    score(slug) = Σ 1/(60 + rank_i)

  Top-K slugs by fused score → matched_chunks[:2] per slug
```

**Key library: LanceDB**
Written in Rust, embeds directly in-process via Python bindings (no server to manage). Supports hybrid vector + full-text search natively. The Python API is a thin wrapper — search operations execute in Rust. Ideal for local-first: zero infrastructure overhead.

**Markdown as source of truth:** User owns their data. Files are Obsidian-compatible, git-trackable, exportable. The LanceDB index is a cache and can always be rebuilt from the `.md` files.

**Boundary rule:** `memvox/wiki/` has zero imports from `memvox/voice/` or `memvox/session/`. It is independently usable with any LLM interface, not just the voice pipeline.

---

### 7. WikiCompiler

**Purpose:** After each session, synthesize the conversation transcript into structured wiki articles using a non-streaming LLM call. Runs entirely in the background.

**Interface**

```python
Input:
  CompileRequest(
      session_id: str,
      transcript: list[ConversationTurn],
      existing_slugs: list[str]
  )

Output:
  CompileResult(
      created: list[WikiArticle],
      updated: list[WikiArticle],
      skipped_reason: str | None   # e.g. "transcript too short"
  )
```

**Process**

```
transcript (list of turns)
    │
    ▼
LLM prompt (Qwen3-8B, non-streaming)
  "Given this transcript and existing wiki slugs,
   return a JSON array of articles to create or update."
    │
    ▼
JSON: [{slug, title, body, tags, action: "create"|"update"}, ...]
    │
    ├── Pydantic validation
    ├── retry once on parse failure (simplified prompt)
    └── skip if < 4 turns or < 100 tokens
    │
    ▼
WikiStore.upsert_article() for each result
```

**Fire-and-forget:** `SessionOrchestrator.stop()` launches `WikiCompiler.compile()` as a background `asyncio.Task` and does not await it. The session ends immediately. Memory catches up over 2–10 seconds in the background.

---

### 8. SessionOrchestrator

**Purpose:** Wire all components into a live turn-taking loop. Own conversation history and session lifecycle. The only component that knows the full pipeline shape.

**Concurrent task structure (Phase 2)**

```
asyncio.Task: _barge_in_monitor  ◄── watches SpeechStarted during playback
                    │
                    │ fires CancelPlayback + aclose(llm_gen) + asyncio.Event
                    ▼
asyncio.Task: _ingest_loop
  ingress_socket → ASREngine → transcript_queue(maxsize=4)
                    │
                    ▼
asyncio.Task: _turn_loop
  transcript_queue → WikiStore.search() → LLMEngine → token_queue(maxsize=4)
                                                  │
                                                  ▼
                                        asyncio.Task: _tts_loop
                                          TTSEngine → audio_queue(maxsize=4)
                                                  │
                                                  ▼
                                        asyncio.Task: _playback_loop
                                          egress_socket.write(AudioChunk)
```

**Phase flag:** `SessionConfig.overlapping: bool`
- `False` (Phase 1): `_turn_loop` collects the full LLM response before calling TTSEngine. No concurrency between LLM and TTS.
- `True` (Phase 2): All five tasks run concurrently. TTS starts on first sentence. Playback starts on first chunk.

**History management:** `conversation_history: list[ChatMessage]`, capped at 20 turns. Passed in full to `LLMEngine` every call. History lives here — not in `LLMEngine` — so `LLMEngine` stays stateless and testable.

**Session lifecycle:**
```python
await orchestrator.start()   # spawns all tasks, opens Unix sockets
# ... conversation runs ...
await orchestrator.stop()    # graceful shutdown
                             # fires WikiCompiler.compile() in background
                             # does NOT await — session ends immediately
```

---

### 9. MetricsCollector

**Purpose:** Record per-stage latency and session health as OpenTelemetry spans and events. Imported as a module-level singleton — never injected as a dependency.

**Usage**

```python
from memvox.observability import metrics

# Span (measures elapsed time)
async with metrics.span("asr.transcribe", turn_id=turn_id) as span:
    result = await asr.transcribe(segment)
    span.set("latency_ms", span.elapsed_ms)

# Event (point-in-time)
metrics.event("mouth_to_ear", latency_ms=playback_ts - speech_ended_ts)
```

**Full event inventory**

| Event | Description |
|---|---|
| `mouth_to_ear` | **Primary metric.** `PlaybackStarted.timestamp − SpeechEnded.timestamp` |
| `llm.ttft` | Time to first non-thinking LLM token |
| `asr.latency_ms` | SpeechSegment received → Transcript emitted |
| `tts.first_chunk` | Sentence ready → first AudioChunk yielded |
| `vad.trailing_ms` | Trailing silence duration (tuning signal) |
| `wiki.search_latency_ms` | Hybrid search duration (target: <50ms) |
| `barge_in.latency_ms` | SpeechStarted (during playback) → PlaybackCancelled |
| `pipeline.overlap_ms` | LLM/TTS actual overlap (validates Phase 2) |
| `llm.tokens_per_second` | Generation throughput (GPU contention signal) |
| `tts.sentence_queue_depth` | 0 = LLM bottleneck; maxsize = TTS bottleneck |
| `asr.no_speech_drop_rate` | Junk segment drop rate (VAD threshold signal) |
| `session.start` / `session.end` | Session boundaries |
| `asr.drop` | Individual dropped segment |

**Sinks:** Configured via `MEMVOX_METRICS_SINK=memory|otlp|prometheus`. In tests, `memory` sink makes all spans and events inspectable as a list.

---

## Low-Latency Design

### The Latency Budget

The target metric is **mouth-to-ear latency**: the time from when the user stops speaking to when they hear the first audio back.

```
User stops speaking
        │
        │  ~200ms   VAD trailing silence (floor — human speech patterns)
        ▼
  SpeechSegment emitted
        │
        │  ~100–150ms  ASR transcription (distil-large-v3, RTX 5090, 5s utterance)
        ▼
  Transcript emitted
        │
        │  ~15–30ms   WikiStore hybrid search
        ▼
  context_snippets ready → LLMEngine starts
        │
        │  ~80–120ms  LLM time-to-first-token (Qwen3-8B, vLLM, RTX 5090)
        ▼
  First tokens arrive → SentenceAccumulator filling
        │
        │  ~50–70ms   First sentence generated (~10 tokens at 150 tok/s)
        ▼
  First sentence ready → XTTS starts
        │
        │  ~100–150ms  TTS first chunk
        ▼
  AudioChunk → Unix socket → memvox-audio → speaker
        │
        ▼
  User hears response

  Total: ~545–720ms on RTX 5090 (Phase 2 overlapping)
  Total: ~940–1290ms on RTX 4090
```

The 200ms VAD trailing silence is a floor that cannot be engineered away — it is required to confirm the user has stopped speaking and avoid false endings. Everything else is fair game for optimization.

---

### Form 1: LLM → TTS Overlap

The most impactful latency optimization. Without overlap, TTS waits for the entire LLM response. With overlap, TTS starts on the first sentence while the LLM is still generating.

```
Sequential (Phase 1):
─────────────────────────────────────────────────────────────────────►
  [     LLM generates full response     ] [ TTS ] [ TTS ] [ play ]
                                                                    ▲
                                                         user hears ~1.2s

Overlapping (Phase 2):
─────────────────────────────────────────────────────────────────────►
  [     LLM token stream ...                                        ]
                      [ TTS sentence 1 ]
                                    [ play s1 ]
                              [ TTS sentence 2 ]
                                            [ play s2 ]
                                    ▲
                         user hears ~600ms
```

**Implementation:** Three concurrent `asyncio.Task` objects with bounded queues between them.

```
_turn_loop                     _tts_loop                  _playback_loop
──────────                     ─────────                  ──────────────

LLMEngine.generate()
  token 1 ─► token_q ────────► SentenceAccumulator.push()
  token 2 ─► token_q ────────► → None (accumulating)
  ...
  token 12 ► token_q ────────► → "Hello, how are you?"
                                 asyncio.to_thread(XTTS)
                                   chunk 1 ─► audio_q ──► Unix socket write
                                   chunk 2 ─► audio_q ──► Unix socket write
  token 13 ► token_q          (XTTS still running ↑)     (egress playing ↑)
```

Queues use `maxsize=4` to create backpressure — a fast LLM cannot outrun a slow TTS unboundedly. Memory usage stays bounded.

---

### Form 2: VAD Pre-warm

`SpeechStarted` is emitted by `AudioIngress` at speech *onset* — before the utterance ends. This fires ~500ms–2s before `SpeechSegment` arrives (depending on how long the user speaks).

```
User starts speaking
        │
        │  ~200ms  VAD onset detection (2–3 frames × 96ms)
        ▼
  SpeechStarted ──────────────────────────────► Orchestrator
        │                                            │
        │  user continues speaking                   │  begin KV cache prefill:
        │                                            │  system_prompt + history
        │                                            │  (~50–100ms on RTX 5090)
        ▼                                            ▼
  VAD trailing silence                    LLM ready and waiting
        │
        ▼
  SpeechSegment → ASR → Transcript → LLMEngine
                                          │
                                          │  TTFT reduced by ~50–100ms
                                          ▼
                                    First token arrives sooner
```

The pre-warm overlaps with the last portion of the user's utterance plus ASR processing time.

---

### Form 3: Barge-in Cancellation

When the user starts speaking while the bot is talking, the bot must stop immediately.

```
memvox-audio (Rust)
─────────────────────────────────────────────────────────────────────

  Ingress thread                           Egress thread
  ──────────────                           ─────────────
  VAD: speech onset detected               playing AudioChunk N...
  watch::Sender::send(BargIn) ───────────► watch::Receiver fires
                                           cpal device.stop()     (~0ms)
                                           send PlaybackCancelled over socket

Python asyncio
─────────────────────────────────────────────────────────────────────

  _barge_in_monitor task
    receives SpeechStarted from ingress socket
    calls llm_generator.aclose()            → HTTP stream cancelled
    sets barge_in_event                     → _turn_loop wakes, discards turn

  New SpeechSegment arrives → fresh turn begins
```

**Total barge-in latency: ~200–300ms from speech onset to audio stopping.**
The floor is VAD onset detection (~200ms). Audio stop is sub-millisecond once the internal Rust signal fires. Python is not in the audio-stopping path.

---

## Data Flow

### Complete turn (Phase 2, with wiki context)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. User speaks                                                     │
│     cpal callback (16kHz PCM) → Silero VAD frames                  │
│     → SpeechStarted (at onset) → SpeechSegment (at end)            │
│     Both sent over Unix socket to Python                            │
│                                                                     │
│  2. ASR                                                             │
│     SpeechSegment → faster-whisper → Transcript                    │
│     latency: ~100–400ms depending on utterance length + hardware   │
│                                                                     │
│  3. Wiki retrieval                                                  │
│     Transcript.text → LanceDB hybrid search → top-5 SearchResults  │
│     latency: <50ms                                                  │
│                                                                     │
│  4. LLM generation (streaming)                                      │
│     history + context_snippets → vLLM (Qwen3-8B) → token stream   │
│     latency to first token: ~80–150ms on RTX 5090                  │
│                                                                     │
│  5. TTS (overlapping with step 4)                                   │
│     token stream → SentenceAccumulator → XTTS-v2 → AudioChunk stream│
│     first audio: ~150–300ms after first sentence complete           │
│                                                                     │
│  6. Playback (overlapping with steps 4+5)                           │
│     AudioChunk → Unix socket → memvox-audio → rubato resample      │
│     → cpal output device → speaker                                  │
│                                                                     │
│  7. Post-session (background, after stop())                         │
│     full transcript → WikiCompiler → Qwen3-8B (non-streaming)      │
│     → upsert_article() × N → LanceDB index updated                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
memvox/                          ← repo root
│
├── Cargo.toml                   ← Cargo workspace (members: memvox-audio, memvox-rs)
├── pyproject.toml               ← Python package (maturin build backend for PyO3)
│
├── memvox-audio/                ← Rust standalone binary (cpal + ort + rubato + tokio)
│   ├── Cargo.toml               ←   [[bin]]
│   └── src/
│       ├── main.rs
│       ├── ingress.rs           ←   AudioIngress: cpal mic + Silero VAD state machine
│       ├── egress.rs            ←   AudioEgress: cpal playback + rubato resampling
│       ├── barge_in.rs          ←   tokio::sync::watch channel: ingress → egress cancel
│       └── ipc.rs               ←   Unix socket framing (bincode)
│
├── memvox-rs/                   ← Rust PyO3 extension (compiled → memvox/_rust.so)
│   ├── Cargo.toml               ←   [lib] crate-type = ["cdylib"]
│   └── src/
│       ├── lib.rs
│       └── sentence_acc.rs      ←   SentenceAccumulator: punctuation boundary detection
│
└── memvox/                      ← Python package
    ├── _rust.pyi                ←   type stubs for compiled Rust extension
    ├── wiki/                    ←   zero imports from voice/ or session/
    │   ├── types.py             ←   WikiArticle, SearchResult, ChatMessage, etc.
    │   ├── store.py             ←   WikiStore (LanceDB + Markdown)
    │   └── compiler.py         ←   WikiCompiler (post-session LLM extraction)
    ├── voice/
    │   ├── types.py             ←   SpeechSegment, Transcript, AudioChunk, etc.
    │   ├── ingress.py           ←   thin asyncio socket client for memvox-audio outbound
    │   ├── asr.py               ←   ASREngine (faster-whisper)
    │   ├── llm.py               ←   LLMEngine (vLLM HTTP, openai client)
    │   ├── tts.py               ←   TTSEngine (XTTS-v2 + SentenceAccumulator)
    │   └── egress.py            ←   thin asyncio socket client for memvox-audio inbound
    ├── session/
    │   ├── types.py             ←   SessionConfig
    │   └── orchestrator.py     ←   SessionOrchestrator (all concurrent tasks)
    ├── observability/
    │   └── metrics.py          ←   MetricsCollector (OTel singleton)
    └── skins/
        └── korean_tutor.py     ←   SessionConfig factory (~15 lines, no logic)
```
