## Components
### Audio I/O — Rust process
Local microphone capture and speaker playback. Reuses the existing memvox Rust binary (with the deep-understanding pass completed before any changes). No browser, no WebRTC, no LiveKit in v1 — direct local audio I/O for tight latency control and architectural focus.
### VAD — Rust, in the audio process
Silero VAD running in the same Rust process as audio I/O via ONNX. Emits speech-start and speech-end events to the Python orchestrator. Adding VAD to the existing Rust binary rather than introducing a new component.
### Turn detection — Python, in the orchestrator
Heuristic for v1: VAD-silence-of-N-ms with tunable but hardcoded threshold. Semantic turn detection (small LLM predicting turn-end) is deferred to a later phase.
### ASR — local, GPU-accelerated
Whisper-large-v3 via faster-whisper, running on the 5090. Streaming partial transcripts during speech, finalization on turn-end. Korean-capable. Local-only (no cloud ASR alternative in v1).
### Retrieval — LanceDB
Vector search only for v1 (not hybrid). Small wiki of personal Korean study notes as markdown files. Every-turn retrieval. Filesystem watcher and Obsidian integration are deferred to v2.
### LLM — pluggable, cloud default
Default backend: Anthropic API, Claude Haiku 4.5.
Local backend: Qwen 2.5 14B (AWQ-4bit) served via vLLM as a separate local service, called by the orchestrator over its OpenAI-compatible streaming API. Both backends implement a common streaming interface with cancellation support for barge-in.
### TTS — pluggable, cloud default
Default backend: Cartesia Sonic-2 over WebSocket streaming, Korean voice.
Local backend: XTTS-v2 on the 5090, Korean voice. Both implement a common streaming interface that takes a text stream in (not a completed string) and emits audio chunks, with cancellation support.
### Orchestrator — Python
Streaming-everywhere: ASR partials feed speculative LLM start, LLM tokens stream into TTS, TTS audio streams to playback. Backend-agnostic — talks to LLM and TTS through the common interfaces. Owns conversation state, retrieval triggering, barge-in coordination, and cancellation propagation.
### Barge-in handling
VAD detects user speech while TTS is playing. Orchestrator propagates cancellation to in-flight LLM and TTS streams via the common interface's cancel_event. Both backend types handle teardown cleanly.
### Configuration system
TOML or YAML config file selecting LLM and TTS backends and their per-backend config. Secrets (API keys) loaded from environment variables, not the config file. A simple registry maps backend names to factories — no dynamic loading or plugin system. Configurability scoped to backend selection only; tuning parameters hardcoded for v1.
### Session transcripts — local files
Per-turn writes to a markdown file per session, in a sessions/ directory. YAML frontmatter with metadata (session_id, started_at, language, turn_count, schema_version). Indexed into LanceDB after session completion. Authored wiki content in authored/ directory is read-only to the agent. Session-end structured extraction is deferred to v2.
### Instrumentation and metrics
End-to-end latency measurement instrumented at every stage boundary: capture-to-VAD, VAD-to-ASR-final, ASR-to-LLM-first-token, LLM-first-token-to-TTS-first-audio, TTS-to-playback. Per-turn records logged. Aggregate p50/p95/p99 dashboards. Comparable across backend configurations (cloud vs local) for the blog post.


### Explicitly deferred to v2 or later

LiveKit / WebRTC / browser client
Filesystem watcher and Obsidian integration
Hybrid retrieval (vector + BM25 + recency weighting + source-type weighting)
Semantic turn detection
Paralinguistic feature extraction
Pronunciation analysis
Session-end structured extraction (vocabulary, grammar points, errors)
Cross-OS support
CPU-only fallback for local models
Packaging and distribution polish
Configurable tuning parameters beyond backend selection
