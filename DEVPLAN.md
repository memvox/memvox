# memvox — Development Plan

A living document. Update phase status as work lands; rewrite estimates when reality
diverges. The phases come from `architecture_components.md` build order +
`project_memvox.md` roadmap; deliverables here are concrete enough to check off.

---

## Current snapshot — 2026-04-28

- **Phase 0**: ✅ complete
- **Phase 1**: ✅ complete (Rust + Python paths both validated end-to-end)
- **Phase 2**: ⏳ next
- **Phase 3 / 4 / 5**: not started

**Latency baseline (Phase 1, RTX 5090, sequential mode)**:

| Metric | Avg | P95 | Notes |
|---|---:|---:|---|
| `mouth_to_ear` | 1241 ms | 2072 ms | end-of-utterance → first audio out |
| `asr.transcribe` | 185 ms | 295 ms | Whisper large-v3 |
| `wiki.query` | 18 ms | 105 ms | LanceDB hybrid; empty wiki |
| `llm.ttft` | 37 ms | 119 ms | Qwen3-8B via vLLM, thinking disabled |
| `tts.first_chunk` | 683 ms | 1163 ms | XTTS-v2 (Korean-capable; heavier than Kokoro) |

`tts.first_chunk` is the dominant single cost. Architecture target was <400 ms
mouth-to-ear; that target assumed Kokoro (~150 ms first chunk) and Phase 2
overlap. Hitting it on the current TTS will require Phase 2 + a faster
Korean-capable TTS in Phase 4.

---

## Phase 0 — Foundations  ✅

Repo skeleton, shared types, observability primitives, and the wiki store.
Everything below the voice pipeline that the engines depend on.

- ✅ Repo layout: `memvox/` Python package + `memvox-audio/` Rust binary +
      `memvox-rs/` PyO3 crate, all sibling at workspace root
- ✅ Shared dataclasses: `wiki/types.py`, `voice/types.py`, `session/types.py`
- ✅ `MetricsCollector` with `MemorySink` / `_OTLPSink` / `_PrometheusSink`,
      `metrics.span()` / `metrics.event()` / `metrics.override()` / `metrics.summary()`
- ✅ `WikiStore` with LanceDB hybrid search (vector + BM25 RRF, 300-token chunks)

Key decisions captured in `architecture_decisions.md`:
- `ChatMessage` lives in `wiki/types.py` to keep `wiki/` extractable (no voice deps)
- `wiki/` has zero imports from `voice/` or `session/` — one-directional boundary
- Metrics is a module-level singleton with an override context manager for tests

---

## Phase 1 — Sequential voice loop  ✅

A working ASR → LLM → TTS turn-taking loop with the Korean tutor skin.
Sequential by design: each stage finishes before the next begins. Goal was
end-to-end correctness, not latency.

- ✅ `ASREngine` (faster-whisper, `large-v3`, GPU + CPU fallback path)
- ✅ `LLMEngine` (OpenAI-compatible HTTP client, vLLM, `_ThinkingParser` for
      Qwen3 `<think>` blocks, `enable_thinking` plumbed via `chat_template_kwargs`)
- ✅ `TTSEngine` (Coqui XTTS-v2 via thread executor, `SentenceAccumulator`
      sentence boundary detection — Python fallback for the Rust crate)
- ✅ `voice/ingress.py` + `voice/egress.py` — thin asyncio Unix-socket clients
      speaking length-prefixed bincode
- ✅ `SessionOrchestrator` (sequential mode; `overlapping=False`)
- ✅ `memvox/__main__.py` entry point + `skins/korean_tutor.py` factory
- ✅ `shim.py` — Python audio shim (sounddevice + webrtcvad) as a dev stand-in
      for `memvox-audio`
- ✅ `memvox-audio` Rust binary (`cpal` mic capture + energy VAD + state
      machine + `cpal` playback + `rubato` resampling + `BargeInSignal` wiring)
- ✅ Wire format pinned: `[u32 LE length][bincode payload]` with `u32` enum
      tag + `u64` Vec lengths + LE fixed-int. Verified Python ↔ Python and
      Python ↔ Rust round-trip.
- ✅ `MOUTH_TO_EAR` instrumented; `metrics.summary()` prints latency table on
      `SessionOrchestrator.stop()`
- ✅ Test coverage: 112 unit tests + 1 socket-roundtrip integration test, green
- ⏳ `memvox-rs` PyO3 `SentenceAccumulator` scaffolded but not compiled —
      Python fallback active, performance cost is negligible. Defer to Phase 4.

Validated runs through both `shim.py` and `./target/release/memvox-audio`
with the Jabra Evolve2 50 via PipeWire. Latency baseline above.

---

## Phase 2 — Overlapping orchestrator + barge-in  ⏳ NEXT

Convert `SessionOrchestrator` from sequential to a concurrent task pipeline.
This is the headline engineering claim of the project: ASR → LLM → TTS →
playback running concurrently with bounded backpressure, and barge-in cancel
that traverses the Rust audio binary in <1 ms.

### Deliverables

- [ ] **Three concurrent `asyncio.Task` objects** in the orchestrator,
      connected by bounded `asyncio.Queue` (maxsize=4):
  - `_turn_loop`: receives `SpeechSegment`, runs ASR + wiki search,
    streams LLM tokens onto `token_queue`
  - `_tts_loop`: drains `token_queue`, runs `SentenceAccumulator`, calls
    XTTS per sentence, puts `AudioChunk` onto `audio_queue`
  - `_playback_loop`: drains `audio_queue` and writes to egress
- [ ] **LLM streaming through `_sequential_tokens`'s replacement** —
      yield each non-thinking token as it arrives instead of buffering the
      full reply. Removes the LLM-completion wait from the critical path.
- [ ] **`_barge_in_monitor`**: fourth concurrent task that watches
      `SpeechStarted` during active playback. On fire:
  - Calls `aclose()` on the in-flight LLM async generator
  - Sets an `asyncio.Event` so `_turn_loop` abandons the current generation
  - Triggers `BargeInSignal.fire()` (already wired) so the Rust egress
    callback flushes the playback buffer within one cpal frame
- [ ] **Same `SessionConfig.overlapping` flag toggles between modes** —
      Phase 1 sequential code stays for fallback / testing; Phase 2 path is
      the new default.
- [ ] **New metrics** wired in:
  - `barge_in.latency_ms` — `SpeechStarted` → `PlaybackCancelled`
  - `pipeline.overlap_ms` — proves Phase 2 is actually working
  - `vad.trailing_ms` — exposes the VAD tuning knob
  - `tts.sentence_queue_depth` — backpressure indicator
- [ ] **Tests**:
  - Unit tests for the three concurrent loops (mocked engines, real queues)
  - Barge-in test (simulate SpeechStarted mid-playback, assert cancel < 50 ms)
  - Overlap test (assert TTS first chunk fires before LLM final token)
  - Existing socket-roundtrip integration test extended for overlap mode

### Expected outcome

| Metric | Phase 1 | Phase 2 (target) |
|---|---:|---:|
| `mouth_to_ear` avg | 1241 ms | ~900 ms |
| `mouth_to_ear` p95 | 2072 ms | ~1500 ms |
| `pipeline.overlap_ms` | 0 | >300 ms |
| `barge_in.latency_ms` | n/a | <50 ms |

The remaining gap to <400 ms target comes from XTTS first-chunk cost; that's
a Phase 4 TTS-swap concern, not Phase 2.

### Out of scope for Phase 2

- TTS engine swap (Phase 4)
- WikiCompiler / wiki auto-update (Phase 3)
- React UI (Phase 4)
- Compiled `memvox-rs` PyO3 (Phase 4)

---

## Phase 3 — LLM Wiki Engine + auto-update

`WikiStore` already serves search; what's missing is the *write* path that
makes the wiki grow over time without the user editing Markdown by hand.

### Deliverables

- [ ] **`WikiCompiler`** in `memvox/wiki/compiler.py`:
  - Single non-streaming LLM call to vLLM with the full session transcript
    + list of existing slugs
  - Pydantic-validated JSON output: array of
    `{slug, title, body, tags, action: "create" | "update"}`
  - Retry once on JSON parse failure with simplified prompt
  - Skip if transcript < 4 turns or < 100 tokens
  - Calls `WikiStore.upsert_article()` for each result
- [ ] **Fire-and-forget integration** in `SessionOrchestrator.stop()`:
  - Build `CompileRequest(session_id, turns, existing_slugs)`
  - `asyncio.create_task(wiki_compiler.compile(req))` — not awaited
  - Stop returns immediately; compilation finishes in background
- [ ] **Wire wiki search into LLM context** (currently empty in the orchestrator):
  - On each turn, search wiki with `transcript.text`, top_k=5
  - Inject `matched_chunks[:1]` per result as `context_snippets` in
    `GenerationRequest` — already plumbed, just need real content
- [ ] **Tests**:
  - Mock OpenAI returning fixed JSON → assert `WikiStore.upsert_article` calls
  - Pydantic validation failure → assert retry
  - Skip logic for short sessions
  - End-to-end: run a session, verify a `.md` file appears in `wiki_dir`

---

## Phase 4 — Polish

The "make it production-feeling" pass. Each item is independently scopable.

### Rust hot paths

- [ ] **Compile `memvox-rs` PyO3 extension** (`SentenceAccumulator`) via
      maturin; have `tts.py` switch from `_PySentenceAccumulator` automatically
- [ ] **Replace energy VAD with Silero VAD** in `memvox-audio/src/ingress.rs`:
  - Re-enable `ort` dependency (currently commented out)
  - Download `silero_vad.onnx`, manage LSTM state across frames
  - Better noise rejection, real probabilistic speech detection

### TTS

- [ ] **Faster Korean-capable TTS** to close the mouth-to-ear gap:
  - Evaluate MeloTTS, OpenVoice, or fine-tuned smaller XTTS variant
  - Target: <200 ms first chunk → mouth_to_ear <500 ms

### UX

- [ ] **React UI** for browser-based interaction:
  - WebSocket transport replacing the Unix socket pair
  - Optionally a small Axum-based audio bridge in Rust
- [ ] **Demo video**: 60-second screencap of a real Korean conversation with
      latency metrics overlaid

### Docs

- [ ] `README.md` rewrite (current state, install, quickstart)
- [ ] `ARCHITECTURE.md` updated with measured latencies and Phase 2 design
- [ ] One blog-post-length writeup of the project for the Anthropic application

---

## Phase 5 — Cloud tier scaffolding + multi-language

Ship as both an open-source self-hosted core and a managed tier. Out of scope
until earlier phases land cleanly.

- [ ] Multi-tenant orchestrator (per-user GPU partitioning)
- [ ] User wiki sync (CRDTs over the Markdown source of truth)
- [ ] Additional skins (Japanese tutor, Spanish, English debate coach)
- [ ] Auth + billing integration
- [ ] Inference provider abstraction (Anthropic API, OpenAI, OpenRouter, local)

---

## Conventions

- **Phase boundaries are commit boundaries.** A phase ships when its
  deliverables are checked off here AND the test suite is green. No
  half-finished phases.
- **Don't pre-build for future phases.** Phase 1 wired up Rust crates we
  didn't fully use (ort, memvox-rs); that was acceptable as part of
  scaffolding but generally adds drag. Land what the current phase needs.
- **Latency numbers in this doc are measured, not aspirational.** Update
  the snapshot section every time a phase closes.
