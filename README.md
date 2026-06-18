# memvox

![memvox-header](./assets/memvox-header.png)

**A local-first, low-latency streaming voice agent with persistent wiki memory.**

memvox is a spoken-conversation agent — built first as a Korean tutor — that runs
the entire voice loop on your own machine by default: speech recognition, a local
LLM, and speech synthesis, with optional bring-your-own-key cloud voices. What
makes it interesting is the engineering underneath: a streaming, barge-in-capable
pipeline that overlaps recognition, generation, and playback, and an LLM that
maintains its own Markdown "wiki" of what you've discussed so memory persists
across sessions.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Foundations (types, metrics, wiki store) | ✅ shipped |
| 1 | Sequential voice loop (ASR → LLM → TTS) | ✅ shipped |
| 2 | Overlapping orchestrator + barge-in | ⏳ in progress |
| 3 | LLM wiki auto-update (write path) | 📋 planned |
| 4 | Polish (faster TTS, Rust hot paths, UI) | 📋 planned |
| 5 | Cloud tier + more languages | 📋 planned |

Phase 1 is validated end-to-end through both the Rust audio binary and the
Python dev shim. See [`DEVPLAN.md`](./DEVPLAN.md) for the detailed, checkable plan.

## Architecture

Two processes talk over a pair of Unix sockets. A **Rust audio process** owns the
hardware and the real-time path; a **Python orchestrator** owns the turn logic
and the AI engines. Inference (LLM/TTS) sits behind streaming interfaces, local
or cloud.

```
 ┌──── Rust audio process (memvox-audio) ────┐   ┌──────── Python orchestrator ────────┐
 │                                            │   │                                      │
 mic ─▶ cpal capture ─▶ VAD ─▶ SpeechSegment ─┼──▶│ Whisper ASR ─▶ wiki lookup ─▶ LLM ─┐ │
 │                                            │   │ (faster-whisper) (LanceDB)         │ │
 speaker ◀─ cpal playback ◀─ AudioChunk ◀─────┼───│ ◀──────────────── TTS ◀────────────┘ │
 │           + rubato resample                │   │              (XTTS / Cartesia)        │
 └────────────────────────────────────────────┘   └───────────────┬──────────────────────┘
        ▲ barge-in: VAD onset flushes playback                     │ HTTP :11434 / WebSocket
        └────────── within one audio frame ──────────              ▼
   IPC: [u32 LE length][bincode payload] over two Unix sockets   Ollama / vLLM · Cartesia
```

**Components**

- **Audio I/O (Rust, `memvox-audio`)** — `cpal` mic capture and playback, VAD,
  `rubato` resampling, and the barge-in signal. Runs on a real-time thread.
- **VAD (Rust)** — voice-activity detection in the audio process (energy VAD in
  Phase 1; Silero ONNX in Phase 4) emits speech-start / speech-end events.
- **ASR (Python)** — Whisper `large-v3` via faster-whisper, GPU with a CPU
  fallback; Korean + English with a language allow-list to filter hallucinations.
- **Retrieval (LanceDB)** — hybrid vector + BM25 search over a Markdown wiki,
  queried every turn; Markdown is the source of truth, the index is rebuilt on demand.
- **LLM (pluggable)** — any OpenAI-compatible endpoint; local-first via Ollama or
  vLLM, with streaming and mid-turn cancellation for barge-in.
- **TTS (pluggable)** — local Coqui XTTS-v2 by default, or Cartesia Sonic over
  WebSocket (bring-your-own-key). Both take a *token stream* in and emit audio chunks.
- **Orchestrator (Python, asyncio)** — owns conversation state, turn detection,
  retrieval, and cancellation; depends only on the streaming Protocols, never a
  concrete backend.

**The turn pipeline:** end-of-utterance (VAD) → ASR transcript → wiki retrieval →
LLM tokens → sentence-chunked TTS → audio chunks → playback. In Phase 1 these run
sequentially; Phase 2 runs them concurrently with bounded back-pressure.

## Design decisions

- **Rust for audio, Python for orchestration.** The audio path needs real-time
  thread priority and deterministic latency — a Python GIL pause mid-playback is
  unacceptable. Rust also lets barge-in cancel playback within a single `cpal`
  frame (sub-millisecond, via an in-process watch channel) instead of an IPC
  round-trip. The orchestrator stays in Python where the AI ecosystem lives.
- **Unix sockets + length-prefixed bincode for IPC.** A tiny fixed wire format —
  `[u32 LE length][bincode payload]`, `u32` enum tag, `u64` vector lengths — keeps
  the two processes isolated and each in its best language, with no WebRTC/browser
  overhead in v1. The round-trip is verified Python↔Python and Python↔Rust.
- **Pluggable engines behind streaming Protocols.** LLM and TTS are swapped by a
  one-line skin config, not code changes; the orchestrator only sees the Protocol.
  This is what lets the same pipeline run fully local or with cloud voice, and it
  keeps cancellation uniform across backends.
- **A wiki layer with a one-directional boundary.** `wiki/` imports nothing from
  `voice/` or `session/` (even `ChatMessage` lives in `wiki/types.py`), so the
  memory layer is independently extractable. Memory is plain Markdown the user can
  read and edit; the vector index is derived, not authoritative.
- **Only the inference layer is containerized.** The orchestrator and Rust audio
  binary stay on the host (they need mic/speaker and socket access); Docker is
  used just to run Ollama on machines without a native install.

## Performance

Honest baseline — **Phase 1, sequential mode, RTX 5090** (LLM: Qwen3-8B via vLLM;
TTS: XTTS-v2). These are *measured*, not aspirational:

| Stage | Avg | P95 | Notes |
|---|---:|---:|---|
| **`mouth_to_ear`** | **1241 ms** | **2072 ms** | end-of-utterance → first audio out |
| `asr.transcribe` | 185 ms | 295 ms | Whisper large-v3 |
| `wiki.query` | 18 ms | 105 ms | LanceDB hybrid (empty wiki) |
| `llm.ttft` | 37 ms | 119 ms | time to first token, thinking disabled |
| `tts.first_chunk` | 683 ms | 1163 ms | XTTS-v2 (Korean-capable, heavier) |

**TTS first-chunk is the dominant cost.** The architecture target is <400 ms
mouth-to-ear, which assumed a faster TTS (~150 ms first chunk) *and* Phase 2
overlap. Getting there needs both: Phase 2 removes the LLM-completion wait from
the critical path (targeting ~900 ms avg), and Phase 4 swaps in a faster
Korean-capable TTS to close the rest. Numbers will be updated as each phase lands.

## Roadmap

- **Phase 2 — Overlapping orchestrator + barge-in** *(in progress)*: three
  concurrent asyncio tasks (ASR/LLM → TTS → playback) joined by bounded queues;
  LLM tokens stream straight into TTS; a barge-in monitor that `aclose()`s the
  in-flight LLM/TTS streams and flushes playback in <50 ms.
- **Phase 3 — Wiki write path**: a `WikiCompiler` that turns each session
  transcript into create/update operations on the Markdown wiki, fire-and-forget
  on session end, plus wiring retrieved snippets into LLM context.
- **Phase 4 — Polish**: compiled Rust `SentenceAccumulator`, Silero VAD, a faster
  TTS to hit the latency target, and a browser UI.
- **Phase 5 — Cloud tier**: multi-tenant orchestration, wiki sync, and more skins
  (Japanese, Spanish, debate coach).

## Quick start

You need **Python ≥ 3.11** and either a running **Ollama** or **Docker** (the
launcher uses Docker to run Ollama if you don't already have it; both the
`docker compose` v2 plugin and the standalone `docker-compose` v1 binary work).
For the Rust audio binary you also need **Rust/cargo** — if it's missing, the
launcher falls back to a pure-Python audio shim automatically.

```bash
git clone <repo> && cd memvox
./run.sh up
```

That single command will:

1. create a virtualenv and install Python dependencies,
2. start **Ollama** (reusing a native one on `:11434`, else via Docker — with a
   GPU overlay when an NVIDIA GPU is detected) and pull the model,
3. build & start the **Rust audio** binary (or the shim),
4. start the **orchestrator**.

When it prints `✅ memvox is up`, just **speak into your mic**. By default memvox
uses your operating system's **default** input and output devices — so the
zero-config path is to pick your mic/speaker in **System Settings → Sound**
(macOS) and run; you don't need to know any device names.

```bash
./run.sh status     # what's running
./run.sh devices    # list audio input/output device names (only needed to override)
./run.sh logs       # tail audio + orchestrator logs (Ctrl-C to stop tailing)
./run.sh down       # stop everything the launcher started
```

### macOS / Apple Silicon: use native Ollama, not Docker

Docker on macOS runs Linux containers inside a VM with **no GPU passthrough**, so
a containerized Ollama is **CPU-only** and throws away your Apple Silicon (Metal)
acceleration — bad for voice latency. Install Ollama natively instead:

```bash
brew install ollama
ollama serve            # or: brew services start ollama
./run.sh up
```

`run.sh` probes `:11434` first and **skips Docker entirely** when a native Ollama
is already serving, so you get Metal acceleration and avoid Compose setup
altogether. (Docker-run Ollama is meant for Linux hosts, ideally with an NVIDIA
GPU.)

### Premium voice (Cartesia, optional)

The default skin (`korean_tutor`) is fully local and needs no API keys. For
Cartesia Sonic voices, bring your own key:

```bash
cp .env.example .env        # then edit:
#   CARTESIA_API_KEY=sk_car_...
#   CARTESIA_VOICE_ID=<a voice UUID from https://play.cartesia.ai/>

./run.sh up cartesia_demo
```

The `cartesia_demo` skin is a bilingual Korean tutor: it talks mostly in English
and teaches Korean phrases (rendered slowly so you can repeat them), and you can
say *"let's do Korean only"* to switch to immersion mode mid-conversation.

### Configuration

`run.sh` reads `.env` and these environment variables:

| Variable               | Default          | Purpose                                  |
|------------------------|------------------|------------------------------------------|
| `MEMVOX_SKIN`          | `korean_tutor`   | Skin to run (or pass as `./run.sh up <skin>`) |
| `OLLAMA_MODEL`         | `exaone3.5:7.8b` | Model the launcher ensures is pulled     |
| `MEMVOX_AUDIO`         | _(auto)_         | Set to `shim` to force the Python audio shim |
| `MEMVOX_INPUT_DEVICE`  | _(system default)_ | Mic device (substring match; Rust binary only) |
| `MEMVOX_OUTPUT_DEVICE` | _(system default)_ | Speaker device (substring match; Rust binary only) |

> **Note:** `OLLAMA_MODEL` must match the model the chosen skin expects. The
> launcher auto-installs Python deps and, if `cargo` is missing, installs Rust
> via `rustup` to build the audio binary (falling back to the Python shim).

### Bluetooth earbuds (AirPods, etc.)

Avoid using Bluetooth earbuds as the **microphone**. When an app opens their mic,
macOS switches them from the high-quality A2DP (output-only) profile to the
HFP/hands-free profile, which forces *both* directions down to a ~16 kHz mono
"telephone" codec — so your playback suddenly sounds muffled and glitchy, on top
of Bluetooth's added latency.

Use a **non-Bluetooth mic for input** (the built-in mic on laptops/iMacs, or a
USB/wired mic — note the Mac mini and Mac Studio have *no* built-in mic) and keep
the earbuds for output, which stays on the high-quality A2DP profile:

```bash
MEMVOX_INPUT_DEVICE="USB" MEMVOX_OUTPUT_DEVICE="AirPods" ./run.sh up
```

If your only mic is the earbuds, both directions are stuck in the degraded HFP
profile — there's no way around it while their mic is in use; sending output to a
wired speaker instead at least keeps playback off the Bluetooth path. A **USB or
USB-C headset** (e.g. a Jabra) sidesteps all of this — it's a USB-audio device,
not Bluetooth, so mic and speaker both run at full quality with no profile
switching. Ideal for a Mac mini.

(List device names with `./run.sh devices`.)

## Architecture details

`./run.sh` is just a wrapper around the individual processes. For the full design
— component interfaces, the IPC wire format, and per-process run instructions —
see [`ARCHITECTURE.md`](./ARCHITECTURE.md), [`DEVPLAN.md`](./DEVPLAN.md), and the
comments in `docker-compose.yml`, `shim.py`, and `memvox/__main__.py`.
