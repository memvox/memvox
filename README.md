# memvox

![memvox-header](./assets/memvox-header.png)

A low-latency, streaming voice agent with persistent wiki memory. Phase 1 runs a
full speak-listen loop locally: **mic → VAD → Whisper ASR → LLM → TTS → speaker**.

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
GPU — see the GPU overlay below.)

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

## Premium voice (Cartesia, optional)

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

## Configuration

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

## Manual / advanced

`./run.sh` is just a wrapper. To run pieces by hand, see the per-process docs in
[`ARCHITECTURE.md`](./ARCHITECTURE.md) and the comments in `docker-compose.yml`,
`shim.py`, and `memvox/__main__.py`.
