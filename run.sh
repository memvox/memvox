#!/usr/bin/env bash
#
# memvox launcher — one command to bring up the whole Phase-1 stack.
#
#   ./run.sh up   [skin]   install deps if needed, then start everything
#   ./run.sh down          stop everything this script started
#   ./run.sh status        show what's running
#   ./run.sh devices       list audio input/output device names
#   ./run.sh logs [name]   tail logs (name = audio | orchestrator | all)
#   ./run.sh setup         (re)install Python deps and build the audio binary
#
# It starts three things in the background and writes their logs/PIDs to .run/:
#   1. Ollama        (skipped if a native Ollama already answers on :11434;
#                     otherwise via docker compose, with the GPU overlay when
#                     an NVIDIA GPU is detected)
#   2. memvox-audio  (the Rust audio binary; falls back to shim.py if Rust
#                     isn't available)
#   3. orchestrator  (python -m memvox --skin <skin>)
#
# Cartesia (premium voice) is bring-your-own-key: put CARTESIA_API_KEY and
# CARTESIA_VOICE_ID in a .env file (see .env.example) and run with the
# cartesia_demo skin:  ./run.sh up cartesia_demo
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT/.run"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
AUDIO_BIN="$ROOT/target/release/memvox-audio"
OUT_SOCK="/tmp/memvox-audio-out.sock"
IN_SOCK="/tmp/memvox-audio-in.sock"
OLLAMA_URL="http://localhost:11434"
DC=""   # resolved by detect_compose(): "docker compose" (v2) or "docker-compose" (v1)

# Defaults (override via env or .env)
SKIN="${MEMVOX_SKIN:-korean_tutor}"
OLLAMA_MODEL="${OLLAMA_MODEL:-exaone3.5:7.8b}"

mkdir -p "$RUN_DIR"
# Load .env (Cartesia key, model overrides, etc.) into the environment.
if [[ -f "$ROOT/.env" ]]; then set -a; . "$ROOT/.env"; set +a; fi

# ── tiny output helpers ──────────────────────────────────────────────────────
log()  { printf '\033[1;36m[memvox]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[memvox]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[memvox]\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

is_running() { local pf="$RUN_DIR/$1.pid"; [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; }
ollama_up()  { curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }

# Resolve the Compose command. Prefer the v2 plugin ("docker compose"); fall
# back to the standalone "docker-compose" binary (common in Homebrew installs
# that lack the CLI plugin — using "docker compose" there errors with
# "unknown shorthand flag: -f"). Sets DC, or leaves it empty if neither exists.
detect_compose() {
  if docker compose version >/dev/null 2>&1; then DC="docker compose";
  elif have docker-compose; then DC="docker-compose";
  else DC=""; fi
}

# ── Python environment ───────────────────────────────────────────────────────
# Verify the modules the orchestrator imports at runtime are actually present —
# `import memvox` alone succeeds even when heavy deps failed to install, which
# otherwise surfaces as a cryptic ModuleNotFound mid-session.
verify_python_imports() {
  # Core deps every skin needs — hard-fail if any are missing.
  local missing
  missing="$("$PY" - <<'PY'
import importlib.util as u
mods = ["faster_whisper", "sounddevice", "openai", "lancedb",
        "sentence_transformers", "numpy", "pyarrow"]
print(" ".join(m for m in mods if u.find_spec(m) is None))
PY
)"
  [[ -z "$missing" ]] || die "Python modules missing after install: $missing
  Run a clean reinstall:  ./run.sh setup"
  # The local XTTS voice (coqui-tts) is heavy and flaky on Apple Silicon. It's
  # only needed by xtts skins, so warn rather than fail — Cartesia skins work
  # without it.
  if ! "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('TTS') else 1)" 2>/dev/null; then
    warn "coqui-tts (local XTTS voice) not importable — xtts skins won't work."
    warn "Use a Cartesia skin (./run.sh up cartesia_demo) or reinstall to fix."
  fi
}

ensure_python() {
  have python3 || die "python3 not found. Install Python >= 3.11 first."
  if [[ ! -x "$PY" ]]; then
    log "creating virtualenv (.venv)…"
    python3 -m venv "$VENV"
  fi
  local extra=""
  [[ -n "${CARTESIA_API_KEY:-}" ]] && extra="[cartesia]"
  local stamp="$RUN_DIR/deps.ok"
  # (Re)install when forced, never installed, or pyproject changed since.
  if [[ "${1:-}" == "--force" || ! -f "$stamp" || "$ROOT/pyproject.toml" -nt "$stamp" ]]; then
    log "installing Python dependencies (pip install -e .${extra})…"
    log "first run downloads PyTorch + models and can take several minutes."
    "$PIP" install --upgrade pip >/dev/null
    "$PIP" install -e ".${extra}" || die "pip install failed — see the error above.
  On Apple Silicon, coqui-tts (local XTTS voice) is the usual culprit. If you
  have a Cartesia key, skip it entirely with:  ./run.sh up cartesia_demo"
    touch "$stamp"
  fi
  verify_python_imports
}

# The pure-Python shim needs webrtcvad + sounddevice, which are NOT base deps.
ensure_shim_deps() {
  "$PY" -c "import webrtcvad, sounddevice" 2>/dev/null && return
  log "installing audio-shim deps (webrtcvad, sounddevice)…"
  "$PIP" install webrtcvad sounddevice
}

# ── Rust toolchain (auto-install via rustup if missing) ──────────────────────
ensure_rust() {
  have cargo && return 0
  [[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"
  have cargo && return 0
  have curl || { warn "curl is required to auto-install Rust." >&2; return 1; }
  log "Rust/cargo not found — installing via rustup (non-interactive)…" >&2
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y >&2 || return 1
  . "$HOME/.cargo/env"
  have cargo
}

# ── Audio backend (Rust binary, else build it, else shim) ────────────────────
audio_cmd() {
  if [[ "${MEMVOX_AUDIO:-}" == "shim" ]]; then
    ensure_shim_deps >&2; echo "$PY $ROOT/shim.py"; return
  fi
  if [[ -x "$AUDIO_BIN" ]]; then echo "$AUDIO_BIN"; return; fi
  if ensure_rust; then
    log "building memvox-audio (cargo build --release)…" >&2
    if ( cd "$ROOT" && cargo build --release --bin memvox-audio ) >&2; then
      echo "$AUDIO_BIN"; return
    fi
    warn "cargo build failed (on macOS this usually means the Xcode command-line" >&2
    warn "tools are missing — run: xcode-select --install). Using the shim." >&2
  fi
  warn "Falling back to the pure-Python audio shim." >&2
  ensure_shim_deps >&2
  echo "$PY $ROOT/shim.py"
}

# ── Ollama (native if present, otherwise docker compose) ─────────────────────
ensure_ollama() {
  if ollama_up; then
    log "Ollama already serving on :11434 — using it (no container)."
  else
    have docker || die "Ollama isn't running and docker isn't installed.
  Install Docker, or start Ollama natively (https://ollama.com)."
    detect_compose
    [[ -n "$DC" ]] || die "Docker is installed but Docker Compose isn't.
  Install Compose v2 (Docker Desktop) or the 'docker-compose' binary,
  or just run Ollama natively (https://ollama.com) and re-run ./run.sh up."
    local files=(-f "$ROOT/docker-compose.yml")
    if have nvidia-smi; then
      files+=(-f "$ROOT/docker-compose.gpu.yml")
      log "NVIDIA GPU detected — using GPU compose overlay."
    fi
    log "starting Ollama via docker compose…"
    ( cd "$ROOT" && $DC "${files[@]}" up -d ollama )
    touch "$RUN_DIR/docker.started"
    log "waiting for Ollama to become healthy…"
    for _ in $(seq 1 60); do ollama_up && break; sleep 1; done
    ollama_up || die "Ollama did not come up within 60s. Check: $DC logs ollama"
  fi

  # Make sure the model the skin expects is present.
  if curl -fsS "$OLLAMA_URL/api/tags" | grep -q "\"$OLLAMA_MODEL\""; then
    log "model '$OLLAMA_MODEL' present."
  else
    log "pulling model '$OLLAMA_MODEL' (one-time; this can take a while)…"
    if [[ -f "$RUN_DIR/docker.started" ]]; then
      ( cd "$ROOT" && $DC exec -T ollama ollama pull "$OLLAMA_MODEL" )
    elif have ollama; then
      ollama pull "$OLLAMA_MODEL"
    else
      die "Can't pull '$OLLAMA_MODEL': no docker stack and no native ollama CLI."
    fi
  fi
}

# ── process starters ─────────────────────────────────────────────────────────
start_audio() {
  if is_running audio; then log "audio already running (pid $(cat "$RUN_DIR/audio.pid"))."; return; fi
  local cmd; cmd="$(audio_cmd)"
  rm -f "$OUT_SOCK" "$IN_SOCK"
  # Optional explicit device selection (substring match). Only the Rust binary
  # accepts these flags; the shim ignores device choice.
  local dev=()
  if [[ "$cmd" == *memvox-audio* ]]; then
    [[ -n "${MEMVOX_INPUT_DEVICE:-}"  ]] && dev+=(--input-device  "$MEMVOX_INPUT_DEVICE")
    [[ -n "${MEMVOX_OUTPUT_DEVICE:-}" ]] && dev+=(--output-device "$MEMVOX_OUTPUT_DEVICE")
  elif [[ -n "${MEMVOX_INPUT_DEVICE:-}${MEMVOX_OUTPUT_DEVICE:-}" ]]; then
    warn "MEMVOX_INPUT_DEVICE/OUTPUT_DEVICE are ignored by the Python shim."
  fi
  log "starting audio: $cmd ${dev[*]:-}"
  nohup $cmd ${dev[@]+"${dev[@]}"} >"$RUN_DIR/audio.log" 2>&1 &
  echo $! >"$RUN_DIR/audio.pid"
  log "waiting for audio sockets…"
  for _ in $(seq 1 30); do
    [[ -S "$OUT_SOCK" && -S "$IN_SOCK" ]] && return
    is_running audio || die "audio process exited early. See $RUN_DIR/audio.log"
    sleep 0.5
  done
  die "audio sockets never appeared. See $RUN_DIR/audio.log"
}

start_orchestrator() {
  if is_running orchestrator; then log "orchestrator already running."; return; fi
  log "starting orchestrator (skin: $SKIN)…"
  nohup "$PY" -m memvox --skin "$SKIN" >"$RUN_DIR/orchestrator.log" 2>&1 &
  echo $! >"$RUN_DIR/orchestrator.pid"
  sleep 2
  is_running orchestrator || die "orchestrator exited early. See $RUN_DIR/orchestrator.log"
}

# ── subcommands ──────────────────────────────────────────────────────────────
cmd_up() {
  [[ -n "${1:-}" ]] && SKIN="$1"
  ensure_python
  ensure_ollama
  start_audio
  start_orchestrator
  echo
  log "✅ memvox is up. Speak into your mic."
  log "   skin   : $SKIN"
  log "   logs   : ./run.sh logs        (audio + orchestrator)"
  log "   stop   : ./run.sh down"
}

cmd_down() {
  for name in orchestrator audio; do
    if is_running "$name"; then
      log "stopping $name…"; kill "$(cat "$RUN_DIR/$name.pid")" 2>/dev/null || true
    fi
    rm -f "$RUN_DIR/$name.pid"
  done
  if [[ -f "$RUN_DIR/docker.started" ]]; then
    detect_compose
    log "stopping Ollama container…"
    [[ -n "$DC" ]] && ( cd "$ROOT" && $DC down ) || true
    rm -f "$RUN_DIR/docker.started"
  fi
  log "✅ stopped."
}

cmd_status() {
  for name in audio orchestrator; do
    if is_running "$name"; then printf '  %-13s up   (pid %s)\n' "$name" "$(cat "$RUN_DIR/$name.pid")";
    else printf '  %-13s down\n' "$name"; fi
  done
  if ollama_up; then printf '  %-13s up   (%s)\n' "ollama" "$OLLAMA_URL"; else printf '  %-13s down\n' "ollama"; fi
}

cmd_devices() {
  # Authoritative list from the Rust binary (cpal names) if it's built;
  # otherwise a sounddevice fallback. Names are matched as substrings by
  # MEMVOX_INPUT_DEVICE / MEMVOX_OUTPUT_DEVICE.
  if [[ -x "$AUDIO_BIN" ]]; then
    "$AUDIO_BIN" --list-devices
    return
  fi
  if [[ -x "$PY" ]] && "$PY" -c "import sounddevice" >/dev/null 2>&1; then
    "$PY" - <<'PY'
import sounddevice as sd
print("Audio devices (use any substring of the name):")
for d in sd.query_devices():
    tags = []
    if d["max_input_channels"]  > 0: tags.append("mic")
    if d["max_output_channels"] > 0: tags.append("speaker")
    print(f"  {d['name']}  [{', '.join(tags)}]")
print("\nLeave MEMVOX_INPUT_DEVICE/OUTPUT_DEVICE unset to use the system default.")
PY
    return
  fi
  die "Audio tooling not installed yet. Run ./run.sh up (or setup) first."
}

cmd_logs() {
  local which="${1:-all}" files=()
  case "$which" in
    audio)        files=("$RUN_DIR/audio.log");;
    orchestrator) files=("$RUN_DIR/orchestrator.log");;
    all)          files=("$RUN_DIR/audio.log" "$RUN_DIR/orchestrator.log");;
    *) die "unknown log '$which' (use: audio | orchestrator | all)";;
  esac
  tail -n 40 -f "${files[@]}"
}

case "${1:-up}" in
  up)     shift || true; cmd_up "${1:-}";;
  down)   cmd_down;;
  status)  cmd_status;;
  devices) cmd_devices;;
  logs)    shift || true; cmd_logs "${1:-all}";;
  setup)   ensure_python --force; audio_cmd >/dev/null; log "✅ setup complete.";;
  *) die "usage: ./run.sh [up [skin] | down | status | devices | logs [name] | setup]";;
esac
