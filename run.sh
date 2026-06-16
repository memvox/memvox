#!/usr/bin/env bash
#
# memvox launcher — one command to bring up the whole Phase-1 stack.
#
#   ./run.sh up   [skin]   install deps if needed, then start everything
#   ./run.sh down          stop everything this script started
#   ./run.sh status        show what's running
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
ensure_python() {
  have python3 || die "python3 not found. Install Python >= 3.11 first."
  if [[ ! -x "$PY" ]]; then
    log "creating virtualenv (.venv)…"
    python3 -m venv "$VENV"
  fi
  # Reinstall when forced, or when the package can't be imported yet.
  if [[ "${1:-}" == "--force" ]] || ! "$PY" -c "import memvox" >/dev/null 2>&1; then
    local extra=""
    [[ -n "${CARTESIA_API_KEY:-}" ]] && extra="[cartesia]"
    log "installing Python dependencies (pip install -e .${extra})…"
    "$PIP" install --quiet --upgrade pip
    "$PIP" install --quiet -e ".${extra}"
  fi
}

# ── Audio backend (Rust binary, else build it, else shim) ────────────────────
audio_cmd() {
  case "${MEMVOX_AUDIO:-}" in
    shim) echo "$PY $ROOT/shim.py"; return;;
  esac
  if [[ -x "$AUDIO_BIN" ]]; then
    echo "$AUDIO_BIN"
  elif have cargo; then
    log "building memvox-audio (cargo build --release)…" >&2
    ( cd "$ROOT" && cargo build --release --bin memvox-audio ) >&2
    echo "$AUDIO_BIN"
  elif [[ -f "$ROOT/shim.py" ]]; then
    warn "Rust/cargo not found — falling back to the pure-Python audio shim." >&2
    echo "$PY $ROOT/shim.py"
  else
    die "No audio backend available (need the Rust binary, cargo, or shim.py)."
  fi
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
  log "starting audio: $cmd"
  nohup $cmd >"$RUN_DIR/audio.log" 2>&1 &
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
  status) cmd_status;;
  logs)   shift || true; cmd_logs "${1:-all}";;
  setup)  ensure_python --force; audio_cmd >/dev/null; log "✅ setup complete.";;
  *) die "usage: ./run.sh [up [skin] | down | status | logs [name] | setup]";;
esac
