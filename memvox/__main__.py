"""
memvox entry point — wires a SessionConfig into a running SessionOrchestrator.

Usage:
  python -m memvox                              # defaults: korean_tutor skin
  python -m memvox --skin korean_tutor
  python -m memvox --wiki-dir ~/memvox/wiki

This connects to the audio I/O sockets exposed by either:
  • shim.py        (dev — pure Python sounddevice + webrtcvad)
  • memvox-audio   (prod — Rust binary with cpal + Silero VAD + rubato)
Both speak the same length-prefixed bincode protocol.
"""

import argparse
import asyncio
import importlib
import os
import signal
from pathlib import Path

from memvox.session.orchestrator import SessionOrchestrator
from memvox.session.types import SessionConfig
from memvox.voice.asr import ASREngine
from memvox.voice.egress import AudioEgressClient
from memvox.voice.ingress import AudioIngressClient
from memvox.voice.llm import LLMEngine
from memvox.voice.tts import TTSEngine
from memvox.wiki.store import WikiStore


DEFAULT_OUTBOUND_SOCK = "/tmp/memvox-audio-out.sock"
DEFAULT_INBOUND_SOCK  = "/tmp/memvox-audio-in.sock"
DEFAULT_WIKI_DIR      = "~/memvox/wiki"
DEFAULT_DB_PATH       = "~/memvox/wiki.lance"


def _load_skin(name: str) -> SessionConfig:
    """Import memvox.skins.<name> and call <name>() to get a SessionConfig."""
    module = importlib.import_module(f"memvox.skins.{name}")
    factory = getattr(module, name)
    return factory()


async def _run(args: argparse.Namespace) -> None:
    config = _load_skin(args.skin)

    wiki_dir = Path(os.path.expanduser(args.wiki_dir))
    db_path  = Path(os.path.expanduser(args.db_path))
    wiki_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Components ────────────────────────────────────────────────────────────
    # ASR queues are unused in Phase 1 sequential mode (orchestrator calls
    # transcribe() directly), but the constructor still requires them.
    asr_in: asyncio.Queue  = asyncio.Queue()
    asr_out: asyncio.Queue = asyncio.Queue()

    asr  = ASREngine(asr_in, asr_out, model_name=config.asr_model)
    llm  = LLMEngine(base_url=config.llm_base_url, model=config.llm_model)
    tts  = TTSEngine(voice=config.voice, lang_code=config.tts_lang_code)
    wiki = WikiStore(wiki_dir=wiki_dir, db_path=db_path)
    await wiki.initialize()

    ingress = AudioIngressClient(args.out_sock)
    egress  = AudioEgressClient(args.in_sock)

    orch = SessionOrchestrator(config, asr, llm, tts, wiki, ingress, egress)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    print(f"[memvox] skin     : {args.skin}")
    print(f"[memvox] wiki dir : {wiki_dir}")
    print(f"[memvox] LLM      : {config.llm_model} @ {config.llm_base_url}")
    print(f"[memvox] sockets  : out={args.out_sock}  in={args.in_sock}")
    print("[memvox] Starting session — speak into the mic. Ctrl-C to stop.")

    await orch.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    print("\n[memvox] Stopping session…")
    await orch.stop()
    print("[memvox] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="memvox")
    parser.add_argument("--skin",     default="korean_tutor", help="skin module name under memvox.skins")
    parser.add_argument("--wiki-dir", default=DEFAULT_WIKI_DIR)
    parser.add_argument("--db-path",  default=DEFAULT_DB_PATH)
    parser.add_argument("--out-sock", default=DEFAULT_OUTBOUND_SOCK, help="audio binary's outbound socket (shim writes here)")
    parser.add_argument("--in-sock",  default=DEFAULT_INBOUND_SOCK,  help="audio binary's inbound socket (shim reads here)")
    args = parser.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
