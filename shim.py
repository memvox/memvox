#!/usr/bin/env python3
"""
shim.py — Python audio stand-in for the memvox-audio Rust binary.

╔══════════════════════════════════════════════════════════════════════════════╗
║  DEVELOPMENT / DEMO USE ONLY — NOT FOR PRODUCTION                          ║
║                                                                              ║
║  This file is a temporary drop-in for the memvox-audio Rust binary while    ║
║  that binary is still being built.  It speaks the same Unix socket IPC      ║
║  protocol (length-prefixed bincode messages) so the Python orchestrator      ║
║  connects to it identically.                                                 ║
║                                                                              ║
║  Limitations vs. the real Rust binary (memvox-audio):                       ║
║    • No real-time OS thread priority — mic/playback can be preempted by      ║
║      the Python GIL and OS scheduler in ways that a dedicated Rust thread    ║
║      would not be.                                                           ║
║    • CancelPlayback latency is ~10–50 ms (sd.stop() call from asyncio)       ║
║      vs. <1 ms for the internal tokio watch channel in Rust.                 ║
║    • No rubato resampling — PortAudio resamples if device rate ≠ 24 kHz.    ║
║    • PausePlayback / ResumePlayback are no-ops (sounddevice has no pause     ║
║      API for the play() call; use the Rust binary for barge-in demos).       ║
║    • speech_prob in SpeechSegment is a fixed 0.9 — webrtcvad is binary       ║
║      (speech / not speech), not probabilistic like Silero VAD.               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
-----
  # Terminal 1 — start the audio shim
  python shim.py

  # Terminal 2 — start the orchestrator (entry point TBD)
  python -m memvox ...

The shim opens two Unix socket servers that the orchestrator's
AudioIngressClient and AudioEgressClient connect to as clients:

  Outbound socket (shim → orchestrator):
    Sends SpeechStarted and SpeechSegment frames as the mic captures speech.

  Inbound socket (orchestrator → shim):
    Receives AudioChunk frames from TTS and plays them; handles CancelPlayback.
"""

import argparse
import asyncio
import queue
import struct
import threading
import time
import os
import sys

import numpy as np
import sounddevice as sd
import webrtcvad


# ── Socket paths ──────────────────────────────────────────────────────────────
# These must match the paths passed to AudioIngressClient / AudioEgressClient
# in your SessionOrchestrator setup.

OUTBOUND_SOCK_DEFAULT = "/tmp/memvox-audio-out.sock"  # shim writes, orchestrator reads
INBOUND_SOCK_DEFAULT  = "/tmp/memvox-audio-in.sock"   # orchestrator writes, shim reads


# ── Microphone / VAD configuration ───────────────────────────────────────────

MIC_SAMPLE_RATE   = 16_000     # Hz — Whisper and Silero both expect 16 kHz mono
MIC_CHANNELS      = 1
VAD_FRAME_MS      = 30         # webrtcvad only accepts 10, 20, or 30 ms frames
VAD_FRAME_SAMPLES = MIC_SAMPLE_RATE * VAD_FRAME_MS // 1000   # 480 samples = 960 bytes
VAD_AGGRESSIVENESS = 2         # 0 (least aggressive) … 3 (most aggressive)

ONSET_FRAMES    = 2    # consecutive speech frames required to confirm utterance start
TRAILING_FRAMES = 15   # consecutive silence frames to end utterance (15 × 30 ms = 450 ms)
MIN_SPEECH_FRAMES = 3  # segments shorter than this are discarded (< 90 ms = likely noise)


# ── Wire format tags — must match ipc.rs exactly ─────────────────────────────

_TAG_SPEECH_STARTED = 0   # outbound: SpeechStarted { timestamp_ms: u64 }
_TAG_SPEECH_SEGMENT = 1   # outbound: SpeechSegment { Vec<i16>, f32, f32, u64 }
_TAG_AUDIO_CHUNK    = 0   # inbound:  AudioChunk { Vec<f32>, u32, bool }
_TAG_CANCEL         = 1   # inbound:  CancelPlayback
_TAG_PAUSE          = 2   # inbound:  PausePlayback  (no-op in shim)
_TAG_RESUME         = 3   # inbound:  ResumePlayback (no-op in shim)


# ── Wire format helpers ───────────────────────────────────────────────────────

def _frame(payload: bytes) -> bytes:
    """Prepend 4-byte little-endian length prefix (matches ipc.rs framing)."""
    return struct.pack("<I", len(payload)) + payload


def _encode_speech_started(ts_ms: int) -> bytes:
    # OutboundMsg::SpeechStarted: tag(u32) + timestamp_ms(u64)
    return struct.pack("<IQ", _TAG_SPEECH_STARTED, ts_ms)


def _encode_speech_segment(
    audio_bytes: bytes,  # raw int16 PCM at MIC_SAMPLE_RATE
    speech_prob: float,
    duration_ms: float,
    ts_ms: int,
) -> bytes:
    # OutboundMsg::SpeechSegment: tag(u32) + n(u64) + i16*n + f32 + f32 + u64
    n = len(audio_bytes) // 2   # number of int16 samples
    return (
        struct.pack("<IQ", _TAG_SPEECH_SEGMENT, n)
        + audio_bytes
        + struct.pack("<ff", speech_prob, duration_ms)
        + struct.pack("<Q", ts_ms)
    )


def _decode_audio_chunk(payload: bytes) -> tuple[np.ndarray, int, bool]:
    """Parse InboundMsg::AudioChunk payload (tag already consumed)."""
    # payload layout after tag: n(u64) + f32*n + sample_rate(u32) + is_final(bool)
    n, = struct.unpack_from("<Q", payload, 0)
    pcm = np.frombuffer(payload[8 : 8 + n * 4], dtype=np.float32).copy()
    sr, is_final = struct.unpack_from("<I?", payload, 8 + n * 4)
    return pcm, sr, is_final


# ── VAD state machine ─────────────────────────────────────────────────────────

class _VadStateMachine:
    """
    Four-state machine driven by per-frame webrtcvad binary decisions.

    SILENT   ──(ONSET_FRAMES consecutive speech)──▶ SPEAKING
    SPEAKING ──(any non-speech frame)──────────────▶ TRAILING
    TRAILING ──(any speech frame)──────────────────▶ SPEAKING
    TRAILING ──(TRAILING_FRAMES silence frames)────▶ SILENT  (+ emit segment)
    """

    def __init__(self) -> None:
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._state = "SILENT"
        self._speech_frames: list[bytes] = []
        self._onset_count   = 0
        self._trailing_count = 0
        self._onset_ts_ms   = 0

    def reset(self) -> None:
        self._state = "SILENT"
        self._speech_frames.clear()
        self._onset_count    = 0
        self._trailing_count = 0
        self._onset_ts_ms    = 0

    def push(self, frame: bytes, ts_ms: int) -> tuple[str | None, bytes | None, int]:
        """
        Process one VAD frame (exactly VAD_FRAME_SAMPLES * 2 bytes of int16).

        Returns (event, audio, onset_ts_ms):
          ("onset",   None,       onset_ts_ms) — speech has begun
          ("segment", pcm_bytes,  onset_ts_ms) — utterance complete, pcm_bytes ready
          (None,      None,       0)           — no event this frame
        """
        is_speech = self._vad.is_speech(frame, MIC_SAMPLE_RATE)

        if self._state == "SILENT":
            if is_speech:
                self._onset_count += 1
                self._speech_frames.append(frame)
                if self._onset_count >= ONSET_FRAMES:
                    self._state = "SPEAKING"
                    self._onset_ts_ms = ts_ms
                    return "onset", None, self._onset_ts_ms
            else:
                self._onset_count = 0
                self._speech_frames.clear()

        elif self._state == "SPEAKING":
            self._speech_frames.append(frame)
            if not is_speech:
                self._state = "TRAILING"
                self._trailing_count = 1

        elif self._state == "TRAILING":
            self._speech_frames.append(frame)
            if is_speech:
                self._state = "SPEAKING"
                self._trailing_count = 0
            else:
                self._trailing_count += 1
                if self._trailing_count >= TRAILING_FRAMES:
                    audio    = b"".join(self._speech_frames)
                    onset_ts = self._onset_ts_ms
                    n_frames = len(self._speech_frames)
                    self.reset()
                    if n_frames >= MIN_SPEECH_FRAMES:
                        return "segment", audio, onset_ts
                    # Too short — likely noise; discard silently
                    return None, None, 0

        return None, None, 0

    def drain(self) -> tuple[bytes | None, int]:
        """Force-emit any in-progress utterance (called on shim shutdown)."""
        if self._state in ("SPEAKING", "TRAILING") and len(self._speech_frames) >= MIN_SPEECH_FRAMES:
            audio = b"".join(self._speech_frames)
            onset_ts = self._onset_ts_ms
            self.reset()
            return audio, onset_ts
        self.reset()
        return None, 0


# ── Mic capture thread ────────────────────────────────────────────────────────

def _mic_thread(
    loop: asyncio.AbstractEventLoop,
    out_q: asyncio.Queue,
    stop_event: threading.Event,
) -> None:
    """
    Runs as a daemon thread.  Opens the microphone via sounddevice, feeds
    30 ms frames into the VAD state machine, and schedules SpeechStarted /
    SpeechSegment payloads onto `out_q` using loop.call_soon_threadsafe so
    the async outbound sender can write them to the socket.
    """
    vad  = _VadStateMachine()
    mic_q: queue.Queue[bytes] = queue.Queue()

    def _callback(indata: bytes, frames: int, time_info, status) -> None:
        # sounddevice RawInputStream gives raw bytes; one call = blocksize frames
        mic_q.put(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="int16",
        blocksize=VAD_FRAME_SAMPLES,  # guarantees exactly one VAD frame per callback
        callback=_callback,
    )

    print("[shim] Microphone open — listening for speech")
    stream.start()

    try:
        while not stop_event.is_set():
            try:
                frame = mic_q.get(timeout=0.1)
            except queue.Empty:
                continue

            ts_ms = int(time.monotonic() * 1000)
            event, audio, onset_ts = vad.push(frame, ts_ms)

            if event == "onset":
                payload = _encode_speech_started(onset_ts)
                loop.call_soon_threadsafe(out_q.put_nowait, payload)
                print(f"[shim] ▶ SpeechStarted  ts={onset_ts} ms")

            elif event == "segment":
                n_samples   = len(audio) // 2
                duration_ms = n_samples / MIC_SAMPLE_RATE * 1000
                payload = _encode_speech_segment(
                    audio_bytes=audio,
                    speech_prob=0.9,          # webrtcvad is binary; Silero gives real probs
                    duration_ms=duration_ms,
                    ts_ms=onset_ts,
                )
                loop.call_soon_threadsafe(out_q.put_nowait, payload)
                print(f"[shim] ▶ SpeechSegment  {n_samples} samples  {duration_ms:.0f} ms")

        # Drain any in-progress utterance on clean shutdown
        audio, onset_ts = vad.drain()
        if audio is not None:
            n_samples   = len(audio) // 2
            duration_ms = n_samples / MIC_SAMPLE_RATE * 1000
            payload = _encode_speech_segment(audio, 0.9, duration_ms, onset_ts)
            loop.call_soon_threadsafe(out_q.put_nowait, payload)

    finally:
        stream.stop()
        stream.close()
        print("[shim] Microphone closed")


# ── Playback thread ───────────────────────────────────────────────────────────

def _playback_thread(play_q: queue.Queue) -> None:
    """
    Runs as a daemon thread.  Plays float32 PCM chunks from `play_q`
    sequentially via sounddevice.  sd.stop() (called from the async inbound
    handler on CancelPlayback) unblocks sd.wait() immediately.
    """
    while True:
        item = play_q.get()
        if item is None:          # shutdown sentinel
            break
        pcm, sr = item
        sd.play(pcm, samplerate=sr)
        sd.wait()                 # blocks until playback finishes or sd.stop() is called


# ── Async socket handlers ─────────────────────────────────────────────────────

async def _handle_outbound(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    out_q: asyncio.Queue,
) -> None:
    """
    Outbound connection handler.  Reads serialized speech events from `out_q`
    (written by the mic thread) and forwards them to the AudioIngressClient.
    """
    peer = writer.get_extra_info("peername") or "AudioIngressClient"
    print(f"[shim] {peer} connected (outbound)")
    try:
        while True:
            payload = await out_q.get()
            writer.write(_frame(payload))
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        print(f"[shim] {peer} disconnected (outbound)")
        try:
            writer.close()
        except Exception:
            pass


async def _handle_inbound(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    play_q: queue.Queue,
) -> None:
    """
    Inbound connection handler.  Reads AudioChunk / CancelPlayback messages
    from the AudioEgressClient and either queues PCM for playback or cancels
    the current audio output.
    """
    peer = writer.get_extra_info("peername") or "AudioEgressClient"
    print(f"[shim] {peer} connected (inbound)")
    try:
        while True:
            try:
                len_buf = await reader.readexactly(4)
            except asyncio.IncompleteReadError:
                break
            length, = struct.unpack("<I", len_buf)
            payload  = await reader.readexactly(length)

            tag, = struct.unpack_from("<I", payload, 0)
            rest = payload[4:]

            if tag == _TAG_AUDIO_CHUNK:
                pcm, sr, is_final = _decode_audio_chunk(rest)
                if not is_final and len(pcm) > 0:
                    play_q.put((pcm, sr))

            elif tag == _TAG_CANCEL:
                print("[shim] ◼ CancelPlayback — stopping audio")
                sd.stop()
                # Drain any queued but unplayed chunks for this utterance
                while not play_q.empty():
                    try:
                        play_q.get_nowait()
                    except queue.Empty:
                        break

            elif tag == _TAG_PAUSE:
                # sounddevice has no pause API for sd.play(); no-op in shim.
                # The real Rust binary uses an atomic flag inside the cpal callback.
                print("[shim] PausePlayback received — not implemented in shim")

            elif tag == _TAG_RESUME:
                print("[shim] ResumePlayback received — not implemented in shim")

    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        print(f"[shim] {peer} disconnected (inbound)")
        try:
            writer.close()
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(outbound_sock: str, inbound_sock: str) -> None:
    loop = asyncio.get_running_loop()

    # Queues that cross the thread / async boundary
    out_q: asyncio.Queue[bytes] = asyncio.Queue()   # mic thread → outbound socket
    play_q: queue.Queue = queue.Queue()              # inbound socket → playback thread

    # Start daemon threads before binding sockets so audio is ready immediately
    stop_event = threading.Event()

    threading.Thread(
        target=_mic_thread,
        args=(loop, out_q, stop_event),
        daemon=True,
        name="shim-mic",
    ).start()

    threading.Thread(
        target=_playback_thread,
        args=(play_q,),
        daemon=True,
        name="shim-playback",
    ).start()

    # Remove stale socket files from a previous run
    for path in (outbound_sock, inbound_sock):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    # Bind both servers — closures capture out_q / play_q
    out_server = await asyncio.start_unix_server(
        lambda r, w: _handle_outbound(r, w, out_q),
        outbound_sock,
    )
    in_server = await asyncio.start_unix_server(
        lambda r, w: _handle_inbound(r, w, play_q),
        inbound_sock,
    )

    print(f"[shim] Outbound socket : {outbound_sock}")
    print(f"[shim] Inbound socket  : {inbound_sock}")
    print("[shim] Waiting for orchestrator to connect…  (Ctrl-C to stop)")

    try:
        async with out_server, in_server:
            await asyncio.gather(
                out_server.serve_forever(),
                in_server.serve_forever(),
            )
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        play_q.put(None)   # unblock _playback_thread so it can exit cleanly


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="memvox audio shim — dev/demo stand-in for memvox-audio binary"
    )
    parser.add_argument(
        "--out-sock",
        default=OUTBOUND_SOCK_DEFAULT,
        help=f"Outbound Unix socket path (default: {OUTBOUND_SOCK_DEFAULT})",
    )
    parser.add_argument(
        "--in-sock",
        default=INBOUND_SOCK_DEFAULT,
        help=f"Inbound Unix socket path (default: {INBOUND_SOCK_DEFAULT})",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.out_sock, args.in_sock))
    except KeyboardInterrupt:
        print("\n[shim] Stopped.")
