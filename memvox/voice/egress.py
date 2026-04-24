"""Asyncio Unix socket client for memvox-audio's inbound socket.

Sends length-prefixed bincode messages to the Rust binary:
  InboundMsg::AudioChunk(AudioChunk { pcm: Vec<f32>, sample_rate: u32, is_final: bool })
  InboundMsg::CancelPlayback

Wire format matches ipc.rs exactly:
  [u32 LE payload length] [bincode payload]

Bincode v1 encoding (default): little-endian fixed-size integers, u64 Vec lengths.
"""

import asyncio
import struct

import numpy as np

from memvox.voice.types import AudioChunk, CancelPlayback, PausePlayback, ResumePlayback

# InboundMsg enum tags (match Rust definition order)
_TAG_AUDIO_CHUNK    = 0
_TAG_CANCEL         = 1
_TAG_PAUSE          = 2
_TAG_RESUME         = 3

_InboundMsg = AudioChunk | CancelPlayback | PausePlayback | ResumePlayback


def _encode_inbound(msg: _InboundMsg) -> bytes:
    if isinstance(msg, AudioChunk):
        pcm = np.frombuffer(msg.pcm_bytes, dtype=np.float32)
        n = len(pcm)
        # tag(u32) + len(u64) + pcm(f32*n) + sample_rate(u32) + is_final(bool)
        header = struct.pack("<IQ", _TAG_AUDIO_CHUNK, n)
        body   = pcm.tobytes()
        footer = struct.pack("<I?", msg.sample_rate, msg.is_final)
        return header + body + footer

    if isinstance(msg, CancelPlayback):
        return struct.pack("<I", _TAG_CANCEL)

    if isinstance(msg, PausePlayback):
        return struct.pack("<I", _TAG_PAUSE)

    if isinstance(msg, ResumePlayback):
        return struct.pack("<I", _TAG_RESUME)

    raise TypeError(f"Unsupported inbound message type: {type(msg)}")


def _frame(payload: bytes) -> bytes:
    """Prepend 4-byte LE length prefix."""
    return struct.pack("<I", len(payload)) + payload


class AudioEgressClient:
    """Writes AudioChunk and control messages to memvox-audio."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        _, self._writer = await asyncio.open_unix_connection(self._socket_path)

    async def send(self, msg: _InboundMsg) -> None:
        """Encode and send one message. Thread-safe via asyncio lock."""
        if self._writer is None:
            raise RuntimeError("call connect() before send()")
        payload = _frame(_encode_inbound(msg))
        async with self._lock:
            self._writer.write(payload)
            await self._writer.drain()

    async def close(self) -> None:
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
