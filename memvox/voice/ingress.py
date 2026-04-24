"""Asyncio Unix socket client for memvox-audio's outbound socket.

The Rust binary sends length-prefixed bincode messages:
  OutboundMsg::SpeechStarted(SpeechStarted { timestamp_ms: u64 })
  OutboundMsg::SpeechSegment(SpeechSegment { audio: Vec<i16>, speech_prob: f32,
                                              duration_ms: f32, timestamp_start_ms: u64 })

Wire format matches ipc.rs exactly:
  [u32 LE payload length] [bincode payload]

Bincode v1 encoding (default): little-endian fixed-size integers, u64 Vec lengths.
"""

import asyncio
import struct

from memvox.voice.types import SpeechSegment, SpeechStarted

# OutboundMsg enum tags (match Rust definition order)
_TAG_SPEECH_STARTED = 0
_TAG_SPEECH_SEGMENT = 1


def _decode_outbound(payload: bytes) -> SpeechStarted | SpeechSegment:
    tag, = struct.unpack_from("<I", payload, 0)  # u32 enum tag

    if tag == _TAG_SPEECH_STARTED:
        ts_ms, = struct.unpack_from("<Q", payload, 4)  # u64
        return SpeechStarted(timestamp=ts_ms / 1000.0)

    if tag == _TAG_SPEECH_SEGMENT:
        off = 4
        n_samples, = struct.unpack_from("<Q", payload, off)   # u64 Vec len
        off += 8
        raw_audio = payload[off : off + n_samples * 2]        # i16 * n
        off += n_samples * 2
        speech_prob, duration_ms = struct.unpack_from("<ff", payload, off)  # f32 f32
        off += 8
        ts_ms, = struct.unpack_from("<Q", payload, off)       # u64
        return SpeechSegment(
            audio=raw_audio,
            speech_prob=speech_prob,
            duration_ms=duration_ms,
            timestamp_start=ts_ms / 1000.0,
        )

    raise ValueError(f"Unknown OutboundMsg tag: {tag}")


class AudioIngressClient:
    """Reads SpeechStarted and SpeechSegment messages from memvox-audio."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_unix_connection(
            self._socket_path
        )

    async def recv(self) -> SpeechStarted | SpeechSegment | None:
        """Return the next message, or None if the connection is closed."""
        if self._reader is None:
            raise RuntimeError("call connect() before recv()")
        try:
            len_buf = await self._reader.readexactly(4)
            length = struct.unpack("<I", len_buf)[0]          # u32 LE
            payload = await self._reader.readexactly(length)
            return _decode_outbound(payload)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return None

    async def close(self) -> None:
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
