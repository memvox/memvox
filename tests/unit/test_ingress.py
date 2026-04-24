import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from memvox.voice.ingress import AudioIngressClient, _decode_outbound
from memvox.voice.types import SpeechSegment, SpeechStarted


# ── Wire-format helpers ───────────────────────────────────────────────────────

def _speech_started_bytes(timestamp_ms: int = 2000) -> bytes:
    return struct.pack("<IQ", 0, timestamp_ms)


def _speech_segment_bytes(
    samples: list[int] | None = None,
    speech_prob: float = 0.9,
    duration_ms: float = 400.0,
    timestamp_ms: int = 3000,
) -> bytes:
    if samples is None:
        samples = [1000, -500, 2000]
    n = len(samples)
    header = struct.pack("<IQ", 1, n)
    audio  = struct.pack(f"<{n}h", *samples) if n else b""
    floats = struct.pack("<ff", speech_prob, duration_ms)
    ts     = struct.pack("<Q", timestamp_ms)
    return header + audio + floats + ts


def _framed(payload: bytes) -> bytes:
    return struct.pack("<I", len(payload)) + payload


# ── _decode_outbound ──────────────────────────────────────────────────────────

class TestDecodeOutbound:
    def test_speech_started_type(self):
        msg = _decode_outbound(_speech_started_bytes())
        assert isinstance(msg, SpeechStarted)

    def test_speech_started_timestamp_converted_to_seconds(self):
        msg = _decode_outbound(_speech_started_bytes(timestamp_ms=5000))
        assert msg.timestamp == pytest.approx(5.0)

    def test_speech_segment_type(self):
        msg = _decode_outbound(_speech_segment_bytes())
        assert isinstance(msg, SpeechSegment)

    def test_speech_segment_audio_bytes_match(self):
        samples = [100, -200, 300]
        msg = _decode_outbound(_speech_segment_bytes(samples=samples))
        decoded = list(struct.unpack(f"<{len(samples)}h", msg.audio))
        assert decoded == samples

    def test_speech_segment_speech_prob(self):
        msg = _decode_outbound(_speech_segment_bytes(speech_prob=0.87))
        assert msg.speech_prob == pytest.approx(0.87, abs=1e-5)

    def test_speech_segment_duration_ms(self):
        msg = _decode_outbound(_speech_segment_bytes(duration_ms=250.0))
        assert msg.duration_ms == pytest.approx(250.0, abs=1e-5)

    def test_speech_segment_timestamp_converted_to_seconds(self):
        msg = _decode_outbound(_speech_segment_bytes(timestamp_ms=4500))
        assert msg.timestamp_start == pytest.approx(4.5)

    def test_speech_segment_empty_audio(self):
        msg = _decode_outbound(_speech_segment_bytes(samples=[]))
        assert msg.audio == b""

    def test_unknown_tag_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            _decode_outbound(struct.pack("<I", 99))


# ── AudioIngressClient ────────────────────────────────────────────────────────

class TestAudioIngressClient:
    def _client_with_reader(self, *payloads: bytes) -> AudioIngressClient:
        side_effects = []
        for p in payloads:
            side_effects.append(struct.pack("<I", len(p)))
            side_effects.append(p)
        reader = MagicMock()
        reader.readexactly = AsyncMock(side_effect=side_effects)
        client = AudioIngressClient("/tmp/fake.sock")
        client._reader = reader
        return client

    async def test_recv_decodes_speech_started(self):
        payload = _speech_started_bytes(timestamp_ms=1000)
        client = self._client_with_reader(payload)
        msg = await client.recv()
        assert isinstance(msg, SpeechStarted)
        assert msg.timestamp == pytest.approx(1.0)

    async def test_recv_decodes_speech_segment(self):
        payload = _speech_segment_bytes(samples=[10, -20, 30])
        client = self._client_with_reader(payload)
        msg = await client.recv()
        assert isinstance(msg, SpeechSegment)
        assert list(struct.unpack("<3h", msg.audio)) == [10, -20, 30]

    async def test_recv_returns_none_on_eof(self):
        reader = MagicMock()
        reader.readexactly = AsyncMock(
            side_effect=asyncio.IncompleteReadError(b"", 4)
        )
        client = AudioIngressClient("/tmp/fake.sock")
        client._reader = reader
        assert await client.recv() is None

    async def test_recv_returns_none_on_connection_reset(self):
        reader = MagicMock()
        reader.readexactly = AsyncMock(side_effect=ConnectionResetError)
        client = AudioIngressClient("/tmp/fake.sock")
        client._reader = reader
        assert await client.recv() is None

    async def test_recv_without_connect_raises(self):
        client = AudioIngressClient("/tmp/fake.sock")
        with pytest.raises(RuntimeError, match="connect"):
            await client.recv()

    async def test_successive_recv_calls(self):
        p1 = _speech_started_bytes(timestamp_ms=100)
        p2 = _speech_segment_bytes(samples=[5, -5])
        client = self._client_with_reader(p1, p2)
        first  = await client.recv()
        second = await client.recv()
        assert isinstance(first,  SpeechStarted)
        assert isinstance(second, SpeechSegment)
