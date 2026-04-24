import struct
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from memvox.voice.egress import AudioEgressClient, _encode_inbound, _frame
from memvox.voice.types import AudioChunk, CancelPlayback, PausePlayback, ResumePlayback


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(
    pcm: np.ndarray | None = None,
    sample_rate: int = 24_000,
    is_final: bool = False,
) -> AudioChunk:
    if pcm is None:
        pcm = np.array([0.5, -0.5, 0.1], dtype=np.float32)
    return AudioChunk(
        pcm_bytes=pcm.tobytes(),
        sample_rate=sample_rate,
        is_final=is_final,
        sentence_text="test",
        chunk_latency_ms=5.0,
    )


def _n_from_encoded(data: bytes) -> int:
    """Extract the Vec<f32> length from an AudioChunk encoding."""
    n, = struct.unpack_from("<Q", data, 4)   # tag(4) + n(8)
    return n


# ── _frame ────────────────────────────────────────────────────────────────────

class TestFrame:
    def test_prepends_u32_le_length(self):
        payload = b"hello"
        assert _frame(payload)[:4] == struct.pack("<I", 5)

    def test_payload_follows_length(self):
        payload = b"world"
        assert _frame(payload)[4:] == payload

    def test_empty_payload(self):
        assert _frame(b"") == struct.pack("<I", 0)


# ── _encode_inbound ───────────────────────────────────────────────────────────

class TestEncodeInbound:
    def test_cancel_is_tag_1(self):
        assert _encode_inbound(CancelPlayback()) == struct.pack("<I", 1)

    def test_pause_is_tag_2(self):
        assert _encode_inbound(PausePlayback()) == struct.pack("<I", 2)

    def test_resume_is_tag_3(self):
        assert _encode_inbound(ResumePlayback()) == struct.pack("<I", 3)

    def test_audio_chunk_tag_is_0(self):
        tag, = struct.unpack_from("<I", _encode_inbound(_chunk()), 0)
        assert tag == 0

    def test_audio_chunk_vec_length(self):
        pcm = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assert _n_from_encoded(_encode_inbound(_chunk(pcm=pcm))) == 3

    def test_audio_chunk_pcm_bytes(self):
        pcm = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        data = _encode_inbound(_chunk(pcm=pcm))
        # tag(4) + n(8) = 12 bytes; pcm is next 3*4 bytes
        recovered = np.frombuffer(data[12:24], dtype=np.float32)
        np.testing.assert_array_almost_equal(recovered, pcm)

    def test_audio_chunk_sample_rate(self):
        pcm = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        data = _encode_inbound(_chunk(pcm=pcm, sample_rate=22_050))
        n = _n_from_encoded(data)
        offset = 4 + 8 + n * 4
        sr, = struct.unpack_from("<I", data, offset)
        assert sr == 22_050

    def test_audio_chunk_is_final_true(self):
        pcm = np.array([0.5], dtype=np.float32)
        data = _encode_inbound(_chunk(pcm=pcm, is_final=True))
        n = _n_from_encoded(data)
        offset = 4 + 8 + n * 4 + 4
        is_final, = struct.unpack_from("<?", data, offset)
        assert is_final is True

    def test_audio_chunk_is_final_false(self):
        pcm = np.array([0.5], dtype=np.float32)
        data = _encode_inbound(_chunk(pcm=pcm, is_final=False))
        n = _n_from_encoded(data)
        offset = 4 + 8 + n * 4 + 4
        is_final, = struct.unpack_from("<?", data, offset)
        assert is_final is False

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            _encode_inbound("not a message")  # type: ignore[arg-type]

    def test_framed_length_prefix_matches_payload(self):
        pcm = np.array([0.1, 0.2], dtype=np.float32)
        payload = _encode_inbound(_chunk(pcm=pcm))
        framed = _frame(payload)
        length, = struct.unpack_from("<I", framed, 0)
        assert length == len(payload)
        assert len(framed) == 4 + length


# ── AudioEgressClient ─────────────────────────────────────────────────────────

class TestAudioEgressClient:
    def _client_with_writer(self) -> tuple[AudioEgressClient, MagicMock]:
        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        client = AudioEgressClient("/tmp/fake.sock")
        client._writer = writer
        return client, writer

    async def test_send_without_connect_raises(self):
        client = AudioEgressClient("/tmp/fake.sock")
        with pytest.raises(RuntimeError, match="connect"):
            await client.send(CancelPlayback())

    async def test_send_writes_once(self):
        client, writer = self._client_with_writer()
        await client.send(CancelPlayback())
        writer.write.assert_called_once()

    async def test_send_drains_after_write(self):
        client, writer = self._client_with_writer()
        await client.send(CancelPlayback())
        writer.drain.assert_called_once()

    async def test_send_audio_chunk_produces_correct_tag(self):
        client, writer = self._client_with_writer()
        await client.send(_chunk())
        written: bytes = writer.write.call_args[0][0]
        # First 4 bytes = framing length; next 4 bytes = enum tag
        tag, = struct.unpack_from("<I", written, 4)
        assert tag == 0

    async def test_send_cancel_produces_correct_tag(self):
        client, writer = self._client_with_writer()
        await client.send(CancelPlayback())
        written: bytes = writer.write.call_args[0][0]
        tag, = struct.unpack_from("<I", written, 4)
        assert tag == 1
