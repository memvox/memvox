"""
Integration test: real Unix sockets + real bincode wire format.

This wires the orchestrator's real AudioIngressClient / AudioEgressClient to a
fake 'audio binary' peer that speaks the same length-prefixed bincode protocol
the Rust binary (and shim.py) uses. ASR / LLM / TTS / wiki are still mocked,
but every byte that crosses the orchestrator <-> binary boundary is real.

If this test passes, the wire format is sound and the only remaining risks for
the e2e demo are hardware-level (mic/speaker config) and external services
(vLLM, whisper).
"""

import asyncio
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from memvox.session.orchestrator import SessionOrchestrator
from memvox.session.types import SessionConfig
from memvox.voice.egress import AudioEgressClient
from memvox.voice.ingress import AudioIngressClient
from memvox.voice.types import AudioChunk, SpeechSegment, TokenChunk, Transcript


# ── Wire-format helpers (mirror shim.py) ──────────────────────────────────────

def _frame(payload: bytes) -> bytes:
    return struct.pack("<I", len(payload)) + payload


def _encode_speech_segment(audio_i16: bytes, ts_ms: int = 1000) -> bytes:
    n = len(audio_i16) // 2
    return (
        struct.pack("<IQ", 1, n)            # tag=1 (SpeechSegment), n samples
        + audio_i16
        + struct.pack("<ff", 0.95, 500.0)   # speech_prob, duration_ms
        + struct.pack("<Q", ts_ms)
    )


def _decode_audio_chunk_payload(payload: bytes) -> tuple[np.ndarray, int, bool]:
    """Decode an InboundMsg::AudioChunk payload (tag already consumed)."""
    n, = struct.unpack_from("<Q", payload, 0)
    pcm = np.frombuffer(payload[8 : 8 + n * 4], dtype=np.float32).copy()
    sr, is_final = struct.unpack_from("<I?", payload, 8 + n * 4)
    return pcm, sr, is_final


# ── Fake binary peer ──────────────────────────────────────────────────────────

class FakeAudioBinary:
    """Stand-in for shim.py / memvox-audio. Listens on two Unix sockets and
    speaks the same wire format."""

    def __init__(self, out_sock: str, in_sock: str) -> None:
        self.out_sock = out_sock
        self.in_sock  = in_sock
        self.received_chunks: list[tuple[np.ndarray, int, bool]] = []
        self._out_writer: asyncio.StreamWriter | None = None
        self._out_server: asyncio.AbstractServer | None = None
        self._in_server:  asyncio.AbstractServer | None = None
        self._client_connected = asyncio.Event()

    async def start(self) -> None:
        self._out_server = await asyncio.start_unix_server(
            self._handle_outbound, self.out_sock
        )
        self._in_server = await asyncio.start_unix_server(
            self._handle_inbound, self.in_sock
        )

    async def stop(self) -> None:
        if self._out_server:
            self._out_server.close()
            await self._out_server.wait_closed()
        if self._in_server:
            self._in_server.close()
            await self._in_server.wait_closed()

    async def _handle_outbound(self, reader, writer):
        self._out_writer = writer
        self._client_connected.set()
        # Hold the connection open; the test will write through self._out_writer.
        try:
            await reader.read()  # blocks until disconnect
        finally:
            writer.close()

    async def _handle_inbound(self, reader, writer):
        try:
            while True:
                len_buf = await reader.readexactly(4)
                length, = struct.unpack("<I", len_buf)
                payload = await reader.readexactly(length)
                tag, = struct.unpack_from("<I", payload, 0)
                if tag == 0:  # AudioChunk
                    chunk = _decode_audio_chunk_payload(payload[4:])
                    self.received_chunks.append(chunk)
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()

    async def send_speech_segment(self, audio_i16: bytes) -> None:
        """Push a SpeechSegment to the orchestrator (simulating mic VAD onset)."""
        await self._client_connected.wait()
        assert self._out_writer is not None
        self._out_writer.write(_frame(_encode_speech_segment(audio_i16)))
        await self._out_writer.drain()


# ── Orchestrator builder ──────────────────────────────────────────────────────

def _build_orchestrator(out_sock: str, in_sock: str):
    """Wire the real AudioIngressClient / AudioEgressClient into the orchestrator
    with all inference engines mocked."""

    config = SessionConfig(
        system_prompt="test",
        language="ko",
        voice="Ana Florence",
    )

    # Real socket clients — these are what we want to exercise
    ingress = AudioIngressClient(out_sock)
    egress  = AudioEgressClient(in_sock)

    # Mocked engines
    asr  = MagicMock()
    asr.initialize = AsyncMock()
    asr.transcribe = AsyncMock(return_value=Transcript(
        text="안녕하세요",
        language="ko",
        confidence=0.95,
        no_speech_prob=0.05,
        latency_ms=42.0,
        source_segment=SpeechSegment(audio=b"", speech_prob=0.95, duration_ms=500.0, timestamp_start=0.0),
    ))

    async def _llm_gen(_):
        yield TokenChunk(text="안녕!", is_thinking=False, is_final=False, turn_id="t1")
        yield TokenChunk(text="",    is_thinking=False, is_final=True,  turn_id="t1")

    llm = MagicMock()
    llm.generate = MagicMock(side_effect=_llm_gen)

    async def _tts_gen(tokens):
        # consume tokens to populate orchestrator's content_parts
        async for _ in tokens:
            pass
        # Emit two real AudioChunks then a final sentinel
        yield AudioChunk(
            pcm_bytes=np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes(),
            sample_rate=24_000,
            is_final=False,
            sentence_text="안녕!",
            chunk_latency_ms=10.0,
        )
        yield AudioChunk(
            pcm_bytes=np.array([0.4, 0.5], dtype=np.float32).tobytes(),
            sample_rate=24_000,
            is_final=False,
            sentence_text="안녕!",
            chunk_latency_ms=12.0,
        )
        yield AudioChunk(
            pcm_bytes=b"",
            sample_rate=24_000,
            is_final=True,
            sentence_text="",
            chunk_latency_ms=0.0,
        )

    tts = MagicMock()
    tts.initialize = AsyncMock()
    tts.synthesize = MagicMock(side_effect=_tts_gen)

    wiki = MagicMock()
    wiki.search = AsyncMock(return_value=[])

    return SessionOrchestrator(config, asr, llm, tts, wiki, ingress, egress)


# ── The actual test ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_socket_roundtrip():
    """One full turn end-to-end: SpeechSegment in over a real socket, two
    AudioChunks out over a real socket. Validates wire format + framing in both
    directions against the actual orchestrator code path."""

    with tempfile.TemporaryDirectory() as td:
        out_sock = os.path.join(td, "out.sock")
        in_sock  = os.path.join(td, "in.sock")

        # Start the fake binary first so the orchestrator can connect on start()
        binary = FakeAudioBinary(out_sock, in_sock)
        await binary.start()

        orch = _build_orchestrator(out_sock, in_sock)

        try:
            await orch.start()

            # Send a SpeechSegment payload — the orchestrator's _run loop should
            # wake, transcribe, generate, synthesize, and write AudioChunks back.
            fake_pcm = np.zeros(1600, dtype=np.int16).tobytes()
            await binary.send_speech_segment(fake_pcm)

            # Wait for the orchestrator to push two non-final chunks back.
            for _ in range(50):  # up to ~500 ms
                if len(binary.received_chunks) >= 2:
                    break
                await asyncio.sleep(0.01)

        finally:
            await orch.stop()
            await binary.stop()

        # ── Assertions ────────────────────────────────────────────────────────
        assert len(binary.received_chunks) == 2, (
            f"Expected 2 AudioChunks over the wire, got {len(binary.received_chunks)}"
        )

        chunk1_pcm, chunk1_sr, chunk1_final = binary.received_chunks[0]
        np.testing.assert_array_almost_equal(chunk1_pcm, [0.1, 0.2, 0.3])
        assert chunk1_sr == 24_000
        assert chunk1_final is False

        chunk2_pcm, chunk2_sr, chunk2_final = binary.received_chunks[1]
        np.testing.assert_array_almost_equal(chunk2_pcm, [0.4, 0.5])
        assert chunk2_sr == 24_000
        assert chunk2_final is False

        # History should reflect the turn
        assert len(orch._history) == 2
        assert orch._history[0].role == "user"
        assert orch._history[0].content == "안녕하세요"
        assert orch._history[1].role == "assistant"
        assert orch._history[1].content == "안녕!"
