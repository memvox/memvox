"""Tests for the web UI WebSocket bridge (observability/uibridge.py)."""

import asyncio
import json

import pytest
import websockets

from memvox.observability.uibridge import UIBridge


async def _recv_json(ws, timeout: float = 2.0) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), timeout))


class TestUIBridge:
    @pytest.mark.asyncio
    async def test_emit_reaches_connected_client(self):
        bridge = UIBridge(port=0)
        await bridge.start()
        assert bridge.running
        try:
            async with websockets.connect(f"ws://127.0.0.1:{bridge.port}") as ws:
                bridge.emit({"type": "user_final", "turn_id": "t1",
                             "text": "안녕하세요", "language": "ko"})
                ev = await _recv_json(ws)
                assert ev["type"] == "user_final"
                assert ev["text"] == "안녕하세요"   # ensure_ascii=False survives
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_replay_buffer_for_late_joiners(self):
        bridge = UIBridge(port=0)
        await bridge.start()
        try:
            bridge.emit({"type": "hello", "session_id": "s1"})
            bridge.emit({"type": "user_final", "turn_id": "t1",
                         "text": "hi", "language": "en"})
            async with websockets.connect(f"ws://127.0.0.1:{bridge.port}") as ws:
                first = await _recv_json(ws)
                second = await _recv_json(ws)
                assert first["type"] == "hello"
                assert second["turn_id"] == "t1"
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_emit_without_start_is_safe(self):
        bridge = UIBridge(port=0)
        assert not bridge.running
        bridge.emit({"type": "hello", "session_id": "s1"})  # must not raise

    @pytest.mark.asyncio
    async def test_bind_conflict_disables_bridge(self):
        first = UIBridge(port=0)
        await first.start()
        try:
            second = UIBridge(port=first.port)
            await second.start()  # port taken → disabled, no raise
            assert not second.running
        finally:
            await first.stop()

    @pytest.mark.asyncio
    async def test_multiple_clients_all_receive(self):
        bridge = UIBridge(port=0)
        await bridge.start()
        try:
            url = f"ws://127.0.0.1:{bridge.port}"
            async with websockets.connect(url) as a, websockets.connect(url) as b:
                bridge.emit({"type": "session_end", "session_id": "s1"})
                assert (await _recv_json(a))["type"] == "session_end"
                assert (await _recv_json(b))["type"] == "session_end"
        finally:
            await bridge.stop()
