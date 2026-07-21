"""WebSocket broadcast bridge for the web UI (webui/).

Pushes live transcript events (what the human said, what the agent replied)
to any connected browser so the web UI can render the conversation as it
happens. Fire-and-forget: the voice pipeline never blocks on a browser, and
a missing `websockets` package or an unbound port just disables the bridge
with a log line — the session itself is unaffected.

Event shapes are mirrored in webui/src/lib/types.ts (BridgeEvent). Keep the
two in sync when adding event types.

A short replay buffer is kept so a browser opened mid-session still sees the
conversation so far.
"""

import asyncio
import json
from collections import deque

_REPLAY_MAX = 200


class UIBridge:
    """Broadcast JSON events to WebSocket clients on localhost."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._server = None
        self._clients: set = set()
        self._replay: deque[str] = deque(maxlen=_REPLAY_MAX)

    @property
    def running(self) -> bool:
        return self._server is not None

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        """Bind the server. Failure disables the bridge, never the session."""
        try:
            import websockets
        except ImportError:
            print("[ui] `websockets` not installed — web UI bridge disabled.")
            return
        try:
            self._server = await websockets.serve(
                self._handle_client, self._host, self._port
            )
        except OSError as e:
            print(f"[ui] could not bind ws://{self._host}:{self._port} ({e}) — "
                  "web UI bridge disabled.")
            self._server = None
            return
        # Resolve the real port (supports port=0 in tests).
        for sock in self._server.sockets:
            self._port = sock.getsockname()[1]
            break
        print(f"[ui] web UI bridge listening on ws://{self._host}:{self._port}")

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        self._clients.clear()

    def emit(self, event: dict) -> None:
        """Queue an event for broadcast. Safe to call when not running."""
        payload = json.dumps(event, ensure_ascii=False)
        self._replay.append(payload)
        if self._server is None or not self._clients:
            return
        for client in list(self._clients):
            # One task per client; a slow/dead browser can't stall the pipeline.
            asyncio.get_running_loop().create_task(self._send(client, payload))

    async def _send(self, client, payload: str) -> None:
        try:
            await client.send(payload)
        except Exception:
            self._clients.discard(client)

    async def _handle_client(self, websocket) -> None:
        self._clients.add(websocket)
        try:
            for payload in list(self._replay):
                await websocket.send(payload)
            # Hold the connection open; we never expect inbound messages.
            async for _ in websocket:
                pass
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
