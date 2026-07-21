import { useEffect, useRef, useState } from "react";
import type { BridgeEvent, Turn } from "./types";

const WS_URL =
  (import.meta.env.VITE_MEMVOX_WS_URL as string | undefined) ??
  "ws://localhost:8765";

const RECONNECT_MS = 2500;

/** Fold one bridge event into the running transcript. */
function applyEvent(turns: Turn[], ev: BridgeEvent): Turn[] {
  switch (ev.type) {
    case "user_final":
      return [
        ...turns,
        { id: `${ev.turn_id}-u`, role: "user", text: ev.text, language: ev.language },
      ];
    case "assistant_sentence": {
      const id = `${ev.turn_id}-a`;
      const existing = turns.find((t) => t.id === id);
      if (existing) {
        return turns.map((t) =>
          t.id === id ? { ...t, text: `${t.text} ${ev.text}`.trim() } : t,
        );
      }
      return [...turns, { id, role: "agent", text: ev.text, pending: true }];
    }
    case "assistant_final": {
      const id = `${ev.turn_id}-a`;
      if (!turns.some((t) => t.id === id)) {
        return [...turns, { id, role: "agent", text: ev.text }];
      }
      // Trust the final text — it's authoritative over accumulated sentences.
      return turns.map((t) =>
        t.id === id ? { ...t, text: ev.text, pending: false } : t,
      );
    }
    default:
      return turns;
  }
}

export interface LiveSession {
  turns: Turn[];
  connected: boolean;
  wsUrl: string;
}

/** Subscribe to the memvox UI bridge; auto-reconnects while mounted. */
export function useLiveSession(): LiveSession {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let disposed = false;
    let retryTimer: number | undefined;

    const connect = () => {
      if (disposed) return;
      const ws = new WebSocket(WS_URL);
      socketRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onmessage = (msg) => {
        try {
          const ev = JSON.parse(msg.data as string) as BridgeEvent;
          setTurns((prev) => applyEvent(prev, ev));
        } catch {
          // Ignore malformed frames; the bridge only sends JSON.
        }
      };
      ws.onclose = () => {
        setConnected(false);
        socketRef.current = null;
        if (!disposed) retryTimer = window.setTimeout(connect, RECONNECT_MS);
      };
      ws.onerror = () => ws.close();
    };

    connect();
    return () => {
      disposed = true;
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
      socketRef.current?.close();
    };
  }, []);

  return { turns, connected, wsUrl: WS_URL };
}
