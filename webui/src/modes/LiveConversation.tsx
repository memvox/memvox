import { useEffect, useRef } from "react";
import { useLiveSession } from "../lib/useLiveSession";

export function LiveConversation() {
  const { turns, connected, wsUrl } = useLiveSession();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  return (
    <section>
      <h2 className="mode-title">Live Conversation</h2>
      <p className="mode-blurb">
        Words spoken between you and the voice agent, as they happen.
      </p>

      <div className={`live-status${connected ? " connected" : ""}`}>
        <span className="dot" />
        {connected ? "Connected to memvox" : `Waiting for memvox at ${wsUrl}…`}
      </div>

      {turns.length === 0 ? (
        <div className="live-empty">
          <p>
            No conversation yet. Start a voice session with{" "}
            <code>./run.sh up</code> and this page will display the transcript
            live — your words on the right, the agent's on the left.
          </p>
        </div>
      ) : (
        <div className="transcript">
          {turns.map((turn) => (
            <div
              key={turn.id}
              className={`turn ${turn.role}${turn.pending ? " pending" : ""}`}
            >
              <span className="speaker">
                {turn.role === "user" ? "You" : "memvox"}
                {turn.language ? ` · ${turn.language}` : ""}
              </span>
              <div className="bubble" lang="ko">
                {turn.text}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      )}
    </section>
  );
}
