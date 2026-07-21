import { useState } from "react";
import { LiveConversation } from "./modes/LiveConversation";
import { Flashcards } from "./modes/Flashcards";
import { Alphabet } from "./modes/Alphabet";
import { useTheme } from "./lib/useTheme";

const MODES = [
  { id: "live", label: "Live", render: () => <LiveConversation /> },
  { id: "flashcards", label: "Flashcards", render: () => <Flashcards /> },
  { id: "alphabet", label: "Alphabet", render: () => <Alphabet /> },
] as const;

type ModeId = (typeof MODES)[number]["id"];

export default function App() {
  const [mode, setMode] = useState<ModeId>("live");
  const [theme, toggleTheme] = useTheme();
  const active = MODES.find((m) => m.id === mode) ?? MODES[0];

  return (
    <div className="center-column">
      <header className="header">
        <h1 className="header-title">
          mem<span className="vox">vox</span>
          <span className="subtitle" lang="ko">
            한국어 practice
          </span>
        </h1>
        <button
          type="button"
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
          title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
        >
          {theme === "light" ? (
            /* moon */
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12.3 2a9.9 9.9 0 0 0-2 .2 8.1 8.1 0 0 1 3.6 6.7 8.1 8.1 0 0 1-8.1 8.1 8 8 0 0 1-3.6-.8A10 10 0 1 0 12.3 2z" />
            </svg>
          ) : (
            /* sun */
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zm0-6h.01L13 4h-2l.99-3zM12 23h.01L13 20h-2l.99 3zM4.2 5.6 6.5 7.9 7.9 6.5 5.6 4.2 4.2 5.6zm12.1 12.1 2.1 2.1 1.4-1.4-2.1-2.1-1.4 1.4zM1 13h3v-2H1v2zm19 0h3v-2h-3v2zM4.2 18.4l1.4 1.4 2.1-2.1-1.4-1.4-2.1 2.1zM18.4 4.2l-2.1 2.1 1.4 1.4 2.1-2.1-1.4-1.4z" />
            </svg>
          )}
        </button>
      </header>
      <hr className="header-separator" />

      <nav className="side-nav" aria-label="Practice modes">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            className={m.id === mode ? "active" : ""}
            onClick={() => setMode(m.id)}
          >
            {m.label}
          </button>
        ))}
        <span className="nav-note">
          Korean tutor
          <br />
          voice agent
        </span>
      </nav>

      <main className="main-content">{active.render()}</main>

      <footer>memvox</footer>
    </div>
  );
}
