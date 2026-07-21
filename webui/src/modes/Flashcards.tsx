import { useEffect, useMemo, useState } from "react";
import type { SubmitEvent } from "react";
import { SEED_DECK } from "../data/seedDeck";
import type { Flashcard } from "../lib/types";
import { SpeakButton } from "../components/SpeakButton";

const STORAGE_KEY = "memvox.flashcards.v1";

function loadDeck(): Flashcard[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Flashcard[];
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch {
    // Corrupt storage — fall through to the seed deck.
  }
  return SEED_DECK;
}

function shuffled<T>(items: T[]): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

export function Flashcards() {
  const [deck, setDeck] = useState<Flashcard[]>(loadDeck);
  const [order, setOrder] = useState<string[]>(() => deck.map((c) => c.id));
  const [index, setIndex] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [showBank, setShowBank] = useState(false);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(deck));
  }, [deck]);

  // Keep the practice order in sync when cards are added or removed.
  useEffect(() => {
    setOrder((prev) => {
      const ids = new Set(deck.map((c) => c.id));
      const kept = prev.filter((id) => ids.has(id));
      const added = deck.filter((c) => !prev.includes(c.id)).map((c) => c.id);
      return [...kept, ...added];
    });
  }, [deck]);

  const cardsById = useMemo(
    () => new Map(deck.map((c) => [c.id, c])),
    [deck],
  );
  const safeIndex = order.length === 0 ? 0 : Math.min(index, order.length - 1);
  const current = order.length > 0 ? cardsById.get(order[safeIndex]) : undefined;

  const go = (delta: number) => {
    if (order.length === 0) return;
    setIndex((safeIndex + delta + order.length) % order.length);
    setRevealed(false);
  };

  const reshuffle = () => {
    setOrder(shuffled(deck.map((c) => c.id)));
    setIndex(0);
    setRevealed(false);
  };

  const addCard = (e: SubmitEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const data = new FormData(form);
    const korean = String(data.get("korean") ?? "").trim();
    const english = String(data.get("english") ?? "").trim();
    const romanization = String(data.get("romanization") ?? "").trim();
    if (!korean || !english) return;
    setDeck((prev) => [
      ...prev,
      {
        id: `card-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        korean,
        english,
        romanization: romanization || undefined,
      },
    ]);
    form.reset();
    (form.elements.namedItem("korean") as HTMLInputElement | null)?.focus();
  };

  const removeCard = (id: string) => {
    setDeck((prev) => prev.filter((c) => c.id !== id));
  };

  return (
    <section>
      <h2 className="mode-title">Flashcards</h2>
      <p className="mode-blurb">
        Practice your vocabulary bank — tap a card to reveal, listen for native
        pronunciation.
      </p>

      <div className="flash-toolbar">
        <button type="button" className="pill-button" onClick={reshuffle}>
          Shuffle
        </button>
        <button
          type="button"
          className="pill-button subtle"
          onClick={() => setShowBank((s) => !s)}
        >
          {showBank ? "Hide card bank" : "Manage card bank"}
        </button>
        <span className="flash-count">
          {deck.length} card{deck.length === 1 ? "" : "s"}
        </span>
      </div>

      {current ? (
        <>
          <div
            className="flashcard"
            onClick={() => setRevealed((r) => !r)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === " " || e.key === "Enter") {
                e.preventDefault();
                setRevealed((r) => !r);
              }
              if (e.key === "ArrowRight") go(1);
              if (e.key === "ArrowLeft") go(-1);
            }}
          >
            <div className="big" lang="ko">
              {current.korean}
            </div>
            {revealed ? (
              <>
                <div className="reveal">{current.english}</div>
                {current.romanization && (
                  <div className="roman">{current.romanization}</div>
                )}
              </>
            ) : (
              <div className="hint">tap to reveal</div>
            )}
            <SpeakButton text={current.korean} />
          </div>

          <div className="flash-nav">
            <button type="button" className="pill-button subtle" onClick={() => go(-1)}>
              ← Prev
            </button>
            <span className="pos">
              {safeIndex + 1} / {order.length}
            </span>
            <button type="button" className="pill-button subtle" onClick={() => go(1)}>
              Next →
            </button>
          </div>
        </>
      ) : (
        <p className="live-empty">
          Your card bank is empty — add a card below to start practicing.
        </p>
      )}

      {showBank && (
        <div className="bank-list">
          <h3>Card bank</h3>
          {deck.map((card) => (
            <div className="bank-row" key={card.id}>
              <span className="kr" lang="ko">
                {card.korean}
              </span>
              <span className="en">
                {card.english}
                {card.romanization ? ` · ${card.romanization}` : ""}
              </span>
              <SpeakButton text={card.korean} label="" />
              <button
                type="button"
                className="delete"
                onClick={() => removeCard(card.id)}
                aria-label={`Delete ${card.korean}`}
              >
                remove
              </button>
            </div>
          ))}
          <form className="add-form" onSubmit={addCard}>
            <input name="korean" placeholder="한국어 (Korean)" lang="ko" required />
            <input name="english" placeholder="English meaning" required />
            <input name="romanization" placeholder="Romanization (optional)" />
            <button type="submit" className="pill-button">
              Add card
            </button>
          </form>
        </div>
      )}
    </section>
  );
}
