import { useState } from "react";
import { HANGUL_SECTIONS } from "../data/hangul";
import type { Jamo } from "../data/hangul";
import { SpeakButton } from "../components/SpeakButton";
import { pronounce } from "../lib/pronounce";

export function Alphabet() {
  const [selected, setSelected] = useState<Jamo | null>(null);

  const pick = (jamo: Jamo) => {
    setSelected(jamo);
    pronounce(jamo.speak);
  };

  return (
    <section>
      <h2 className="mode-title">Alphabet</h2>
      <p className="mode-blurb">
        한글 — tap any letter to hear it and see how it's pronounced.
      </p>

      {HANGUL_SECTIONS.map((section) => (
        <div className="jamo-section" key={section.title}>
          <h3>{section.title}</h3>
          <div className="jamo-grid">
            {section.letters.map((jamo) => (
              <button
                type="button"
                key={jamo.glyph}
                className={`jamo-cell${selected?.glyph === jamo.glyph ? " selected" : ""}`}
                onClick={() => pick(jamo)}
              >
                <span className="glyph" lang="ko">
                  {jamo.glyph}
                </span>
                <span className="rr">{jamo.romanization}</span>
              </button>
            ))}
          </div>
        </div>
      ))}

      {selected && (
        <div className="jamo-detail">
          <span className="glyph" lang="ko">
            {selected.glyph}
          </span>
          <div className="info">
            <div className="name" lang="ko">
              {selected.name} · {selected.romanization}
            </div>
            <p className="desc">{selected.description}</p>
            <div className="example" lang="ko">
              {selected.example.word} — {selected.example.meaning}
            </div>
          </div>
          <SpeakButton text={selected.speak} slow />
        </div>
      )}
    </section>
  );
}
