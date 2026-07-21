import { pronounce, pronunciationAvailable } from "../lib/pronounce";

interface Props {
  text: string;
  label?: string;
  slow?: boolean;
}

/** Small orange speaker control shared by Flashcards and Alphabet modes. */
export function SpeakButton({ text, label = "Listen", slow }: Props) {
  if (!pronunciationAvailable()) return null;
  return (
    <button
      type="button"
      className="speak-button"
      onClick={(e) => {
        e.stopPropagation();
        pronounce(text, { slow });
      }}
      aria-label={`Pronounce ${text}`}
    >
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3a4.5 4.5 0 0 0-2.5-4.03v8.05A4.5 4.5 0 0 0 16.5 12zM14 3.23v2.06a7 7 0 0 1 0 13.42v2.06a9 9 0 0 0 0-17.54z" />
      </svg>
      {label}
    </button>
  );
}
