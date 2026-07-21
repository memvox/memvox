# memvox web UI

A minimal React + TypeScript companion for the memvox voice agent, themed for
Korean practice. Three modes, selected from the left menu:

- **Live** — the words spoken between you and the voice agent, rendered live
  during a session. Connects to the orchestrator's UI bridge
  (`ws://localhost:8765`, see `memvox/observability/uibridge.py`) and
  auto-reconnects, so it can be left open between sessions.
- **Flashcards** — a persistent vocabulary bank (browser localStorage, seeded
  with starter words). Tap to reveal, arrow keys or buttons to navigate,
  shuffle, and manage cards. Each card has a *Listen* button for reference
  pronunciation.
- **Alphabet** — the full Hangul jamo set (basic/tense consonants, basic and
  compound vowels). Tap a letter to hear it and see its name, romanization,
  sound description, and an example word.

## Run it

```bash
../run.sh ui        # or: npm install && npm run dev
```

Then open the printed URL (default http://localhost:5173). To see the Live
view populate, start a voice session alongside it: `../run.sh up`.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `VITE_MEMVOX_WS_URL` | `ws://localhost:8765` | UI bridge WebSocket address |

The orchestrator side is configured with `python -m memvox --ui-port <port>`
(`0` disables the bridge).

## Pronunciation

All pronunciation goes through one interface (`src/lib/pronounce.ts`),
currently implemented with the browser's SpeechSynthesis ko-KR voice as a
reference-quality stub. Planned upgrades keep the same interface:

1. Cartesia-rendered reference audio served via the memvox bridge.
2. Recording learner audio and scoring it against the reference
   (pronunciation assessment).

## Notes

- Styling: brockwade.com's layout (cream background, Open Sans Variable,
  nudge-on-hover links, muted metadata, narrow centered reading column with a
  slim left nav) recolored to the memvox logo palette — navy ink,
  indigo→violet primary, cyan accent, teal/magenta used sparingly. Hangul
  renders with Noto Sans KR Variable.
- Event shapes shared with the Python bridge live in `src/lib/types.ts`;
  keep them in sync with `memvox/observability/uibridge.py`.
- This directory is self-contained (own package.json/tsconfig) so it can be
  lifted into a dedicated repo later without untangling.
