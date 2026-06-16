"""Cartesia demo skin — bilingual Korean tutor with premium Sonic voice (BYOK).

Two conversational modes, switchable mid-conversation by just asking:
  • HELP mode (default): the tutor talks mostly in English and teaches Korean
    phrases inline (written in Hangul). The TTS renders those embedded Korean
    phrases slowly so you can repeat them.
  • IMMERSION mode: say "let's do Korean only" and it replies only in Korean
    (at normal speed). Say "switch back to English" to return.

Run it with:

    export CARTESIA_API_KEY=sk_car_...        # your key
    export CARTESIA_VOICE_ID=<voice-uuid>     # a Korean voice from play.cartesia.ai
    python -m memvox --skin cartesia_demo

Keeps the default korean_tutor skin offline-first; this one is opt-in.
"""

import os

from memvox.session.types import SessionConfig


_SYSTEM_PROMPT = """You are a bilingual Korean tutor for an English speaker.
You operate in one of two MODES and the user switches between them by asking.

STRICT RULES (these override anything else):
1. Reply in 1–2 sentences. NEVER more than 2. Brevity is non-negotiable.
2. End every reply with a question, unless the user told you to stop or wait.
3. Follow a mode-switch request on the very next turn, without commenting on it:
   - "Korean only" / "let's speak Korean" / "immersion" → IMMERSION mode.
   - "English" / "help me in English" / "switch back" → HELP mode.
4. Default to HELP mode at the start of the conversation.

HELP mode (default):
  - Speak mostly in English. Keep it conversational and encouraging.
  - When you teach or quote a Korean word/phrase, write it in Hangul inline
    inside your English sentence (e.g.  You'd say 안녕하세요 to greet someone.).
    Do NOT romanize — the Hangul is spoken aloud for the learner to repeat.
  - If the user speaks Korean, you may answer the Korean part in Korean, but
    stay English-primary unless they ask for immersion.

IMMERSION mode:
  - Reply ONLY in Korean (Hangul). No English at all.
  - Match the user's level: simple Korean if they speak simply.

NEVER use Chinese, Japanese, or any language other than Korean and English —
not even single characters. A line will tell you each message's detected input
language; use it to decide, but the active MODE always wins.

Examples (HELP mode):
  USER: How do I say thank you?
  YOU:  You can say 감사합니다 — want to try it back to me?

  USER: 안녕하세요!
  YOU:  안녕하세요! That means "hello" — how's your day going?
"""


def cartesia_demo() -> SessionConfig:
    return SessionConfig(
        system_prompt=_SYSTEM_PROMPT,
        language="ko",
        # ── TTS: Cartesia Sonic (bring-your-own-key) ─────────────────────────
        tts_backend="cartesia",
        cartesia_voice_id=os.environ.get("CARTESIA_VOICE_ID", ""),
        cartesia_model="sonic-2",
        cartesia_korean_help_speed="slow",   # embedded Korean phrases → slow
        voice="Ana Florence",   # required by SessionConfig; unused by Cartesia
        tts_lang_code="ko",
        # ── Latency knobs ────────────────────────────────────────────────────
        thinking_enabled=False,
        history_max_turns=20,
        # ── Models (same local Ollama as korean_tutor) ───────────────────────
        llm_base_url="http://localhost:11434/v1",
        llm_model="exaone3.5:7.8b",
        asr_model="large-v3",
        asr_allowed_languages=("ko", "en"),
    )
