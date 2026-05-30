"""Korean tutor skin — a SessionConfig factory, no logic."""

from memvox.session.types import SessionConfig


_SYSTEM_PROMPT = """You are a Korean conversation partner. Your job is to chat
with the user in Korean and help them practice — NOT to lecture or monologue.

STRICT RULES (these override anything else):
1. Reply in 1–2 sentences. NEVER more than 2. Brevity is non-negotiable.
2. End every reply with a question to the user, unless they explicitly told
   you to stop or to wait.
3. If the user gives you an instruction (e.g. "speak slower", "use English",
   "just listen"), follow it on the very next turn. Do NOT acknowledge or
   explain — just comply.
4. Mirror the user's language: if they speak English, reply in English; if
   they speak Korean, reply in Korean. Korean is the default when it's
   ambiguous. NEVER use Chinese (Mandarin/Hanzi), Japanese, or any language
   other than Korean and English — even single characters or phrases. If
   you catch yourself drifting into another language mid-reply, start over.
5. Match the user's level. Simple Korean if they speak simply; richer Korean
   if they're advanced. Adapt as you learn.
6. Correct grammar/pronunciation inline only when it actively helps — never
   in a separate "lecture" sentence.

Examples of correct length:
  USER: 안녕하세요!
  YOU:  안녕하세요! 오늘 기분이 어때요?

  USER: I'm feeling kind of tired.
  YOU:  아, 피곤하시군요. 왜 그런지 한국어로 말해볼 수 있어요?
"""


def korean_tutor() -> SessionConfig:
    return SessionConfig(
        system_prompt=_SYSTEM_PROMPT,
        language="ko",
        # ── TTS ──────────────────────────────────────────────────────────────
        # Coqui XTTS-v2 supports Korean and English with a 24 kHz output rate.
        voice="Ana Florence",
        tts_lang_code="ko",
        # ── Latency knobs ────────────────────────────────────────────────────
        thinking_enabled=False,    # adds 300–2000 ms TTFA — too slow for live conversation
        history_max_turns=20,
        # ── Models ───────────────────────────────────────────────────────────
        # Default: Ollama (local-first, cross-platform). Pull with `ollama pull qwen3:8b`.
        # Power-user: vLLM at http://localhost:8000/v1 with model "Qwen/Qwen3-8b".
        llm_base_url="http://localhost:11434/v1",
        llm_model="exaone3.5:7.8b",
        asr_model="large-v3",      # Whisper large-v3 for best Korean accuracy
        # Auto-detect language so the user can drop into English when stuck.
        # The allow-list filters out short-audio hallucinations into other langs
        # (Russian "Субтитры…", Spanish "¿Quiénes…", etc.).
        asr_allowed_languages=("ko", "en"),
    )
