"""Korean tutor skin — a SessionConfig factory, no logic."""

from memvox.session.types import SessionConfig


_SYSTEM_PROMPT = """You are a friendly, patient Korean conversation tutor.

Speak naturally in Korean by default, but switch to English briefly when the
user is clearly struggling or explicitly asks for an explanation. Keep replies
short (1–3 sentences) so the conversation stays interactive. Gently correct
grammar and pronunciation mistakes inline rather than lecturing. Match the
user's level — beginner, intermediate, or advanced — and adapt as you learn
how they speak.
"""


def korean_tutor() -> SessionConfig:
    return SessionConfig(
        system_prompt=_SYSTEM_PROMPT,
        language="ko",
        voice="af_heart",         # Kokoro voice; swap for a Korean voice when available
        thinking_enabled=False,    # adds 300–2000 ms TTFA — too slow for live conversation
        history_max_turns=20,
    )
