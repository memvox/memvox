"""Select and construct the TTS backend named by a SessionConfig.

Keeping construction here (rather than in __main__) means a skin chooses its
voice backend declaratively via `tts_backend`, and adding a new backend is a
single branch — the orchestrator only ever sees the TTSBackend Protocol.
"""

from memvox.session.types import SessionConfig
from memvox.voice.tts_base import TTSBackend


def build_tts(config: SessionConfig) -> TTSBackend:
    backend = config.tts_backend

    if backend == "xtts":
        from memvox.voice.tts import TTSEngine

        return TTSEngine(voice=config.voice, lang_code=config.tts_lang_code)

    if backend == "cartesia":
        from memvox.voice.tts_cartesia import CartesiaTTS

        return CartesiaTTS(
            voice_id=config.cartesia_voice_id,
            lang_code=config.tts_lang_code,
            model=config.cartesia_model,
            korean_help_speed=config.cartesia_korean_help_speed,
        )

    raise ValueError(
        f"unknown tts_backend {backend!r}; expected 'xtts' or 'cartesia'"
    )
