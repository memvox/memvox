from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SessionConfig:
    system_prompt: str
    language: str                            # ISO 639-1, e.g. "ko" — used by ASR/LLM context
    voice: str                               # XTTS speaker name or speaker WAV path
    tts_lang_code: str = "en"                # XTTS ISO language code, e.g. "en" or "ko"
    # ── TTS backend selection ────────────────────────────────────────────────
    # "xtts"     → local Coqui XTTS-v2 (offline, no key)
    # "cartesia" → Cartesia Sonic (BYOK; set CARTESIA_API_KEY + cartesia_voice_id)
    tts_backend: Literal["xtts", "cartesia"] = "xtts"
    cartesia_voice_id: str = ""              # Cartesia voice UUID (required for cartesia)
    cartesia_model: str = "sonic-2"          # Sonic model id, e.g. "sonic-2"
    overlapping: bool = False                # Phase 2 concurrent pipeline
    history_max_turns: int = 20
    thinking_enabled: bool = False           # Qwen3 thinking; off by default (adds ~300–2000ms)
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen3-8B-Instruct"
    asr_model: str = "large-v3"
    # ASR auto-detects language by default. Set this to e.g. ("ko", "en") to
    # drop transcripts where Whisper picks any other language — useful to
    # filter out short-audio hallucinations into unrelated languages.
    asr_allowed_languages: tuple[str, ...] = ()
