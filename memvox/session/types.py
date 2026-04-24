from dataclasses import dataclass, field


@dataclass
class SessionConfig:
    system_prompt: str
    language: str                            # ISO 639-1, e.g. "ko"
    voice: str                               # Kokoro voice identifier
    overlapping: bool = False                # Phase 2 concurrent pipeline
    history_max_turns: int = 20
    thinking_enabled: bool = False           # Qwen3 thinking; off by default (adds ~300–2000ms)
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen3-8B-Instruct"
    asr_model: str = "large-v3"
