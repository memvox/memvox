from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ConversationTurn:
    turn_id: str
    user_message: ChatMessage
    assistant_message: ChatMessage
    timestamp: datetime


@dataclass
class WikiArticle:
    slug: str
    title: str
    body: str  # plain Markdown
    tags: list[str]
    updated_at: datetime


@dataclass
class SearchResult:
    article: WikiArticle
    score: float  # RRF-fused
    matched_chunks: list[str]  # top-2 excerpt windows, not full body


@dataclass
class CompileRequest:
    session_id: str
    transcript: list[ConversationTurn]
    existing_slugs: list[str]


@dataclass
class CompileResult:
    created: list[WikiArticle]
    updated: list[WikiArticle]
    skipped_reason: str | None = None
