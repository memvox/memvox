import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa

from memvox.wiki.types import WikiArticle, SearchResult

_UTC = timezone.utc

# Approximate 300-token / 50-token overlap using word counts (~1.3 tokens/word)
_CHUNK_WORDS = 230
_OVERLAP_WORDS = 38

_SCHEMA = pa.schema([
    pa.field("slug", pa.utf8()),
    pa.field("chunk_text", pa.utf8()),
    pa.field("embedding", pa.list_(pa.float32(), 384)),
    pa.field("tags", pa.list_(pa.utf8())),
    pa.field("updated_at", pa.float64()),  # POSIX timestamp
])


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + _CHUNK_WORDS, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += _CHUNK_WORDS - _OVERLAP_WORDS
    return chunks


def _parse_markdown(slug: str, content: str) -> WikiArticle:
    title = slug
    tags: list[str] = []
    updated_at = datetime.now(_UTC)
    body = content

    if content.startswith("---\n"):
        end = content.find("\n---\n", 4)
        if end != -1:
            fm = content[4:end]
            body = content[end + 5:].lstrip("\n")
            for line in fm.splitlines():
                if line.startswith("title: "):
                    title = line[7:].strip()
                elif line.startswith("tags: "):
                    tags = [t.strip() for t in line[6:].split(",") if t.strip()]
                elif line.startswith("updated_at: "):
                    try:
                        updated_at = datetime.fromisoformat(line[12:].strip())
                    except ValueError:
                        pass

    return WikiArticle(slug=slug, title=title, body=body, tags=tags, updated_at=updated_at)


def _safe_slug(slug: str) -> str:
    """Validate slug is safe to interpolate into LanceDB filter expressions."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9\-]*", slug):
        raise ValueError(f"Invalid slug: {slug!r}. Must match [a-z0-9][a-z0-9-]*")
    return slug


class WikiStore:
    """Persist, index, and retrieve user knowledge as Markdown + hybrid vector search.

    Call `await store.initialize()` before any other method.
    """

    def __init__(self, wiki_dir: str | Path, db_path: str | Path) -> None:
        self._wiki_dir = Path(wiki_dir)
        self._db_path = Path(db_path)
        self._wiki_dir.mkdir(parents=True, exist_ok=True)
        self._model = None   # SentenceTransformer, loaded in initialize()
        self._db = None      # lancedb.AsyncConnection
        self._table = None   # lancedb.AsyncTable

    async def initialize(self) -> None:
        from sentence_transformers import SentenceTransformer
        import lancedb
        from lancedb.index import FTS

        self._model = await asyncio.to_thread(
            SentenceTransformer, "all-MiniLM-L6-v2"
        )
        self._db = await lancedb.connect_async(str(self._db_path))

        names = await self._db.table_names()
        if "chunks" in names:
            self._table = await self._db.open_table("chunks")
        else:
            self._table = await self._db.create_table("chunks", schema=_SCHEMA)
            await self._table.create_index("chunk_text", config=FTS(), replace=True)

    # ── write ──────────────────────────────────────────────────────────────────

    async def upsert_article(self, article: WikiArticle) -> None:
        _safe_slug(article.slug)

        # Write source Markdown (source of truth)
        path = self._wiki_dir / f"{article.slug}.md"
        fm = (
            f"---\n"
            f"title: {article.title}\n"
            f"tags: {', '.join(article.tags)}\n"
            f"updated_at: {article.updated_at.isoformat()}\n"
            f"---\n\n"
            f"{article.body}"
        )
        await asyncio.to_thread(path.write_text, fm, encoding="utf-8")

        # Rebuild index for this slug
        await self._table.delete(f"slug = '{article.slug}'")

        chunks = _chunk_text(article.body)
        if not chunks:
            return

        embeddings: np.ndarray = await asyncio.to_thread(
            self._model.encode, chunks, normalize_embeddings=True
        )

        rows = [
            {
                "slug": article.slug,
                "chunk_text": chunk,
                "embedding": emb.tolist(),
                "tags": article.tags,
                "updated_at": article.updated_at.timestamp(),
            }
            for chunk, emb in zip(chunks, embeddings)
        ]
        await self._table.add(rows)

    async def delete_article(self, slug: str) -> None:
        _safe_slug(slug)
        path = self._wiki_dir / f"{slug}.md"
        if path.exists():
            await asyncio.to_thread(path.unlink)
        await self._table.delete(f"slug = '{slug}'")

    # ── read ───────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Hybrid vector + BM25 search with RRF fusion."""
        k = top_k * 3

        query_emb: np.ndarray = await asyncio.to_thread(
            self._model.encode, [query], normalize_embeddings=True
        )

        # Vector search — async lancedb: await search() first, then chain.
        vec_query = await self._table.search(query_emb[0].tolist())
        vec_rows = await vec_query.limit(k).to_list()

        # Full-text search
        try:
            fts_query = await self._table.search(query, query_type="fts")
            fts_rows = await fts_query.limit(k).to_list()
        except Exception:
            fts_rows = []

        # RRF fusion (k=60 is standard)
        rrf_scores: dict[str, float] = {}
        slug_chunks: dict[str, list[str]] = {}

        for rank, row in enumerate(vec_rows):
            slug = row["slug"]
            rrf_scores[slug] = rrf_scores.get(slug, 0.0) + 1.0 / (60 + rank + 1)
            slug_chunks.setdefault(slug, []).append(row["chunk_text"])

        for rank, row in enumerate(fts_rows):
            slug = row["slug"]
            rrf_scores[slug] = rrf_scores.get(slug, 0.0) + 1.0 / (60 + rank + 1)
            chunk = row["chunk_text"]
            if chunk not in slug_chunks.get(slug, []):
                slug_chunks.setdefault(slug, []).append(chunk)

        top_slugs = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

        results: list[SearchResult] = []
        for slug in top_slugs:
            article = await self.get_article(slug)
            if article is None:
                continue
            results.append(SearchResult(
                article=article,
                score=rrf_scores[slug],
                matched_chunks=slug_chunks[slug][:2],
            ))

        return results

    async def get_article(self, slug: str) -> WikiArticle | None:
        path = self._wiki_dir / f"{slug}.md"
        if not path.exists():
            return None
        content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return _parse_markdown(slug, content)

    async def list_articles(self) -> list[WikiArticle]:
        paths = await asyncio.to_thread(lambda: sorted(self._wiki_dir.glob("*.md")))
        articles: list[WikiArticle] = []
        for p in paths:
            content = await asyncio.to_thread(p.read_text, encoding="utf-8")
            articles.append(_parse_markdown(p.stem, content))
        return articles
