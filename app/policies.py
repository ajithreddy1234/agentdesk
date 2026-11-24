"""Policy knowledge base: section-level chunks, embeddings, and hybrid search.

Backends: in-memory (numpy + BM25) for tests and laptops; pgvector + Postgres full-text
search in production. Both fuse dense and lexical rankings with Reciprocal Rank Fusion.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import settings

POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"


@dataclass
class PolicyChunk:
    id: str
    doc: str
    title: str
    section: str
    text: str

    @property
    def embed_text(self) -> str:
        return f"{self.title} - {self.section}\n{self.text}"


def load_chunks(folder: Path = POLICY_DIR) -> list[PolicyChunk]:
    chunks = []
    for md in sorted(folder.glob("*.md")):
        raw = md.read_text()
        title = raw.splitlines()[0].lstrip("# ").strip()
        for part in re.split(r"\n(?=## )", raw):
            lines = part.strip().splitlines()
            if not lines or lines[0].startswith("# "):
                body = "\n".join(l for l in lines if not l.startswith("# ")).strip()
                section = "Overview"
            else:
                section, body = lines[0].lstrip("# ").strip(), "\n".join(lines[1:]).strip()
            if body:
                cid = hashlib.md5(f"{md.stem}:{section}".encode()).hexdigest()[:10]
                chunks.append(PolicyChunk(cid, md.stem, title, section, re.sub(r"\s+", " ", body)))
    return chunks


# ---------- embeddings ----------
class HashEmbedder:
    dim = 384

    def embed(self, texts, is_query=False):  # noqa: ARG002
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            toks = re.findall(r"[a-z0-9]+", t.lower())
            for tok in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1 if h & 1 else -1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class STEmbedder:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.prefix = "Represent this sentence for searching relevant passages: " if "bge" in name.lower() else ""

    def embed(self, texts, is_query=False):
        if is_query and self.prefix:
            texts = [self.prefix + t for t in texts]
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype="float32")


@lru_cache(maxsize=1)
def get_embedder():
    return HashEmbedder() if settings.embedding_backend == "hash" else STEmbedder(settings.embedding_model)


_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in on for and or is are be by with at as it this that from can do does what how when which my i".split())


def _tok(s):
    return [t for t in _TOK.findall(s.lower()) if t not in _STOP]


def _rrf(*rankings: list[str], k: int = 60) -> list[tuple[str, float]]:
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking):
            score[cid] = score.get(cid, 0) + 1 / (k + r + 1)
    return sorted(score.items(), key=lambda x: -x[1])


class MemoryPolicyStore:
    def __init__(self, chunks: list[PolicyChunk] | None = None):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks or load_chunks()
        self.by_id = {c.id: c for c in self.chunks}
        self.emb = get_embedder()
        self.vecs = self.emb.embed([c.embed_text for c in self.chunks])
        self.bm25 = BM25Okapi([_tok(c.embed_text) for c in self.chunks])

    def search(self, query: str, k: int = 4) -> list[tuple[PolicyChunk, float]]:
        q = self.emb.embed([query], is_query=True)[0]
        dense = np.argsort(-(self.vecs @ q))[:20]
        lex = np.argsort(-self.bm25.get_scores(_tok(query)))[:20]
        fused = _rrf([self.chunks[i].id for i in dense], [self.chunks[i].id for i in lex])[:k]
        return [(self.by_id[cid], s) for cid, s in fused]


class PgVectorPolicyStore:
    """Requires the pgvector extension (use the pgvector/pgvector:pg16 image)."""

    def __init__(self, url: str | None = None):
        from sqlalchemy import create_engine

        self.engine = create_engine(url or settings.database_url)
        self.emb = get_embedder()

    def index(self, chunks: list[PolicyChunk] | None = None):
        from sqlalchemy import text

        chunks = chunks or load_chunks()
        vecs = self.emb.embed([c.embed_text for c in chunks])
        with self.engine.begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            c.execute(text(f"""CREATE TABLE IF NOT EXISTS policy_chunks (
                id TEXT PRIMARY KEY, doc TEXT, title TEXT, section TEXT, body TEXT,
                embedding vector({vecs.shape[1]}), tsv tsvector)"""))
            c.execute(text("CREATE INDEX IF NOT EXISTS policy_tsv_idx ON policy_chunks USING gin(tsv)"))
            c.execute(text("TRUNCATE policy_chunks"))
            for ch, v in zip(chunks, vecs):
                c.execute(text("""INSERT INTO policy_chunks VALUES (:id, :doc, :title, :section, :body, CAST(:emb AS vector),
                                  to_tsvector('english', :title || ' ' || :section || ' ' || :body))"""),
                          {"id": ch.id, "doc": ch.doc, "title": ch.title, "section": ch.section, "body": ch.text,
                           "emb": json.dumps(v.tolist())})

    def search(self, query: str, k: int = 4) -> list[tuple[PolicyChunk, float]]:
        from sqlalchemy import text

        q = json.dumps(self.emb.embed([query], is_query=True)[0].tolist())
        with self.engine.connect() as c:
            dense = c.execute(text("SELECT id, doc, title, section, body FROM policy_chunks "
                                   "ORDER BY embedding <=> CAST(:q AS vector) LIMIT 20"), {"q": q}).fetchall()
            # OR the question's words (websearch_to_tsquery would require all of them) and let ts_rank order them
            terms = " | ".join(t for t in _tok(query) if len(t) > 1) or "''"
            lex = c.execute(text("SELECT id, doc, title, section, body FROM policy_chunks, "
                                 "to_tsquery('english', :q) query WHERE tsv @@ query "
                                 "ORDER BY ts_rank_cd(tsv, query) DESC LIMIT 20"), {"q": terms}).fetchall()
        rows = {r.id: PolicyChunk(r.id, r.doc, r.title, r.section, r.body) for r in [*dense, *lex]}
        fused = _rrf([r.id for r in dense], [r.id for r in lex])[:k]
        return [(rows[cid], s) for cid, s in fused]


@lru_cache(maxsize=1)
def get_policy_store():
    return PgVectorPolicyStore() if settings.vector_backend == "pgvector" else MemoryPolicyStore()
