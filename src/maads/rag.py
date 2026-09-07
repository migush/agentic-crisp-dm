"""Retrieval-augmented generation over the ``knowledge/`` markdown corpus.

Used by the Domain Expert task prompts (explicit passages in ``domain_corpus``).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from maads.knowledge_setup import knowledge_corpus_paths, resolve_embedder_config
from maads.ollama_runtime import (
    is_ollama_cloud_host,
    ollama_auth_headers,
    ollama_client_kwargs,
)
from maads.text_normalize import dedupe_passages, strip_markdown_headers
from maads.state import CrispDMState

_log = logging.getLogger(__name__)

_DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
_CHUNK_MAX_CHARS = 1200
_MERGE_SMALL_THRESHOLD = 400
_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass(frozen=True)
class RetrievedPassage:
    source: str
    text: str
    section: str | None = None

    def format(self) -> str:
        return f"[{self.source}] {self.text}"


@dataclass(frozen=True)
class _Chunk:
    source: str
    text: str
    section: str | None = None


@dataclass
class _Index:
    chunks: list[_Chunk]
    vectors: list[list[float]] | None
    backend: str


def clear_rag_cache() -> None:
    """Drop cached retrievers (e.g. after Loop D appends experience)."""
    _retriever_for.cache_clear()


@lru_cache(maxsize=16)
def _retriever_for(case_id: str) -> "RAGRetriever":
    return RAGRetriever(knowledge_corpus_paths(case_id))


def retrieve_for_state(state: CrispDMState, *, k: int = 6) -> list[str]:
    """Return top-k RAG passages for a CRISP-DM state."""
    query = build_retrieval_query(state)
    return _retriever_for(state.case_id).retrieve(query, k=k)


def retrieve_passages_for_state(state: CrispDMState, *, k: int = 6) -> list[RetrievedPassage]:
    """Return structured top-k RAG passages for a CRISP-DM state."""
    query = build_retrieval_query(state)
    return _retriever_for(state.case_id).retrieve_passages(query, k=k)


def rag_status(case_id: str) -> dict[str, Any]:
    """Index metadata for dashboards and observability."""
    paths = knowledge_corpus_paths(case_id)
    if not paths:
        return {
            "case_id": case_id,
            "corpus_files": [],
            "chunk_count": 0,
            "embedding_backend": "none",
            "embedding_model": None,
            "crewai_knowledge_enabled": False,
            "explicit_rag_enabled": False,
        }
    retriever = RAGRetriever(paths)
    embed_cfg = resolve_embedder_config() or {}
    model = None
    if retriever.backend == "ollama":
        model = (embed_cfg.get("config") or {}).get("model_name")
    elif retriever.backend == "openai":
        model = (os.getenv("OPENAI_EMBED_MODEL") or _DEFAULT_OPENAI_EMBED_MODEL).strip()
    return {
        "case_id": case_id,
        "corpus_files": [_corpus_file_entry(p, case_id) for p in paths],
        "chunk_count": retriever.chunk_count,
        "embedding_backend": retriever.backend,
        "embedding_model": model,
        "crewai_knowledge_enabled": False,
        "explicit_rag_enabled": True,
    }


def _corpus_file_entry(path: Path, case_id: str) -> dict[str, Any]:
    name = path.name
    if name.endswith("_experience.md"):
        role = "experience"
    elif name == f"{case_id}.md":
        role = "case"
    else:
        role = "shared"
    stat = path.stat()
    return {
        "name": name,
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "role": role,
    }


def _report_query_snippet(report: dict[str, Any] | None) -> str:
    if not report:
        return ""
    parts: list[str] = []
    for key in ("row_count", "train_rows", "test_rows", "target_column", "summary"):
        val = report.get(key)
        if val is not None:
            parts.append(f"{key}={val}")
    blockers = report.get("blockers") or report.get("quality_blockers")
    if isinstance(blockers, list) and blockers:
        parts.append("blockers=" + "; ".join(str(b) for b in blockers[:5]))
    target_dist = report.get("target_distribution") or report.get("class_balance")
    if target_dist is not None:
        parts.append(f"target_distribution={json.dumps(target_dist, default=str)[:500]}")
    if parts:
        return " ".join(parts)
    return json.dumps(report, default=str)[:500]


def build_retrieval_query(state: CrispDMState) -> str:
    """Build a retrieval query from case config and summarized DU reports."""
    cfg = state.config
    parts = [
        cfg.case_id,
        cfg.problem_type,
        cfg.target_column,
        cfg.evaluation_metric,
        cfg.problem_statement or "",
        cfg.kaggle_competition or "",
    ]
    for report in (
        state.du.data_description_report,
        state.du.data_quality_report,
        state.du.data_exploration_report,
    ):
        snippet = _report_query_snippet(report)
        if snippet:
            parts.append(snippet)
    return " ".join(p for p in parts if p)


class RAGRetriever:
    """Embed or keyword-search paragraph chunks from markdown knowledge files."""

    def __init__(self, corpus_paths: list[Path] | Path | None = None) -> None:
        if corpus_paths is None:
            paths = knowledge_corpus_paths("")
        elif isinstance(corpus_paths, Path):
            if corpus_paths.is_dir():
                paths = sorted(corpus_paths.glob("*.md"))
            else:
                paths = [corpus_paths]
        else:
            paths = list(corpus_paths)
        self._paths = [Path(p) for p in paths if Path(p).exists()]
        self._index = self._build_index()

    @property
    def backend(self) -> str:
        return self._index.backend

    @property
    def chunk_count(self) -> int:
        return len(self._index.chunks)

    @property
    def corpus_paths(self) -> list[Path]:
        return list(self._paths)

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        """Return top-k passages formatted as ``[source] text``."""
        return [p.format() for p in self.retrieve_passages(query, k=k)]

    def retrieve_passages(self, query: str, k: int = 4) -> list[RetrievedPassage]:
        """Return top-k structured passages after ranking and deduplication."""
        if not query.strip() or not self._index.chunks:
            return []
        k = max(1, k)
        if self._index.vectors:
            ranked = self._rank_by_embedding(query, k)
        else:
            ranked = self._rank_by_keywords(query, k)
        formatted = [RetrievedPassage(c.source, c.text, c.section) for c in ranked]
        deduped_strings = dedupe_passages([p.format() for p in formatted])
        by_format = {p.format(): p for p in formatted}
        return [by_format[s] for s in deduped_strings if s in by_format]

    def _build_index(self) -> _Index:
        chunks: list[_Chunk] = []
        for path in self._paths:
            text = path.read_text(encoding="utf-8")
            for section, body in _split_markdown_sections(text):
                for para_text in _chunk_section_body(body):
                    cleaned = strip_markdown_headers(para_text, keep_first=True)
                    if cleaned:
                        chunks.append(_Chunk(source=path.name, text=cleaned, section=section))
        chunks = _merge_small_chunks(chunks)
        backend, vectors = _embed_chunks([c.text for c in chunks])
        return _Index(chunks=chunks, vectors=vectors, backend=backend)

    def _rank_by_embedding(self, query: str, k: int) -> list[_Chunk]:
        assert self._index.vectors is not None
        q_vecs = _embed_texts([query], self._index.backend)
        if not q_vecs:
            return self._rank_by_keywords(query, k)
        q_vec = q_vecs[0]
        scored: list[tuple[float, int]] = []
        for i, vec in enumerate(self._index.vectors):
            scored.append((_cosine(q_vec, vec), i))
        scored.sort(key=lambda x: -x[0])
        return [self._index.chunks[i] for _, i in scored[:k]]

    def _rank_by_keywords(self, query: str, k: int) -> list[_Chunk]:
        terms = {t.lower() for t in re.findall(r"[a-zA-Z0-9_]{3,}", query)}
        if not terms:
            return self._index.chunks[:k]
        scored: list[tuple[int, int]] = []
        for i, chunk in enumerate(self._index.chunks):
            lower = chunk.text.lower()
            score = sum(1 for t in terms if t in lower)
            if score:
                scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return self._index.chunks[:k]
        return [self._index.chunks[i] for _, i in scored[:k]]


def _split_markdown_sections(text: str) -> list[tuple[str | None, str]]:
    """Split markdown into (section_heading, body) pairs."""
    sections: list[tuple[str | None, str]] = []
    current_section: str | None = None
    current_lines: list[str] = []
    doc_title: str | None = None

    for line in text.splitlines():
        m = _HEADING_LINE_RE.match(line.strip())
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()
            if current_lines:
                sections.append((current_section, "\n".join(current_lines).strip()))
                current_lines = []
            if level == 1 and doc_title is None:
                doc_title = heading
                current_section = None
            else:
                current_section = heading
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, "\n".join(current_lines).strip()))

    if not sections and text.strip():
        return [(doc_title, text.strip())]
    return sections


def _chunk_section_body(body: str) -> list[str]:
    """Split section body into chunks; continuation pieces omit repeated headings."""
    if not body.strip():
        return []
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    out: list[str] = []
    for para in paragraphs:
        if len(para) <= _CHUNK_MAX_CHARS:
            out.append(para)
            continue
        for i in range(0, len(para), _CHUNK_MAX_CHARS):
            piece = para[i : i + _CHUNK_MAX_CHARS].strip()
            if not piece:
                continue
            if i == 0:
                out.append(piece)
            else:
                out.append(strip_markdown_headers(piece, keep_first=False))
    return out


def _merge_small_chunks(chunks: list[_Chunk]) -> list[_Chunk]:
    """Merge adjacent small chunks from the same source and section."""
    if not chunks:
        return []
    merged: list[_Chunk] = []
    buffer: _Chunk | None = None
    for chunk in chunks:
        if buffer is None:
            buffer = chunk
            continue
        same_group = (
            buffer.source == chunk.source
            and buffer.section == chunk.section
            and len(buffer.text) + len(chunk.text) + 2 <= _MERGE_SMALL_THRESHOLD
        )
        if same_group and len(buffer.text) < _MERGE_SMALL_THRESHOLD:
            buffer = _Chunk(
                source=buffer.source,
                text=f"{buffer.text}\n\n{chunk.text}",
                section=buffer.section,
            )
        else:
            merged.append(buffer)
            buffer = chunk
    if buffer is not None:
        merged.append(buffer)
    return merged


def _embed_chunks(texts: list[str]) -> tuple[str, list[list[float]] | None]:
    if not texts:
        return "none", None
    ollama_cfg = resolve_embedder_config()
    if ollama_cfg is not None:
        vecs = _embed_texts(texts, "ollama")
        if vecs:
            return "ollama", vecs
        _log.warning("Ollama embeddings failed; falling back to keyword RAG")
    if (os.getenv("OPENAI_API_KEY") or "").strip():
        vecs = _embed_texts(texts, "openai")
        if vecs:
            return "openai", vecs
        _log.warning("OpenAI embeddings failed; falling back to keyword RAG")
    return "keyword", None


def _embed_texts(texts: list[str], backend: str) -> list[list[float]] | None:
    if not texts:
        return []
    try:
        if backend == "ollama":
            return _embed_ollama(texts)
        if backend == "openai":
            return _embed_openai(texts)
    except Exception as exc:  # noqa: BLE001 — degrade to keyword search
        _log.debug("embedding backend %s failed: %s", backend, exc)
    return None


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    cfg = resolve_embedder_config() or {}
    model = (cfg.get("config") or {}).get("model_name") or "nomic-embed-text"
    try:
        import ollama

        client = ollama.Client(**ollama_client_kwargs())
        resp = client.embed(model=model, input=texts)
        embeddings = resp.embeddings
        if embeddings and len(embeddings) == len(texts):
            return [list(map(float, e)) for e in embeddings]
    except Exception:
        pass
    return _embed_ollama_http(texts, model)


def _embed_ollama_http(texts: list[str], model: str) -> list[list[float]]:
    cfg = resolve_embedder_config() or {}
    url = (cfg.get("config") or {}).get("url") or f"{ollama_client_kwargs()['host']}/api/embeddings"
    out: list[list[float]] = []
    headers = {"Content-Type": "application/json", **ollama_auth_headers()}
    for text in texts:
        body = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        embedding = payload.get("embedding")
        if not embedding:
            raise ValueError(f"no embedding in Ollama response from {url}")
        out.append([float(x) for x in embedding])
    return out


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    model = (os.getenv("OPENAI_EMBED_MODEL") or _DEFAULT_OPENAI_EMBED_MODEL).strip()
    client = OpenAI()
    batch_size = 64
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend([list(d.embedding) for d in resp.data])
    return vectors


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def ensure_embedding_model_available() -> str | None:
    """Pull the default Ollama embedding model when using local embeddings.

    Returns a warning message when the model could not be verified, else None.
    Does not ``ollama pull`` against ollama.com.
    """
    cfg = resolve_embedder_config()
    if cfg is None:
        return None
    model = (cfg.get("config") or {}).get("model_name") or "nomic-embed-text"
    try:
        import ollama

        client = ollama.Client(**ollama_client_kwargs())
        client.show(model)
        return None
    except Exception:
        if is_ollama_cloud_host():
            return None
        try:
            import ollama

            client = ollama.Client(**ollama_client_kwargs())
            _log.info("Pulling Ollama embedding model %s …", model)
            client.pull(model)
            return None
        except Exception as exc:
            return (
                f"Could not verify or pull Ollama embedding model '{model}': {exc}. "
                f"Run: ollama pull {model}"
            )
