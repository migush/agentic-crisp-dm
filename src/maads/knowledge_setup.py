"""Resolve CrewAI skills, knowledge, and tools paths for agent construction."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

from maads.ollama_runtime import ollama_base_url

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"


def repo_root() -> Path:
    return _REPO_ROOT


def skill_path(name: str) -> str:
    return str(_REPO_ROOT / "skills" / name)


def _ollama_embeddings_url() -> str:
    base = ollama_base_url()
    explicit = os.getenv("EMBEDDINGS_OLLAMA_URL") or os.getenv("OLLAMA_URL")
    if explicit:
        return explicit.rstrip("/")
    return f"{base}/api/embeddings"


def resolve_embedder_config() -> dict[str, Any] | None:
    """Return CrewAI Knowledge embedder config, preferring Ollama when offline.

    Resolution:
    - ``EMBEDDINGS_PROVIDER=ollama`` or ``EMBEDDINGS_OLLAMA_MODEL_NAME`` → Ollama
    - ``OPENAI_API_KEY`` set and provider not forced to ollama → CrewAI default (None)
    - ``MODEL`` starts with ``ollama/`` and no OpenAI key → Ollama
    """
    provider = (os.getenv("EMBEDDINGS_PROVIDER") or "").strip().lower()
    ollama_model = (
        os.getenv("EMBEDDINGS_OLLAMA_MODEL_NAME")
        or os.getenv("OLLAMA_EMBED_MODEL")
        or ""
    ).strip()
    has_openai = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    model = (os.getenv("MODEL") or "").strip()

    use_ollama = provider == "ollama" or bool(ollama_model)
    if not use_ollama and not has_openai and model.startswith("ollama/"):
        use_ollama = True

    if not use_ollama:
        return None

    return {
        "provider": "ollama",
        "config": {
            "model_name": ollama_model or _DEFAULT_OLLAMA_EMBED_MODEL,
            "url": _ollama_embeddings_url(),
        },
    }


_AGENT_SKILLS: dict[str, list[str]] = {
    "pm": ["crisp-dm-loops", "json-output-contract"],
    "domain": ["json-output-contract"],
    "data_engineer": ["leakage-cv-discipline", "tabular-prep", "nlp-prep"],
    "data_scientist": ["leakage-cv-discipline", "nlp-prep"],
    "developer": ["kaggle-submission-contract", "developer-debug-rubric"],
    "storyteller": ["json-output-contract"],
}


def skills_for(agent_name: str) -> list[str]:
    return [skill_path(s) for s in _AGENT_SKILLS.get(agent_name, [])]


def knowledge_corpus_paths(case_id: str = "") -> list[Path]:
    """Markdown files indexed for Domain RAG (shared + per-case + prior experience)."""
    knowledge_dir = _REPO_ROOT / "knowledge"
    paths = [
        knowledge_dir / "crisp-dm-excerpt.md",
        knowledge_dir / "ml-problem-approach-notes.md",
    ]
    if case_id:
        for name in (f"{case_id}.md", f"{case_id}_experience.md"):
            candidate = knowledge_dir / name
            if candidate.exists():
                paths.append(candidate)
    return [p for p in paths if p.exists()]


@lru_cache(maxsize=8)
def domain_knowledge_sources(case_id: str = "") -> list[TextFileKnowledgeSource]:
    """Shared + per-case markdown corpus for the Domain Expert (CrewAI Knowledge)."""
    rel = [p.name for p in knowledge_corpus_paths(case_id)]
    if not rel:
        return []
    return [TextFileKnowledgeSource(file_paths=rel)]


def append_experience_to_knowledge(case_id: str, experience: str) -> Path | None:
    """Loop D: persist experience documentation into knowledge for the next run."""
    if not experience or not experience.strip():
        return None
    knowledge_dir = _REPO_ROOT / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    out = knowledge_dir / f"{case_id}_experience.md"
    # Atomic write so concurrent same-case runs never leave a partial file
    # (last writer wins, which is fine — this doc accumulates for future runs).
    tmp = out.with_suffix(f".md.{os.getpid()}.tmp")
    tmp.write_text(experience.strip() + "\n", encoding="utf-8")
    os.replace(tmp, out)
    domain_knowledge_sources.cache_clear()
    from maads.rag import clear_rag_cache

    clear_rag_cache()
    return out
