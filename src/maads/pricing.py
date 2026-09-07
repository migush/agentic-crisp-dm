"""Per-model USD pricing lookup, backed by LiteLLM's community-maintained cost map.

Deliberately does NOT hand-maintain a local pricing table: LiteLLM (already the
LLM-routing layer CrewAI uses under the hood — see ``model_catalog.py``'s
``ollama/`` id prefix) ships ``litellm.model_cost``, which it refreshes from
https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
at import time (falling back to a bundled local copy if offline). Upgrading the
``litellm`` package is the only "maintenance" this ever needs — no per-model
edits when a provider changes prices or ships a new model.

Pure-Python module (only depends on ``litellm``), safe to import from both the
CLI/pipeline and the ``webapp`` account backend.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _model_cost() -> dict[str, dict]:
    import litellm

    return litellm.model_cost


def price_for_model(model_id: str) -> dict[str, float] | None:
    """Return ``{"input_per_1m": ..., "output_per_1m": ...}`` for ``model_id``, or ``None``.

    Tries ``model_id`` as-is, then with a leading ``"<provider>/"`` prefix
    stripped (LiteLLM's cost map keys some providers, e.g. plain OpenAI ids,
    without the prefix used elsewhere in this codebase for LLM routing).
    """
    entry = _model_cost().get(model_id)
    if entry is None and "/" in model_id:
        entry = _model_cost().get(model_id.split("/", 1)[1])
    if entry is None:
        # model_catalog.py only curates free-tier Ollama Cloud tags (see its
        # module docstring), so a tag LiteLLM hasn't caught up with yet is
        # still $0 — not a guess, a property of which tags get curated.
        if model_id.startswith("ollama/"):
            return {"input_per_1m": 0.0, "output_per_1m": 0.0}
        return None
    return {
        "input_per_1m": entry.get("input_cost_per_token", 0.0) * 1_000_000,
        "output_per_1m": entry.get("output_cost_per_token", 0.0) * 1_000_000,
    }


def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for ``input_tokens``/``output_tokens`` spent on ``model_id``.

    Falls back to ``0.0`` for unknown models rather than raising, since pricing
    is advisory display data, not something that should ever break a run.
    """
    prices = price_for_model(model_id)
    if prices is None:
        return 0.0
    input_cost = (input_tokens / 1_000_000) * prices.get("input_per_1m", 0.0)
    output_cost = (output_tokens / 1_000_000) * prices.get("output_per_1m", 0.0)
    return input_cost + output_cost
