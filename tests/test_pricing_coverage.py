"""Every model in model_catalog() must resolve a price via LiteLLM's cost map.

No hand-maintained pricing table to keep in sync here — pricing.py reads
litellm.model_cost directly, so this test just guards against LiteLLM dropping
coverage for a model this catalog still offers, or the lookup logic breaking.
"""

from maads.model_catalog import model_catalog
from maads.pricing import estimate_cost_usd, price_for_model


def _all_catalog_ids() -> set[str]:
    ids: set[str] = set()
    for entries in model_catalog().values():
        ids.update(entry["id"] for entry in entries)
    return ids


def test_every_catalog_model_has_pricing():
    missing = {model_id for model_id in _all_catalog_ids() if price_for_model(model_id) is None}
    assert not missing, f"model_catalog() ids with no LiteLLM pricing entry: {missing}"


def test_price_for_unknown_model_is_none():
    assert price_for_model("not-a-real-model") is None


def test_estimate_cost_usd_computes_input_plus_output():
    cost = estimate_cost_usd("gpt-5.4-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    prices = price_for_model("gpt-5.4-mini")
    assert cost == prices["input_per_1m"] + prices["output_per_1m"]


def test_free_ollama_cloud_models_are_zero_cost():
    assert estimate_cost_usd("ollama/gpt-oss:120b-cloud", input_tokens=1_000_000, output_tokens=1_000_000) == 0.0


def test_estimate_cost_usd_unknown_model_is_zero():
    assert estimate_cost_usd("not-a-real-model", input_tokens=1000, output_tokens=1000) == 0.0
