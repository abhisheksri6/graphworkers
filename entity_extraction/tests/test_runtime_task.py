"""KG-AC-20 (worker side): the runtime preview runs the SAME engine on inline sample_text and
returns entities + relations + usage[] — STORE-ONLY (run_preview has no DB handle; it can't and
doesn't write kg_entities/kg_edges, no run_state, no M1–M4)."""
import pytest

import runtime


class _Llm:
    resolved_model = "amazon.nova-pro-v1:0"
    usage = [{"charge_category": "llm", "model": "amazon.nova-pro-v1:0",
              "input_tokens": 8, "output_tokens": 4, "connection_id": "c"}]

    def complete_tool(self, **_kwargs):
        return {"entities": [{"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
                             {"type": "Person", "surface": "Jane Roe", "confidence": 0.8}],
                "relations": []}


@pytest.mark.ac("KG-AC-20")
def test_runtime_preview_returns_entities_store_only():
    out = runtime.run_preview(
        {"engine": "llm", "ontology_pack": "fibo_core"},
        "Acme Corp is led by Jane Roe.", llm_client=_Llm(),
    )
    assert out["status"] == "success"
    types = {e["entity_type"] for e in out["entities"]}
    assert {"Organization", "Person"} <= types
    assert out["ontology_pack"] == "fibo_core" and out["ontology_version"] == "1.1"
    assert out["usage"] and out["usage"][0]["charge_category"] == "llm"
    # store-only: the preview payload carries entities/relations/usage — no persistence keys.
    assert "folders" not in out and "run_state" not in out


@pytest.mark.ac("KG-AC-20")
def test_runtime_preview_unknown_type_counted_not_written():
    class _LlmInvented:
        resolved_model = "m"
        usage: list = []

        def complete_tool(self, **_kwargs):
            return {"entities": [{"type": "Cryptocurrency", "surface": "DogeCoin", "confidence": 0.9}],
                    "relations": []}

    out = runtime.run_preview({"engine": "llm", "ontology_pack": "fibo_core"}, "DogeCoin", llm_client=_LlmInvented())
    assert out["entities"] == []               # invented type dropped
    assert out["unmapped_type_count"] == 1     # ...and counted
