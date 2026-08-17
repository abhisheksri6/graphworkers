"""specs/knowledge-graph v16 — S3: the bare-anaphor filter runs UNCONDITIONALLY (KG-AC-106).

Pre-v16 the deterministic pronoun/anaphor stoplist was applied only when `coreference_enabled`
was on — but that flag defaults to OFF, which is exactly the configuration most pipelines run.
A leaked "it"/"the Adviser" mention then became a real `kg_entities` row and, at canonicalization,
its own spurious canonical node (measured during P26: 2 leaked mentions -> 2 spurious nodes).

A bare anaphor is never a valid entity regardless of how the document was processed, so the guard
belongs to the pipeline, not to a feature flag.
"""
import pytest

from ontologies import load_pack
from strategies import Chunk, ExtractionConfig, run_pipeline

FIBO = load_pack("fibo_core")


class _FakeLlmClient:
    """The suite's established injection shape (see test_coreference.py)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
        self.resolved_model = "fake-model"
        self.usage = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def complete_tool(self, *, system_text, user_text, tool_name, tool_description, tool_schema):
        self.prompts.append(user_text)
        return self._responses.pop(0)


@pytest.mark.ac("KG-AC-106")
def test_bare_anaphors_are_dropped_with_coreference_disabled():
    """The DEFAULT configuration — coreference_enabled is False, which is precisely the case the
    pre-v16 gate left unguarded."""
    graph_response = {
        "entities": [
            {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
            {"type": "Organization", "surface": "It", "confidence": 0.8},
            {"type": "Organization", "surface": "the Company", "confidence": 0.8},
        ],
        "relations": [],
    }
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", graph_scope="s",
                           connection_id="c1")
    assert cfg.coreference_enabled is False

    chunks = [Chunk("c1", "Acme Corp exists. It exists. the Company exists.")]
    ent_rows, _edges, _summary, _usage, _blocked = run_pipeline(
        chunks, cfg, FIBO, folder_id="f1", llm_client=_FakeLlmClient([graph_response]),
    )
    surfaces = {r["surface_form"] for r in ent_rows}
    assert "Acme Corp" in surfaces, "the real entity was dropped too — the filter is over-broad"
    assert "It" not in surfaces, "a bare pronoun was written as an entity with coreference off"
    assert "the Company" not in surfaces, "a bare definite noun phrase survived with coreference off"
