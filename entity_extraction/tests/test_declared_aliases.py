"""KG-AC-96 (P26) — **document-declared aliases**, extraction side: Tier 2 of the identity hierarchy.

A defined term (`GLOBAL UNIVERSITY ENDOWMENT (hereinafter referred to as the "Investor")`) is **not
an anaphor** — the document STATES the binding. Capturing it as data makes the LLM's job "read the
stated definition" (which need succeed ONCE per document) instead of coref's "rewrite every
occurrence correctly" (which must succeed ~33 times and measured 94% live, minting 2 spurious
canonical nodes from its 2-mention residue).

The field is optional and additive throughout: a model that omits `aliases`, or a document with no
declared terms, produces byte-identical behaviour to pre-P26.
"""
import pytest

from core import Candidate, build_entity_records, merge_candidates
from ontologies import load_pack
from strategies import Chunk, ExtractionConfig
from strategies.llm_graph import (
    LlmGraphStrategy, build_graph_system_prompt, build_graph_tool_schema,
)

PACK = load_pack("investment_fibo")


class _FakeClient:
    """Returns one canned tool response, mirroring BedrockLlmClient.complete_tool's parsed shape."""

    def __init__(self, payload):
        self._payload = payload
        self.usage = []

    def complete_tool(self, **kwargs):
        return self._payload


# ---- tool schema + prompt: the contract the model is given --------------------------------------
@pytest.mark.ac("KG-AC-96")
def test_tool_schema_entity_item_offers_an_optional_aliases_array():
    schema = build_graph_tool_schema(PACK)
    item = schema["properties"]["entities"]["items"]
    assert item["properties"]["aliases"]["type"] == "array"
    assert item["properties"]["aliases"]["items"]["type"] == "string"
    # optional -- a model that omits it stays schema-valid (additive, KG-AC-96)
    assert "aliases" not in item["required"]


@pytest.mark.ac("KG-AC-96")
def test_prompt_binds_aliases_to_declared_terms_only():
    prompt = build_graph_system_prompt(PACK)
    low = prompt.lower()
    assert "alias" in low
    # the instruction must be DECLARED-only -- the whole point is that this is stated data, not
    # inference; a prompt that invites the model to guess re-creates coref's failure mode.
    assert "hereinafter" in low
    assert "never invent" in low or "do not invent" in low


# ---- extraction: aliases reach the Candidate ----------------------------------------------------
@pytest.mark.ac("KG-AC-96")
def test_declared_aliases_are_captured_onto_the_candidate():
    text = 'GLOBAL UNIVERSITY ENDOWMENT (hereinafter referred to as the "Investor") agrees.'
    client = _FakeClient({
        "entities": [{"type": "Investor", "surface": "GLOBAL UNIVERSITY ENDOWMENT",
                      "aliases": ["Investor", "the Investor"]}],
        "relations": [], "facts": [],
    })
    cands = LlmGraphStrategy(llm_client=client).extract(
        [Chunk("c1", text)], ExtractionConfig(engine="llm", ontology_pack="investment_fibo"), PACK)
    assert len(cands) == 1
    assert cands[0].declared_aliases == ["Investor", "the Investor"]


@pytest.mark.ac("KG-AC-96")
def test_missing_or_malformed_aliases_degrade_to_empty_never_crash():
    text = "GLOBAL UNIVERSITY ENDOWMENT agrees."
    for payload_aliases in (None, "not-a-list", [], [""], [None, 123]):
        item = {"type": "Investor", "surface": "GLOBAL UNIVERSITY ENDOWMENT"}
        if payload_aliases is not None:
            item["aliases"] = payload_aliases
        client = _FakeClient({"entities": [item], "relations": [], "facts": []})
        cands = LlmGraphStrategy(llm_client=client).extract(
            [Chunk("c1", text)], ExtractionConfig(engine="llm", ontology_pack="investment_fibo"), PACK)
        assert cands[0].declared_aliases == [], payload_aliases


@pytest.mark.ac("KG-AC-96")
def test_aliases_are_deduplicated_and_order_preserved():
    text = "GLOBAL UNIVERSITY ENDOWMENT agrees."
    client = _FakeClient({
        "entities": [{"type": "Investor", "surface": "GLOBAL UNIVERSITY ENDOWMENT",
                      "aliases": ["Investor", "the Investor", "Investor", "  "]}],
        "relations": [], "facts": [],
    })
    cands = LlmGraphStrategy(llm_client=client).extract(
        [Chunk("c1", text)], ExtractionConfig(engine="llm", ontology_pack="investment_fibo"), PACK)
    assert cands[0].declared_aliases == ["Investor", "the Investor"]


# ---- persistence: aliases reach the kg_entities row ---------------------------------------------
@pytest.mark.ac("KG-AC-96")
def test_build_entity_records_emits_declared_aliases():
    c = Candidate(surface_form="GLOBAL UNIVERSITY ENDOWMENT", entity_type="Investor",
                  source_chunk_id="c1", layer="llm", span_start=0, span_end=27,
                  declared_aliases=["Investor"])
    rows = build_entity_records("f1", [c], "investment_fibo", "2.4")
    assert rows[0]["declared_aliases"] == ["Investor"]


@pytest.mark.ac("KG-AC-96")
def test_build_entity_records_defaults_to_an_empty_list():
    c = Candidate(surface_form="Acme", entity_type="Investor", source_chunk_id="c1",
                  layer="llm", span_start=0, span_end=4)
    assert build_entity_records("f1", [c], "investment_fibo", "2.4")[0]["declared_aliases"] == []


@pytest.mark.ac("KG-AC-96")
def test_span_overlap_merge_unions_declared_aliases_rather_than_dropping_them():
    # merge_candidates drops the lower-precedence candidate on span overlap. The alias binding is
    # ORTHOGONAL to which layer won -- both candidates describe the same span, so the same entity.
    # Dropping it would silently lose the binding whenever a rules layer outranks the LLM.
    llm = Candidate(surface_form="Acme Capital", entity_type="Investor", source_chunk_id="c1",
                    layer="llm", span_start=0, span_end=12, declared_aliases=["the Investor"])
    rules = Candidate(surface_form="Acme Capital", entity_type="Investor", source_chunk_id="c1",
                      layer="regex", span_start=0, span_end=12)
    merged = merge_candidates([llm, rules])
    assert len(merged) == 1
    assert merged[0].layer == "regex"          # precedence unchanged
    assert merged[0].declared_aliases == ["the Investor"]  # ...but the binding survived
