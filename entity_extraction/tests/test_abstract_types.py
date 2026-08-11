"""P5 (spec v13, KG-AC-72): pack-declared abstract entity types — a concept the document never
writes as a literal string (an overall relationship, a terms grouping). An abstract type is
accepted without a locatable span; a non-abstract type whose span cannot be found is dropped and
counted (`unlocatable_entity_count`) — a NEW v13 behaviour (pre-v13, such an entity was silently
written with a null span, never dropped; the frozen AC's row originally mis-described this as
"today's behaviour" — corrected in the same change as this implementation, see tasks.md).
"""
import pytest

from core import Candidate, build_entity_records, build_summary, compute_entity_uid
from ontologies import EntityType, Pack, load_pack
from ontologies import Relation as PackRelation
from strategies.llm_graph import LlmGraphStrategy
from strategies.base import Chunk, ExtractionConfig


def _abstract_pack():
    return Pack(
        name="x", version="1", description="",
        entity_types=[
            EntityType("InvestmentRelationship", None, [], "", None, abstract=True),
            EntityType("Investor", None, [], "", None),
        ],
        relations=[],
    )


class _FakeLlmClient:
    def __init__(self, response):
        self._response = response

    def complete_tool(self, *, system_text, user_text, tool_name, tool_description, tool_schema):
        return self._response


# ---- Candidate.is_abstract schema field ------------------------------------------------------
@pytest.mark.ac("KG-AC-72")
def test_candidate_is_abstract_defaults_false():
    c = Candidate(surface_form="Acme Corp", entity_type="Organization", source_chunk_id="c1", layer="llm")
    assert c.is_abstract is False


@pytest.mark.ac("KG-AC-72")
def test_build_entity_records_writes_is_abstract():
    cand = Candidate(surface_form="X - Y Investment Relationship", entity_type="InvestmentRelationship",
                     source_chunk_id="c1", layer="llm", is_abstract=True)
    rows = build_entity_records("f1", [cand], "x", "1.0")
    assert rows[0]["is_abstract"] is True


@pytest.mark.ac("KG-AC-72")
def test_compute_entity_uid_stays_deterministic_with_null_span():
    # verified at clarify: no change needed here -- COALESCE already handles it. Locked as a
    # regression guard.
    uid1 = compute_entity_uid("f1", "c1", "InvestmentRelationship", "X - Y Investment Relationship", None, 0)
    uid2 = compute_entity_uid("f1", "c1", "InvestmentRelationship", "X - Y Investment Relationship", None, 0)
    assert uid1 == uid2
    assert isinstance(uid1, str) and len(uid1) == 64  # sha256 hex digest


# ---- LlmGraphStrategy: abstract types accepted span-less ------------------------------------
@pytest.mark.ac("KG-AC-72")
def test_abstract_type_accepted_without_locatable_span():
    pack = _abstract_pack()
    response = {
        "entities": [
            {"type": "InvestmentRelationship", "surface": "XYZ - Acme Investment Relationship", "confidence": 0.8},
        ],
        "relations": [],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    entities = strat.extract([Chunk("c1", "XYZ Insurance appoints Acme as manager.")],
                             ExtractionConfig(engine="llm"), pack)
    assert len(entities) == 1
    e = entities[0]
    assert e.is_abstract is True
    assert e.span_start is None and e.span_end is None
    assert e.surface_form == "XYZ - Acme Investment Relationship"  # the synthesised name, verbatim
    assert strat.unlocatable_entity_count == 0  # never even attempted to locate -- not a drop case


@pytest.mark.ac("KG-AC-72")
def test_abstract_type_may_serve_as_relation_endpoint():
    pack = Pack(
        name="x", version="1", description="",
        entity_types=[
            EntityType("InvestmentRelationship", None, [], "", None, abstract=True),
            EntityType("Investor", None, [], "", None),
        ],
        relations=[PackRelation("hasInvestor", ["InvestmentRelationship"], ["Investor"], "")],
    )
    response = {
        "entities": [
            {"type": "InvestmentRelationship", "surface": "XYZ Investment Relationship", "confidence": 0.8},
            {"type": "Investor", "surface": "XYZ Insurance", "confidence": 0.9},
        ],
        "relations": [
            {"type": "hasInvestor", "src_id": 0, "dst_id": 1, "evidence": "XYZ Insurance is the investor."},
        ],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    strat.extract([Chunk("c1", "XYZ Insurance is the investor.")], ExtractionConfig(engine="llm"), pack)
    assert len(strat.relations) == 1
    assert strat.relations[0].src_surface == "XYZ Investment Relationship"
    assert strat.relations[0].src_type == "InvestmentRelationship"


# ---- non-abstract unlocatable: dropped + counted (the NEW v13 behaviour) --------------------
@pytest.mark.ac("KG-AC-72")
def test_non_abstract_unlocatable_span_is_dropped_and_counted():
    pack = _abstract_pack()
    response = {
        # "Investor" is NOT abstract, and its surface never appears in the chunk text at all.
        "entities": [{"type": "Investor", "surface": "Ghost Investor", "confidence": 0.7}],
        "relations": [],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    entities = strat.extract([Chunk("c1", "Nothing about any investor here.")],
                             ExtractionConfig(engine="llm"), pack)
    assert entities == []  # dropped, NOT written with a null span (the pre-v13 behaviour)
    assert strat.unlocatable_entity_count == 1


@pytest.mark.ac("KG-AC-72")
def test_non_abstract_locatable_span_unaffected():
    pack = _abstract_pack()
    response = {"entities": [{"type": "Investor", "surface": "Acme Corp", "confidence": 0.9}], "relations": []}
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    entities = strat.extract([Chunk("c1", "Acme Corp is the investor.")], ExtractionConfig(engine="llm"), pack)
    assert len(entities) == 1
    assert entities[0].is_abstract is False
    assert entities[0].span_start == 0
    assert strat.unlocatable_entity_count == 0


@pytest.mark.ac("KG-AC-72")
def test_id_referencing_a_dropped_unlocatable_entity_is_unresolved():
    # the unlocatable entity is dropped BEFORE by_raw_index gets an entry for it -- a relation
    # referencing that position is therefore correctly unresolved (KG-AC-71), not a crash.
    pack = Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("Investor", None, [], "", None), EntityType("Fund", None, [], "", None)],
        relations=[PackRelation("investsInFund", ["Investor"], ["Fund"], "")],
    )
    response = {
        "entities": [
            {"type": "Investor", "surface": "Ghost Investor", "confidence": 0.7},  # unlocatable, dropped
            {"type": "Fund", "surface": "Acme Fund", "confidence": 0.9},
        ],
        "relations": [
            {"type": "investsInFund", "src_id": 0, "dst_id": 1, "evidence": "text"},
        ],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    strat.extract([Chunk("c1", "Acme Fund exists.")], ExtractionConfig(engine="llm"), pack)
    assert strat.relations == []
    assert strat.unlocatable_entity_count == 1
    assert strat.unresolved_reference_count == 1


# ---- run_pipeline end-to-end -----------------------------------------------------------------
@pytest.mark.ac("KG-AC-72")
def test_run_pipeline_writes_abstract_row_and_reports_summary_scalars():
    pack = _abstract_pack()
    response = {
        "entities": [
            {"type": "InvestmentRelationship", "surface": "XYZ Investment Relationship", "confidence": 0.8},
        ],
        "relations": [],
    }
    from strategies.base import run_pipeline
    cfg = ExtractionConfig(engine="llm", ontology_pack="x")
    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [Chunk("c1", "XYZ Insurance appoints Acme.")], cfg, pack, folder_id="f1",
        llm_client=_FakeLlmClient(response),
    )
    assert len(ent_rows) == 1
    assert ent_rows[0]["is_abstract"] is True
    assert ent_rows[0]["span_start"] is None
    assert summary["unlocatable_entity_count"] == 0


@pytest.mark.ac("KG-AC-72")
def test_run_pipeline_reports_unlocatable_entity_count_in_summary():
    pack = _abstract_pack()
    response = {"entities": [{"type": "Investor", "surface": "Ghost Investor", "confidence": 0.7}],
               "relations": []}
    from strategies.base import run_pipeline
    cfg = ExtractionConfig(engine="llm", ontology_pack="x")
    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [Chunk("c1", "Nothing here.")], cfg, pack, folder_id="f1", llm_client=_FakeLlmClient(response),
    )
    assert ent_rows == []
    assert summary["unlocatable_entity_count"] == 1


# ---- investment_fibo v2.0 integration ---------------------------------------------------------
@pytest.mark.ac("KG-AC-72")
def test_investment_fibo_abstract_types_load_correctly():
    pack = load_pack("investment_fibo")
    assert pack.entity_types["InvestmentRelationship"].abstract is True
    assert pack.entity_types["Commitment"].abstract is True
    assert pack.entity_types["CommercialTerms"].abstract is True
    assert pack.entity_types["Agreement"].abstract is False  # locatable (carries a reference number)
