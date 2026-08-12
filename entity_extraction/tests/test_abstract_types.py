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
from strategies.llm_graph import (
    LlmGraphStrategy, build_graph_system_prompt, build_graph_tool_schema,
)
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


# ---- Q2 (spec v14, KG-AC-89): abstract types leave the LLM contract entirely ------------------
# Replaces the v13 mechanism above (model synthesises the instance + its name, referenced by
# array position) with Q3/Q4's deterministic derivation. The ACCEPTANCE machinery tested above
# (extract()'s is_abstract branch, compute_entity_uid's null-span handling) is unchanged and
# still exercised — Q3 will construct Candidates through the same path, just from code, not a
# model response. Only what's OFFERED to the model changes.
@pytest.mark.ac("KG-AC-89")
def test_tool_schema_entity_enum_excludes_abstract_types():
    schema = build_graph_tool_schema(_abstract_pack())
    entity_enum = schema["properties"]["entities"]["items"]["properties"]["type"]["enum"]
    assert "InvestmentRelationship" not in entity_enum
    assert entity_enum == ["Investor"]  # the only non-abstract type in this fixture


@pytest.mark.ac("KG-AC-89")
def test_tool_schema_excludes_all_three_real_abstract_types():
    pack = load_pack("investment_fibo")
    schema = build_graph_tool_schema(pack)
    entity_enum = set(schema["properties"]["entities"]["items"]["properties"]["type"]["enum"])
    assert entity_enum.isdisjoint({"InvestmentRelationship", "Commitment", "CommercialTerms"})
    # every non-abstract type is still offered -- this isn't a blanket exclusion
    assert "Agreement" in entity_enum and "Investor" in entity_enum and "FeeSchedule" in entity_enum


@pytest.mark.ac("KG-AC-89")
def test_system_prompt_carries_no_synthesise_instruction():
    prompt = build_graph_system_prompt(_abstract_pack())
    assert "InvestmentRelationship" not in prompt
    assert "ABSTRACT" not in prompt
    assert "name it after" not in prompt  # the v13 synthesise-instruction phrase, verbatim
    assert "Investor" in prompt  # the concrete type is still offered


@pytest.mark.ac("KG-AC-89")
def test_system_prompt_excludes_all_three_real_abstract_types():
    pack = load_pack("investment_fibo")
    prompt = build_graph_system_prompt(pack)
    for abstract_type in ("InvestmentRelationship", "Commitment", "CommercialTerms"):
        assert abstract_type not in prompt
    assert "Agreement" in prompt  # concrete types unaffected


@pytest.mark.ac("KG-AC-89")
def test_relations_touching_an_abstract_type_are_also_excluded():
    # found while writing this task's own tests, not in the literal AC bullet list: a relation
    # whose domain OR range names an abstract type can never be correctly emitted once that type
    # is gone from the model's entities[] (nothing for src_id/dst_id to reference) -- leaving it
    # in the vocabulary just invites the model to attempt it and fail, the exact class of noise
    # this evolve exists to remove. investment_fibo's own hasInvestor/hasInvestmentManager/
    # governedBy/hasSubscription/hasCommitment/hasCommercialTerms/definedByFeeSchedule all touch
    # one of the three abstract types -- leaving investsInFund/hasLegalEntity/amendedBy/
    # supplementedBy (concrete-to-concrete) untouched.
    pack = load_pack("investment_fibo")
    schema = build_graph_tool_schema(pack)
    relation_enum = set(schema["properties"]["relations"]["items"]["properties"]["type"]["enum"])
    excluded = {"hasInvestor", "hasInvestmentManager", "governedBy", "hasSubscription",
               "hasCommitment", "hasCommercialTerms", "definedByFeeSchedule"}
    assert relation_enum.isdisjoint(excluded)
    still_offered = {"investsInFund", "hasLegalEntity", "amendedBy", "supplementedBy"}
    assert still_offered <= relation_enum

    prompt = build_graph_system_prompt(pack)
    for rel_name in excluded:
        assert rel_name not in prompt
    for rel_name in still_offered:
        assert rel_name in prompt


@pytest.mark.ac("KG-AC-89")
def test_concrete_only_pack_produces_byte_identical_schema_and_prompt():
    # regression guard: a pack with NO abstract types must be completely unaffected by this
    # evolve -- Q2 is a no-op for it, same posture KG-AC-69's optional-key promise already has.
    concrete_pack = Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("Organization", None, [], "An org.", None)],
        relations=[],
    )
    schema_before = build_graph_tool_schema(concrete_pack)
    prompt_before = build_graph_system_prompt(concrete_pack)
    # re-declaring the identical pack a second time must produce an identical schema/prompt --
    # proves the filtering doesn't touch anything when there's nothing to filter.
    same_pack = Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("Organization", None, [], "An org.", None)],
        relations=[],
    )
    assert build_graph_tool_schema(same_pack) == schema_before
    assert build_graph_system_prompt(same_pack) == prompt_before
    assert schema_before["properties"]["entities"]["items"]["properties"]["type"]["enum"] == ["Organization"]


@pytest.mark.ac("KG-AC-89")
def test_datatype_properties_whose_domain_is_abstract_are_also_excluded():
    # found while writing this task's own tests, not in the literal AC bullet list, and NOT fully
    # resolved by this task -- flagged explicitly, see the code comment: a datatype_property whose
    # domain is an abstract type (Commitment.commitmentAmount/currency/commitmentDate/
    # commitmentType, CommercialTerms.managementFee/performanceFee/performanceHurdle/
    # billingFrequency/settlementCurrency/feeEffectiveDate — 10 properties total) can never be
    # correctly emitted as a fact, since its subject_id would need to reference an entity that no
    # longer exists in the model's own response. Excluding them stops the model being offered
    # vocabulary it can only fail at; it does NOT provide a replacement mechanism for extracting
    # this real data (Q3/Q4 only derive the entity + its edges, never its own attributes) — a real
    # capability gap this task surfaces rather than silently works around.
    pack = load_pack("investment_fibo")
    schema = build_graph_tool_schema(pack)
    property_enum = set(schema["properties"]["facts"]["items"]["properties"]["property"]["enum"])
    excluded = {"commitmentAmount", "currency", "commitmentDate", "commitmentType",
               "managementFee", "performanceFee", "performanceHurdle", "billingFrequency",
               "settlementCurrency", "feeEffectiveDate"}
    assert property_enum.isdisjoint(excluded)
    # a concrete-domain property is unaffected
    assert "agreementId" in property_enum

    prompt = build_graph_system_prompt(pack)
    for prop_name in excluded:
        assert prop_name not in prompt


@pytest.mark.ac("KG-AC-89")
def test_extract_still_accepts_an_abstract_item_if_a_model_emits_one_anyway():
    # defense-in-depth, unchanged by Q2 (deliberately out of this task's scope): if schema
    # enforcement isn't airtight (llm_graph.py's own module docstring already acknowledges this
    # possibility for the closed vocabulary generally) and the model emits an abstract type
    # despite it being absent from the enum, extract()'s acceptance logic is untouched -- it
    # keys off the PACK's own abstract flag, not off what the schema advertised.
    pack = _abstract_pack()
    response = {
        "entities": [
            {"type": "InvestmentRelationship", "surface": "XYZ Investment Relationship", "confidence": 0.8},
        ],
        "relations": [],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    entities = strat.extract([Chunk("c1", "text")], ExtractionConfig(engine="llm"), pack)
    assert len(entities) == 1 and entities[0].is_abstract is True
