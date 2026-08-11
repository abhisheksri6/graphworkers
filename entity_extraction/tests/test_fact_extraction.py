"""P6 (spec v13, KG-AC-70): fact extraction, validation, and persistence. The tool schema carries a
`facts[]` array whose items are `{subject_id, property, value, evidence}`; `property` is enum-
constrained to the pack's declared `datatype_properties`; each fact is domain-validated (subject
type satisfies the property's declared domain, ancestors accepted) and evidence-grounded (reuses
KG-AC-64's verbatim, whitespace-normalised mechanism). Survivors persist into `kg_entities.attributes`
(existing jsonb column) as `{property, value, normalized_value, evidence, source_doc_id, page}`,
nested one row per subject entity. `value` is stored verbatim; `normalized_value` is computed per the
property's declared `range` kind (date/number/identifier/string), `None` when unparseable (counted).
Facts come from run 1 only (same posture as entities) — self-consistency repeats never re-collect
them. Intra-call duplicates sharing `(subject_id, property, normalized_value)` collapse to one;
a missing/malformed `facts` key degrades to entities+relations only, no crash, no dedicated counter
(matches the relations key's own uncounted degrade — see requirements.md's in-place correction of
this row, 2026-08-11)."""
import pytest

from core import (
    Fact, attach_facts_to_entity_records, build_entity_records, build_summary,
    normalize_fact_value,
)
from ontologies import DatatypeProperty, EntityType, Pack
from ontologies import Relation as PackRelation
from strategies.base import Chunk, ExtractionConfig, run_pipeline, validate_facts
from strategies.llm_graph import LlmGraphStrategy, build_graph_system_prompt, build_graph_tool_schema


def _agreement_pack():
    return Pack(
        name="x", version="1", description="",
        entity_types=[
            EntityType("Agreement", None, [], "", None),
            EntityType("Investor", None, [], "", None),
        ],
        relations=[PackRelation("investedBy", ["Agreement"], ["Investor"], "")],
        datatype_properties=[
            DatatypeProperty("effectiveDate", "Agreement", "date", "The effective date."),
            DatatypeProperty("agreementId", "Agreement", "identifier", "The agreement id."),
            DatatypeProperty("dealSize", "Agreement", "number", "The deal size."),
            DatatypeProperty("governingLaw", "Agreement", "string", "The governing law."),
        ],
    )


class _FakeLlmClient:
    def __init__(self, response):
        self._response = response
        self.usage = []
        self.resolved_model = "fake-model"

    def complete_tool(self, *, system_text, user_text, tool_name, tool_description, tool_schema):
        return self._response


# ---- normalize_fact_value per range kind ------------------------------------------------------
@pytest.mark.ac("KG-AC-70")
def test_normalize_date_iso8601():
    assert normalize_fact_value("15 January 2026", "date") == "2026-01-15"


@pytest.mark.ac("KG-AC-70")
def test_normalize_date_unparseable_is_none():
    assert normalize_fact_value("some time next quarter", "date") is None


@pytest.mark.ac("KG-AC-70")
def test_normalize_number_strips_currency_and_commas():
    assert normalize_fact_value("USD 1,250,000.50", "number") == "1250000.50"


@pytest.mark.ac("KG-AC-70")
def test_normalize_number_unparseable_is_none():
    assert normalize_fact_value("a large amount", "number") is None


@pytest.mark.ac("KG-AC-70")
def test_normalize_identifier_and_string_are_whitespace_trimmed():
    assert normalize_fact_value("  IMA-2025-018  ", "identifier") == "IMA-2025-018"
    assert normalize_fact_value(" laws of England ", "string") == "laws of England"


# ---- validate_facts: domain-validation + evidence-grounding -----------------------------------
@pytest.mark.ac("KG-AC-70")
def test_validate_facts_keeps_a_valid_fact():
    pack = _agreement_pack()
    chunk_text = "The Agreement is effective 15 January 2026."
    f = Fact(property="effectiveDate", value="15 January 2026", normalized_value="2026-01-15",
             evidence_text="The Agreement is effective 15 January 2026.", subject_type="Agreement",
             subject_surface="the Agreement", source_chunk_id="c1")
    kept, unmapped, ungrounded = validate_facts([f], pack, {"c1": chunk_text})
    assert kept == [f]
    assert unmapped == 0 and ungrounded == 0


@pytest.mark.ac("KG-AC-70")
def test_validate_facts_drops_and_counts_unknown_property():
    pack = _agreement_pack()
    f = Fact(property="notAPackProperty", value="x", normalized_value="x", evidence_text="x is x",
             subject_type="Agreement", subject_surface="the Agreement", source_chunk_id="c1")
    kept, unmapped, ungrounded = validate_facts([f], pack, {"c1": "x is x"})
    assert kept == []
    assert unmapped == 1 and ungrounded == 0


@pytest.mark.ac("KG-AC-70")
def test_validate_facts_drops_and_counts_invalid_domain():
    # "effectiveDate" is declared with domain Agreement -- a fact asserting it for an Investor
    # subject fails domain validation (KG-AC-70's "either gate" -> unmapped_property_count, unlike
    # relations' uncounted illegal-domain posture -- documented interpretation, see validate_facts).
    pack = _agreement_pack()
    f = Fact(property="effectiveDate", value="15 January 2026", normalized_value="2026-01-15",
             evidence_text="15 January 2026", subject_type="Investor",
             subject_surface="Jane Roe", source_chunk_id="c1")
    kept, unmapped, ungrounded = validate_facts([f], pack, {"c1": "15 January 2026"})
    assert kept == []
    assert unmapped == 1 and ungrounded == 0


@pytest.mark.ac("KG-AC-70")
def test_validate_facts_drops_and_counts_ungrounded_evidence():
    pack = _agreement_pack()
    f = Fact(property="agreementId", value="IMA-2025-018", normalized_value="IMA-2025-018",
             evidence_text="a sentence never in the chunk", subject_type="Agreement",
             subject_surface="the Agreement", source_chunk_id="c1")
    kept, unmapped, ungrounded = validate_facts([f], pack, {"c1": "The Agreement is IMA-2025-018."})
    assert kept == []
    assert unmapped == 0 and ungrounded == 1


@pytest.mark.ac("KG-AC-70")
def test_validate_facts_accepts_domain_via_ancestor():
    pack = Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("Agreement", None, [], "", None),
                     EntityType("SideAgreement", "Agreement", [], "", None)],
        relations=[],
        datatype_properties=[DatatypeProperty("effectiveDate", "Agreement", "date", "")],
    )
    f = Fact(property="effectiveDate", value="1 January 2026", normalized_value="2026-01-01",
             evidence_text="effective 1 January 2026", subject_type="SideAgreement",
             subject_surface="the SideAgreement", source_chunk_id="c1")
    kept, unmapped, ungrounded = validate_facts([f], pack, {"c1": "effective 1 January 2026"})
    assert len(kept) == 1
    assert unmapped == 0 and ungrounded == 0


# ---- attach_facts_to_entity_records: persistence into kg_entities.attributes ------------------
@pytest.mark.ac("KG-AC-70")
def test_attach_facts_nests_onto_matching_subject_row():
    rows = [{"source_chunk_id": "c1", "entity_type": "Agreement", "surface_form": "the Agreement"}]
    f = Fact(property="agreementId", value="IMA-2025-018", normalized_value="IMA-2025-018",
             evidence_text="IMA-2025-018", subject_type="Agreement",
             subject_surface="the Agreement", source_chunk_id="c1")
    attach_facts_to_entity_records(rows, [f], chunk_provenance={"c1": ("doc1.pdf", 3)})
    assert rows[0]["attributes"] == [{
        "property": "agreementId", "value": "IMA-2025-018", "normalized_value": "IMA-2025-018",
        "evidence": "IMA-2025-018", "source_doc_id": "doc1.pdf", "page": 3,
    }]


@pytest.mark.ac("KG-AC-70")
def test_attach_facts_gives_every_row_an_empty_list_when_no_facts_match():
    rows = [{"source_chunk_id": "c1", "entity_type": "Agreement", "surface_form": "the Agreement"}]
    attach_facts_to_entity_records(rows, [], chunk_provenance={})
    assert rows[0]["attributes"] == []


@pytest.mark.ac("KG-AC-70")
def test_attach_facts_excludes_facts_whose_subject_row_was_dropped():
    # subject "a ghost entity" was never written to entity_rows (e.g. dropped as unmapped/unlocatable)
    rows = [{"source_chunk_id": "c1", "entity_type": "Agreement", "surface_form": "the Agreement"}]
    f = Fact(property="agreementId", value="X", normalized_value="X", evidence_text="X",
             subject_type="Agreement", subject_surface="a ghost entity", source_chunk_id="c1")
    attach_facts_to_entity_records(rows, [f], chunk_provenance={})
    assert rows[0]["attributes"] == []


# ---- LlmGraphStrategy.extract(): subject_id resolution + intra-call dedup ---------------------
@pytest.mark.ac("KG-AC-70")
def test_extract_resolves_subject_id_and_computes_normalized_value():
    pack = _agreement_pack()
    chunk = Chunk(chunk_id="c1", text="The Agreement is effective 15 January 2026.")
    response = {
        "entities": [{"type": "Agreement", "surface": "The Agreement"}],
        "relations": [],
        "facts": [{"subject_id": 0, "property": "effectiveDate", "value": "15 January 2026",
                   "evidence": "The Agreement is effective 15 January 2026."}],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    strat.extract([chunk], ExtractionConfig(engine="llm"), pack)
    assert len(strat.facts) == 1
    fact = strat.facts[0]
    assert fact.property == "effectiveDate"
    assert fact.value == "15 January 2026"  # verbatim
    assert fact.normalized_value == "2026-01-15"
    assert fact.subject_type == "Agreement" and fact.subject_surface == "The Agreement"
    assert strat.unresolved_reference_count == 0


@pytest.mark.ac("KG-AC-70")
def test_extract_unresolved_subject_id_is_dropped_and_counted():
    pack = _agreement_pack()
    chunk = Chunk(chunk_id="c1", text="The Agreement is effective 15 January 2026.")
    response = {
        "entities": [{"type": "Agreement", "surface": "The Agreement"}],
        "relations": [],
        "facts": [{"subject_id": 99, "property": "effectiveDate", "value": "15 January 2026",
                   "evidence": "The Agreement is effective 15 January 2026."}],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    strat.extract([chunk], ExtractionConfig(engine="llm"), pack)
    assert strat.facts == []
    assert strat.unresolved_reference_count == 1


@pytest.mark.ac("KG-AC-70")
def test_extract_missing_facts_key_degrades_without_crash():
    pack = _agreement_pack()
    chunk = Chunk(chunk_id="c1", text="The Agreement is effective 15 January 2026.")
    response = {"entities": [{"type": "Agreement", "surface": "The Agreement"}], "relations": []}
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    entities = strat.extract([chunk], ExtractionConfig(engine="llm"), pack)
    assert len(entities) == 1  # entities+relations unaffected (KG-AC-43's posture, extended to facts)
    assert strat.facts == []


@pytest.mark.ac("KG-AC-70")
def test_extract_malformed_facts_key_degrades_without_crash():
    pack = _agreement_pack()
    chunk = Chunk(chunk_id="c1", text="The Agreement is effective 15 January 2026.")
    response = {"entities": [{"type": "Agreement", "surface": "The Agreement"}], "relations": [],
                "facts": "not-a-list"}
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    strat.extract([chunk], ExtractionConfig(engine="llm"), pack)
    assert strat.facts == []


@pytest.mark.ac("KG-AC-70")
def test_extract_collapses_identical_intra_call_duplicates():
    pack = _agreement_pack()
    chunk = Chunk(chunk_id="c1", text="The Agreement is effective 15 January 2026, effective 15 January 2026.")
    response = {
        "entities": [{"type": "Agreement", "surface": "The Agreement"}],
        "relations": [],
        "facts": [
            {"subject_id": 0, "property": "effectiveDate", "value": "15 January 2026",
             "evidence": "effective 15 January 2026"},
            {"subject_id": 0, "property": "effectiveDate", "value": "15 January 2026",
             "evidence": "effective 15 January 2026"},
        ],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    strat.extract([chunk], ExtractionConfig(engine="llm"), pack)
    assert len(strat.facts) == 1  # identical (subject_id, property, normalized_value) -> collapsed


@pytest.mark.ac("KG-AC-70")
def test_extract_keeps_both_facts_when_normalized_values_differ():
    pack = _agreement_pack()
    chunk = Chunk(chunk_id="c1", text="The Agreement was effective 15 January 2026. "
                                      "Later amended to 1 February 2026.")
    response = {
        "entities": [{"type": "Agreement", "surface": "The Agreement"}],
        "relations": [],
        "facts": [
            {"subject_id": 0, "property": "effectiveDate", "value": "15 January 2026",
             "evidence": "Effective 15 January 2026."},
            {"subject_id": 0, "property": "effectiveDate", "value": "1 February 2026",
             "evidence": "Later amended to 1 February 2026."},
        ],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    strat.extract([chunk], ExtractionConfig(engine="llm"), pack)
    assert len(strat.facts) == 2  # differing normalized_value -> both retained (cross-mention
    # conflict resolution is KG-AC-78/P11's job, not this loop's)


# ---- tool schema / prompt: facts surface only when the pack declares properties ---------------
@pytest.mark.ac("KG-AC-70")
def test_tool_schema_carries_facts_when_pack_declares_properties():
    schema = build_graph_tool_schema(_agreement_pack())
    assert "facts" in schema["properties"]
    assert "facts" in schema["required"]
    fact_item = schema["properties"]["facts"]["items"]
    assert set(fact_item["required"]) == {"subject_id", "property", "value", "evidence"}
    assert fact_item["properties"]["property"]["enum"] == [
        "agreementId", "dealSize", "effectiveDate", "governingLaw",
    ]


@pytest.mark.ac("KG-AC-69")
def test_tool_schema_omits_facts_for_a_pack_with_no_datatype_properties():
    pack = Pack(name="x", version="1", description="",
               entity_types=[EntityType("Organization", None, [], "", None)], relations=[])
    schema = build_graph_tool_schema(pack)
    assert "facts" not in schema["properties"]
    assert "facts" not in schema["required"]


@pytest.mark.ac("KG-AC-70")
def test_system_prompt_mentions_facts_only_when_pack_declares_properties():
    with_props = build_graph_system_prompt(_agreement_pack())
    assert "effectiveDate" in with_props and "subject_id" in with_props
    no_props_pack = Pack(name="x", version="1", description="",
                         entity_types=[EntityType("Organization", None, [], "", None)], relations=[])
    without_props = build_graph_system_prompt(no_props_pack)
    assert "subject_id" not in without_props


# ---- end-to-end run_pipeline: extraction -> validation -> persistence -> summary scalars ------
@pytest.mark.ac("KG-AC-70")
def test_run_pipeline_persists_facts_and_reports_summary_scalars():
    pack = _agreement_pack()
    chunk = Chunk(chunk_id="c1", text="The Agreement is effective 15 January 2026.",
                  doc_id="doc1.pdf", page=1)
    response = {
        "entities": [{"type": "Agreement", "surface": "The Agreement"}],
        "relations": [],
        "facts": [{"subject_id": 0, "property": "effectiveDate", "value": "15 January 2026",
                   "evidence": "The Agreement is effective 15 January 2026."}],
    }
    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [chunk], ExtractionConfig(engine="llm"), pack, folder_id="f1",
        llm_client=_FakeLlmClient(response),
    )
    assert len(ent_rows) == 1
    assert ent_rows[0]["attributes"] == [{
        "property": "effectiveDate", "value": "15 January 2026", "normalized_value": "2026-01-15",
        "evidence": "The Agreement is effective 15 January 2026.",
        "source_doc_id": "doc1.pdf", "page": 1,
    }]
    assert summary["unmapped_property_count"] == 0
    assert summary["ungrounded_fact_count"] == 0


@pytest.mark.ac("KG-AC-70")
def test_run_pipeline_reports_unmapped_property_and_ungrounded_fact_counts():
    pack = _agreement_pack()
    chunk = Chunk(chunk_id="c1", text="The Agreement is effective 15 January 2026.")
    response = {
        "entities": [{"type": "Agreement", "surface": "The Agreement"}],
        "relations": [],
        "facts": [
            # unknown property -> unmapped_property_count
            {"subject_id": 0, "property": "notDeclared", "value": "x", "evidence": "The Agreement is effective 15 January 2026."},
            # ungrounded evidence -> ungrounded_fact_count
            {"subject_id": 0, "property": "agreementId", "value": "IMA-2025-018", "evidence": "not in the chunk at all"},
        ],
    }
    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [chunk], ExtractionConfig(engine="llm"), pack, folder_id="f1",
        llm_client=_FakeLlmClient(response),
    )
    assert ent_rows[0]["attributes"] == []
    assert summary["unmapped_property_count"] == 1
    assert summary["ungrounded_fact_count"] == 1


@pytest.mark.ac("KG-AC-70")
def test_run_pipeline_facts_come_from_run_1_only_not_self_consistency_repeats():
    pack = _agreement_pack()
    chunk = Chunk(chunk_id="c1", text="The Agreement is effective 15 January 2026.")
    response = {
        "entities": [{"type": "Agreement", "surface": "The Agreement"}],
        "relations": [{"type": "investedBy", "src_id": 0, "dst_id": 0, "evidence": "The Agreement is effective 15 January 2026."}],
        "facts": [{"subject_id": 0, "property": "effectiveDate", "value": "15 January 2026",
                   "evidence": "The Agreement is effective 15 January 2026."}],
    }
    # investedBy's domain/range don't match (Agreement->Investor declared, here both are Agreement) --
    # irrelevant to this test, which only cares that facts are captured once regardless of k.
    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [chunk], ExtractionConfig(engine="llm", relation_self_consistency_k=3), pack, folder_id="f1",
        llm_client=_FakeLlmClient(response),
    )
    assert len(ent_rows[0]["attributes"]) == 1  # not 3x -- facts never re-collected on repeats
