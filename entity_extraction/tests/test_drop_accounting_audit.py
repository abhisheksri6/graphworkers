"""P8 (spec v13, KG-AC-74): drop accounting — a counter behind every discard path.

Every counter this row names (`unmapped_property_count`, `unresolved_reference_count`,
`unlocatable_entity_count`, `ungrounded_fact_count`) plus P7's `guardrails_blocked_facts` was
already wired end-to-end by the task that produced it (P4/P5/P6/P7 each self-wired their own
counter into `build_summary` -> `callback._STATE_SCALARS` -> `capability_schema.output_fields`
in the same task, not deferred to this one — see each task's Done writeup). P8's own job is
therefore the NEGATIVE proof the task's verify text asks for: **one** combined run exercising
FIVE distinct discard reasons at once, asserting each counter reflects exactly its own drops with
no cross-contamination — the class of defect ("a drop happened and nothing counted it, or the
wrong thing counted it") this whole workstream exists to close. Individual counters already have
their own isolated-scenario tests (test_id_binding.py, test_abstract_types.py,
test_fact_extraction.py, test_guardrails_facts.py); this file is the cross-cutting audit, not a
duplicate of those.

Two drop paths are DELIBERATELY uncounted, matching a pre-v13, pre-existing precedent — not a gap
this task introduces or must close (audited here, not silently skipped):
- A relation/fact item missing a mandatory field (`type`/`property`/`value`/`evidence`) is dropped
  uncounted at parse time in every LLM strategy (`llm_graph.py`, `llm_classify.py`,
  `llm_entity_scoped.py`) — mirrors KG-AC-46's own established "evidence mandatory, dropped
  uncounted" posture, unchanged since v5.
- A relation/fact whose SUBJECT/endpoint was itself dropped upstream (unmapped type, unlocatable
  span, guardrails-blocked, ...) has no matching row to attach to and is silently excluded by
  `build_edge_records`/`attach_facts_to_entity_records` — mirrors `build_edge_records`'s own
  pre-v13 dangling-endpoint-drop precedent (uncounted because the subject's own drop is what got
  counted, not a second reason for the same discard).
"""
import pytest

from ontologies import DatatypeProperty, EntityType, Pack
from ontologies import Relation as PackRelation
from strategies.base import Chunk, ExtractionConfig, run_pipeline


def _pack():
    return Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("Agreement", None, [], "", None),
                     EntityType("Investor", None, [], "", None)],
        relations=[PackRelation("investedBy", ["Agreement"], ["Investor"], "")],
        datatype_properties=[
            DatatypeProperty("agreementId", "Agreement", "identifier", ""),
            DatatypeProperty("effectiveDate", "Agreement", "date", ""),
        ],
    )


class _FakeLlmClient:
    resolved_model = "fake-model"
    usage = []

    def complete_tool(self, **_kwargs):
        return {
            "entities": [
                {"type": "Agreement", "surface": "The Agreement"},          # index 0 -- located
                {"type": "Investor", "surface": "Global Fund II"},          # index 1 -- NOT in the
                                                                            # chunk text -> unlocatable
            ],
            "relations": [
                # dst_id=99 doesn't exist in this response's own entities[] -> unresolved
                {"type": "investedBy", "src_id": 0, "dst_id": 99,
                 "evidence": "The Agreement is entered."},
            ],
            "facts": [
                # subject_id=99 doesn't exist -> unresolved (SAME counter relations use)
                {"subject_id": 99, "property": "agreementId", "value": "Z", "evidence": "Z"},
                # subject_id=0 resolves fine; property is not in the pack's vocabulary -> unmapped
                {"subject_id": 0, "property": "notDeclared", "value": "Y",
                 "evidence": "The Agreement is entered."},
                # subject_id=0 resolves fine; property is declared, but evidence isn't verbatim
                # in the chunk -> ungrounded
                {"subject_id": 0, "property": "effectiveDate", "value": "15 January 2026",
                 "evidence": "not in the chunk anywhere"},
                # subject_id=0 resolves fine, property declared, evidence grounded -- would SURVIVE
                # validate_facts entirely, except guardrails blocks it below
                {"subject_id": 0, "property": "agreementId", "value": "IMA-2025-018",
                 "evidence": "The Agreement is entered."},
            ],
        }


_CHUNK = Chunk("c1", "The Agreement is entered.")


def _guardrails_screen(items):
    # items = kept entity candidates (1: Agreement -- Investor already dropped as unlocatable
    # before guardrails ever runs) followed by raw facts (3: notDeclared, ungrounded-date,
    # agreementId -- the subject_id=99 fact never became a Fact object at all). Block ONLY the
    # 4th item (the otherwise-valid agreementId fact) -- proves guardrails_blocked_facts is
    # independent of the other four counters.
    assert len(items) == 4
    return [True, True, True, False]


@pytest.mark.ac("KG-AC-74")
def test_five_simultaneous_discard_reasons_each_count_exactly_once_no_cross_contamination():
    pack = _pack()
    ent_rows, edge_rows, summary, usage, guardrails_blocked = run_pipeline(
        [_CHUNK], ExtractionConfig(engine="llm", ontology_pack="x"), pack, folder_id="f",
        llm_client=_FakeLlmClient(), guardrails_screen=_guardrails_screen,
    )
    # KG-AC-74's four named counters
    assert summary["unlocatable_entity_count"] == 1       # Investor, span not found
    assert summary["unresolved_reference_count"] == 2      # 1 relation (dst_id=99) + 1 fact (subject_id=99)
    assert summary["unmapped_property_count"] == 1         # notDeclared
    assert summary["ungrounded_fact_count"] == 1            # effectiveDate, ungrounded evidence
    # P7's counter, self-wired same-task, reaches the state plane alongside KG-AC-74's own four
    assert summary["guardrails_blocked_facts"] == 1         # agreementId, blocked
    # the pre-existing entity-side guardrails counter is unaffected by any of the above
    assert guardrails_blocked == 0

    # and the WRITTEN graph reflects every drop -- nothing silently kept either
    assert len(ent_rows) == 1                    # only Agreement survives (Investor unlocatable)
    assert ent_rows[0]["attributes"] == []        # every one of the 4 facts was dropped somewhere
    assert len(edge_rows) == 0                    # the one relation was unresolved


@pytest.mark.ac("KG-AC-74")
def test_every_named_counter_is_always_present_in_the_capability_schema():
    from capability_schema import CAPABILITY_SCHEMA
    out = CAPABILITY_SCHEMA["output_fields"]
    for field in ("unmapped_property_count", "unresolved_reference_count",
                 "unlocatable_entity_count", "ungrounded_fact_count", "guardrails_blocked_facts"):
        assert out[field]["always_present"] is True, field
