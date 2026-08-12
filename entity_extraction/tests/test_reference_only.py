"""R5 (spec v15, KG-AC-94): the `reference_only` marker — an entity the document NAMED with an
identifier but never described (`Fee Schedule FS-2025-031` cited in a definitions clause whose
actual terms live in another document) is WRITTEN and marked, never dropped.

Written because this ontology is explicitly multi-document: such a node is the join point
`entity_canonicalization` (P10–P13) merges the real document into when it arrives, and dropping it
also discards a source-stated `definedByFeeSchedule`. Marked because an unmarked stub is worse than
none — nothing would then distinguish "a Fee Schedule we know exists" from "one we have read".

Detection reuses the pack's OWN `range` vocabulary (KG-AC-69): all facts `identifier`-ranged (or
none) => reference-only. No new pack declaration, no heuristic.
"""
import pytest

from core import derive_abstract_entities, mark_reference_only
from ontologies import load_pack


def _pack():
    return load_pack("investment_fibo")


def _row(entity_type, surface, attributes=None, extractor="llm"):
    return {"entity_uid": f"uid-{surface}", "entity_type": entity_type, "surface_form": surface,
            "source_chunk_id": "c1", "source_doc_id": "d.pdf", "page": 1, "span_start": 0,
            "span_end": len(surface), "extractor": extractor, "attributes": attributes or []}


def _attr(prop, value="v"):
    return {"property": prop, "value": value, "normalized_value": value,
            "evidence": "e", "source_doc_id": "d.pdf", "page": 1}


# ---- the core rule ------------------------------------------------------------------------------
@pytest.mark.ac("KG-AC-94")
def test_only_identifier_facts_is_reference_only():
    rows = [_row("FeeSchedule", "FS-2025-031", [_attr("feeScheduleId", "FS-2025-031")])]
    mark_reference_only(rows, _pack())
    assert rows[0]["reference_only"] is True


@pytest.mark.ac("KG-AC-94")
def test_no_facts_at_all_is_reference_only():
    rows = [_row("SideLetter", "SL-2025-006")]
    mark_reference_only(rows, _pack())
    assert rows[0]["reference_only"] is True


@pytest.mark.ac("KG-AC-94")
def test_a_single_descriptive_fact_makes_it_described():
    # subscriptionStatus is `string`-ranged -- one such fact means the document actually said
    # something ABOUT the entity, not merely that it exists.
    rows = [_row("Subscription", "SUB-2025-041",
                [_attr("subscriptionId", "SUB-2025-041"), _attr("subscriptionStatus", "Accepted")])]
    mark_reference_only(rows, _pack())
    assert rows[0]["reference_only"] is False


@pytest.mark.ac("KG-AC-94")
def test_date_and_number_ranged_facts_also_count_as_described():
    for prop in ("effectiveDate", "commitmentAmount"):
        rows = [_row("Agreement", "the Agreement", [_attr("agreementId", "X"), _attr(prop)])]
        mark_reference_only(rows, _pack())
        assert rows[0]["reference_only"] is False, prop


# ---- clarify F2: derived entities are excluded ---------------------------------------------------
@pytest.mark.ac("KG-AC-94")
def test_derived_entities_are_never_marked_reference_only():
    # a reified_relation hub declares ZERO own attributes BY DEFINITION -- without this exclusion
    # the rule would flag every derived hub as an unread stub, which is a different fact about a
    # different thing (they are already marked by extractor='derived'/is_abstract).
    rows = [_row("InvestmentRelationship", "IMA-2025-018", extractor="derived")]
    mark_reference_only(rows, _pack())
    assert rows[0]["reference_only"] is False


@pytest.mark.ac("KG-AC-94")
def test_derived_hub_from_the_real_pass_is_not_reference_only():
    # the same guarantee, through the ACTUAL derivation pass rather than a hand-built row.
    rows = [_row("Agreement", "the Agreement", [_attr("agreementId", "IMA-2025-018")]),
            _row("Investor", "XYZ INSURANCE GROUP PLC")]
    derived, _, _ = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    mark_reference_only(rows + derived, _pack())
    hub = next(r for r in derived if r["entity_type"] == "InvestmentRelationship")
    assert hub["reference_only"] is False


# ---- ordering: after re-parenting (the task's own verify step) -----------------------------------
@pytest.mark.ac("KG-AC-94")
def test_anchor_stays_described_after_its_bundle_attributes_move_off_it():
    # THE ordering trap: managementFee (domain CommercialTerms) is offered under Agreement, then
    # re-parented off it by derivation. If reference_only were computed on the post-re-parent row
    # WITHOUT its own remaining descriptive facts, the Agreement could flip to a stub. It keeps
    # governingLaw, so it must stay described.
    rows = [_row("Agreement", "the Agreement", [
        _attr("agreementId", "IMA-2025-018"),
        _attr("managementFee", "1.5%"),          # moves to CommercialTerms
        _attr("governingLaw", "England and Wales"),  # stays
    ])]
    derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    mark_reference_only(rows, _pack())
    assert [a["property"] for a in rows[0]["attributes"]] == ["agreementId", "governingLaw"]
    assert rows[0]["reference_only"] is False


@pytest.mark.ac("KG-AC-94")
def test_anchor_left_with_only_its_identifier_becomes_reference_only():
    # the honest converse: if EVERY descriptive fact re-parented away, the anchor really is just a
    # named reference in this document, and saying so is correct rather than a bug.
    rows = [_row("Subscription", "SUB-2025-041", [
        _attr("subscriptionId", "SUB-2025-041"),
        _attr("commitmentAmount", "25000000"),  # domain Commitment -- moves off
    ])]
    derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    mark_reference_only(rows, _pack())
    assert [a["property"] for a in rows[0]["attributes"]] == ["subscriptionId"]
    assert rows[0]["reference_only"] is True


# ---- end-to-end through run_pipeline -------------------------------------------------------------
@pytest.mark.ac("KG-AC-94")
def test_reference_only_reaches_entity_rows_end_to_end():
    from strategies.base import Chunk, ExtractionConfig, run_pipeline

    class _Llm:
        resolved_model = "m"
        usage: list = []

        def complete_tool(self, **_kw):
            return {"entities": [{"type": "Agreement", "surface": "IMA-2025-018"},
                                 {"type": "FeeSchedule", "surface": "FS-2025-031"}],
                    "relations": [],
                    "facts": [
                        {"subject_id": 0, "property": "agreementId", "value": "IMA-2025-018",
                         "evidence": "IMA-2025-018"},
                        {"subject_id": 0, "property": "governingLaw", "value": "England and Wales",
                         "evidence": "England and Wales"},
                        {"subject_id": 1, "property": "feeScheduleId", "value": "FS-2025-031",
                         "evidence": "FS-2025-031"},
                    ]}

    text = "IMA-2025-018 governed by England and Wales, per Fee Schedule FS-2025-031."
    ent_rows, _e, _s, _u, _b = run_pipeline(
        [Chunk("c1", text)], ExtractionConfig(engine="llm", ontology_pack="investment_fibo"),
        _pack(), folder_id="f1", llm_client=_Llm())
    by_type = {r["entity_type"]: r for r in ent_rows}
    # the Agreement was described (governingLaw); the fee schedule was only named
    assert by_type["Agreement"]["reference_only"] is False
    assert by_type["FeeSchedule"]["reference_only"] is True
