"""R1 (spec v15, KG-AC-90/91): the derivation pass — mint trigger, structural identity, and
auto-relation attachment. `test_derived_types.py` covers the pack SCHEMA (KG-AC-88); this file
covers the runtime behaviour that consumes it.

The centrepiece is `test_golden_outcome_reproduced_*`: the owner-supplied golden for IMA-2025-018
asserts `InvestmentRelationship` but NOT `CommercialTerms`/`Commitment`, and v14's
presence-of-anchor trigger would have minted a `CommercialTerms` hub with zero attributes for a
document that merely *references* Fee Schedule FS-2025-031 by name. That finding, frozen as a test.
"""
import pytest

from core import compute_derived_entity_uid, derive_abstract_entities
from ontologies import load_pack


def _pack():
    return load_pack("investment_fibo")


def _row(entity_type, surface, uid=None, attributes=None, chunk="c1", doc="d.pdf", span=0):
    return {"entity_uid": uid or f"uid-{surface}", "entity_type": entity_type,
            "surface_form": surface, "source_chunk_id": chunk, "source_doc_id": doc,
            "page": 1, "span_start": span, "span_end": span + len(surface),
            "attributes": attributes or []}


def _attr(prop, value):
    return {"property": prop, "value": value, "normalized_value": value,
            "evidence": "e", "source_doc_id": "d.pdf", "page": 1}


# ---- structural identity (KG-AC-90) -------------------------------------------------------------
@pytest.mark.ac("KG-AC-90")
def test_derived_uid_is_structural_and_deterministic():
    a = compute_derived_entity_uid("f1", "InvestmentRelationship", "IMA-2025-018")
    b = compute_derived_entity_uid("f1", "InvestmentRelationship", "IMA-2025-018")
    assert a == b and len(a) == 64
    # a different identity value is a different entity; no NAME ever enters the hash
    assert a != compute_derived_entity_uid("f1", "InvestmentRelationship", "IMA-2025-019")


@pytest.mark.ac("KG-AC-90")
def test_derived_identity_is_stable_under_reordered_input():
    rows = [_row("Agreement", "the Agreement", attributes=[_attr("agreementId", "IMA-2025-018")]),
            _row("Investor", "XYZ INSURANCE GROUP PLC")]
    first, _, _ = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    second, _, _ = derive_abstract_entities("f1", list(reversed(rows)), _pack(), "investment_fibo", "2.3")
    assert [r["entity_uid"] for r in first] == [r["entity_uid"] for r in second]


# ---- THE golden regression guard (KG-AC-90) -----------------------------------------------------
@pytest.mark.ac("KG-AC-90")
def test_golden_outcome_reproduced_relationship_minted_bundles_not():
    rows = [
        _row("Agreement", "the Agreement", attributes=[
            _attr("agreementId", "IMA-2025-018"), _attr("governingLaw", "England and Wales")]),
        _row("Investor", "XYZ INSURANCE GROUP PLC"),
        _row("InvestmentManager", "T. ROWE PRICE ASSOCIATES, INC."),
        _row("FeeSchedule", "FS-2025-031", attributes=[_attr("feeScheduleId", "FS-2025-031")]),
        _row("SideLetter", "SL-2025-006", attributes=[_attr("sideLetterId", "SL-2025-006")]),
        _row("Subscription", "SUB-2025-041", attributes=[_attr("subscriptionId", "SUB-2025-041")]),
    ]
    derived, edges, counters = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    minted = {r["entity_type"] for r in derived}
    assert minted == {"InvestmentRelationship"}, f"only the reified relation may mint here; got {minted}"
    assert "CommercialTerms" not in minted   # no fee data in this document
    assert "Commitment" not in minted        # no commitment figures in this document
    assert counters["underivable_entity_count"] == 0


@pytest.mark.ac("KG-AC-90")
def test_attribute_bundle_mints_once_its_own_attribute_is_present():
    # a document that DOES state a fee -- CommercialTerms then has content. (R2's anchoring is what
    # delivers such an attribute in production; constructed here to prove the trigger this task owns.)
    rows = [_row("Agreement", "the Agreement", attributes=[
        _attr("agreementId", "IMA-2025-018"), _attr("managementFee", "1.5% per annum")])]
    derived, _, _ = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    assert "CommercialTerms" in {r["entity_type"] for r in derived}


@pytest.mark.ac("KG-AC-90")
def test_reified_relation_does_not_mint_without_a_participant():
    rows = [_row("Agreement", "the Agreement",
                attributes=[_attr("agreementId", "IMA-2025-018")])]  # identity only, no parties
    derived, _, _ = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    assert "InvestmentRelationship" not in {r["entity_type"] for r in derived}


@pytest.mark.ac("KG-AC-92")
def test_content_without_identity_is_skipped_and_counted():
    rows = [_row("Agreement", "the Agreement"),           # no agreementId => identity unresolvable
            _row("Investor", "XYZ INSURANCE GROUP PLC")]   # participant present => has content
    derived, _, counters = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    assert derived == []
    assert counters["underivable_entity_count"] >= 1


# ---- derived row shape (KG-AC-90/92) ------------------------------------------------------------
@pytest.mark.ac("KG-AC-90")
def test_derived_row_carries_identity_not_a_composed_name():
    rows = [_row("Agreement", "the Agreement", attributes=[_attr("agreementId", "IMA-2025-018")]),
            _row("Investor", "XYZ INSURANCE GROUP PLC")]
    derived, _, _ = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    hub = next(r for r in derived if r["entity_type"] == "InvestmentRelationship")
    assert hub["surface_form"] == "IMA-2025-018"     # never "<Investor> - <Manager> Relationship"
    assert hub["source_chunk_id"] is None             # document-scoped, not chunk-scoped
    assert hub["span_start"] is None and hub["span_end"] is None
    assert hub["is_abstract"] is True
    assert hub["extractor"] == "derived"              # KG-AC-92


# ---- auto-relation attachment (KG-AC-91) --------------------------------------------------------
@pytest.mark.ac("KG-AC-91")
def test_five_mentions_of_one_entity_yield_one_edge():
    rows = [_row("Agreement", "the Agreement", attributes=[_attr("agreementId", "IMA-2025-018")])]
    rows += [_row("Investor", "XYZ INSURANCE GROUP PLC", uid=f"u{i}", chunk=f"c{i}", span=i)
             for i in range(5)]
    _, edges, _ = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    investor_edges = [e for e in edges if e["relation_type"] == "hasInvestor"]
    assert len(investor_edges) == 1, f"representative-mention collapse failed: {len(investor_edges)}"


@pytest.mark.ac("KG-AC-91")
def test_two_distinct_investors_both_attach_syndicate_case():
    rows = [_row("Agreement", "the Agreement", attributes=[_attr("agreementId", "IMA-2025-018")]),
            _row("Investor", "XYZ INSURANCE GROUP PLC", uid="u1"),
            _row("Investor", "ABC PENSION TRUST", uid="u2", span=50)]
    _, edges, _ = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    assert len([e for e in edges if e["relation_type"] == "hasInvestor"]) == 2


@pytest.mark.ac("KG-AC-91")
def test_two_hubs_attach_nothing_and_count():
    rows = [_row("Agreement", "A1", uid="a1", attributes=[_attr("agreementId", "IMA-2025-018")]),
            _row("Agreement", "A2", uid="a2", span=50,
                attributes=[_attr("agreementId", "IMA-2025-019")]),
            _row("Investor", "XYZ INSURANCE GROUP PLC")]
    derived, edges, counters = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    hubs = [r for r in derived if r["entity_type"] == "InvestmentRelationship"]
    assert len(hubs) == 2, "both hubs are still minted -- only the pairing is withheld"
    assert edges == [], "a hub<->constituent pairing must never be invented"
    assert counters["ambiguous_attachment_count"] > 0


@pytest.mark.ac("KG-AC-91")
def test_edge_direction_follows_the_declaration_not_a_hub_first_assumption():
    # hasCommercialTerms is Agreement->CommercialTerms (hub in RANGE);
    # definedByFeeSchedule is CommercialTerms->FeeSchedule (hub in DOMAIN). Both must be emitted
    # with the declared direction -- the clarify-F1 domain-or-range pairing, at runtime.
    rows = [
        _row("Agreement", "the Agreement", uid="agr", attributes=[
            _attr("agreementId", "IMA-2025-018"), _attr("managementFee", "1.5%")]),
        _row("FeeSchedule", "FS-2025-031", uid="fs"),
    ]
    derived, edges, _ = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    hub = next(r for r in derived if r["entity_type"] == "CommercialTerms")
    inbound = next(e for e in edges if e["relation_type"] == "hasCommercialTerms")
    assert inbound["src_entity_uid"] == "agr" and inbound["dst_entity_uid"] == hub["entity_uid"]
    outbound = next(e for e in edges if e["relation_type"] == "definedByFeeSchedule")
    assert outbound["src_entity_uid"] == hub["entity_uid"] and outbound["dst_entity_uid"] == "fs"


@pytest.mark.ac("KG-AC-91")
def test_absent_constituent_type_produces_no_edge_and_no_error():
    rows = [_row("Agreement", "the Agreement", attributes=[_attr("agreementId", "IMA-2025-018")]),
            _row("Investor", "XYZ INSURANCE GROUP PLC")]
    _, edges, _ = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    assert [e for e in edges if e["relation_type"] == "hasSubscription"] == []  # none in folder
    assert [e for e in edges if e["relation_type"] == "hasInvestor"] != []


@pytest.mark.ac("KG-AC-92")
def test_derived_edge_carries_no_fabricated_evidence_but_keeps_provenance():
    rows = [_row("Agreement", "the Agreement", attributes=[_attr("agreementId", "IMA-2025-018")]),
            _row("Investor", "XYZ INSURANCE GROUP PLC", doc="ima.pdf")]
    _, edges, _ = derive_abstract_entities("f1", rows, _pack(), "investment_fibo", "2.3")
    edge = next(e for e in edges if e["relation_type"] == "hasInvestor")
    assert edge["evidence_text"] is None      # no sentence asserts a derived edge directly
    assert edge["extractor"] == "derived"
    assert edge["source_doc_id"] == "ima.pdf"  # ...but the constituent's provenance is kept


@pytest.mark.ac("KG-AC-90")
def test_empty_folder_is_a_clean_no_op():
    derived, edges, counters = derive_abstract_entities("f1", [], _pack(), "investment_fibo", "2.3")
    assert derived == [] and edges == []
    assert counters == {"underivable_entity_count": 0, "ambiguous_attachment_count": 0}
