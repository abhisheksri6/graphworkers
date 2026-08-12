"""P1 (spec v13, KG-AC-69): the ontology pack's attribute vocabulary — `datatype_properties`,
loader-validated fail-loud the same way `relations` already is, and the per-type `abstract` flag
(KG-AC-72, schema-only here — P5 wires its runtime behavior). Pure — no spaCy/DB/network.
"""
import pytest

from ontologies import DatatypeProperty, EntityType, OntologyError, Pack, load_pack


def _pack_with_datatype_properties(datatype_properties, entity_types=None):
    entity_types = entity_types if entity_types is not None else [
        EntityType("Agreement", None, [], "", None),
    ]
    return Pack(
        name="x", version="1", description="",
        entity_types=entity_types, relations=[],
        datatype_properties=datatype_properties,
    )


# ---- exposure (KG-AC-69) --------------------------------------------------
@pytest.mark.ac("KG-AC-69")
def test_datatype_property_exposed_with_domain_range_guidance():
    p = _pack_with_datatype_properties(
        [DatatypeProperty(property="agreementId", domain="Agreement", range="identifier",
                          guidance="The agreement's reference number.")])
    dp = p.datatype_properties["agreementId"]
    assert dp.domain == "Agreement"
    assert dp.range == "identifier"
    assert dp.guidance == "The agreement's reference number."


@pytest.mark.ac("KG-AC-69")
def test_pack_without_datatype_properties_loads_unchanged():
    # the key is optional -- omitting it entirely must not raise or change any other behavior.
    p = Pack(name="x", version="1", description="",
             entity_types=[EntityType("Agreement", None, [], "", None)], relations=[])
    assert p.datatype_properties == {}


# ---- validation, fail-loud (KG-AC-69) -------------------------------------
@pytest.mark.ac("KG-AC-69")
def test_datatype_property_undeclared_domain_fails_loud():
    with pytest.raises(OntologyError):
        _pack_with_datatype_properties(
            [DatatypeProperty(property="agreementId", domain="Ghost", range="identifier", guidance="")])


@pytest.mark.ac("KG-AC-69")
def test_datatype_property_unknown_range_kind_fails_loud():
    with pytest.raises(OntologyError):
        _pack_with_datatype_properties(
            [DatatypeProperty(property="agreementId", domain="Agreement", range="boolean", guidance="")])


@pytest.mark.ac("KG-AC-69")
@pytest.mark.parametrize("kind", ["string", "number", "date", "identifier"])
def test_every_declared_range_kind_is_accepted(kind):
    p = _pack_with_datatype_properties(
        [DatatypeProperty(property="x", domain="Agreement", range=kind, guidance="")])
    assert p.datatype_properties["x"].range == kind


# ---- abstract flag (schema surface only — NOT KG-AC-72 itself: that AC's Given/When/Then is
# about RUNTIME span-less acceptance, which P5 implements and tests. These prove the pack schema
# carries the flag through loading, a prerequisite P5 depends on but does not itself satisfy the
# AC — marking KG-AC-72 here would claim more than these tests prove (the CB-OBS-15 lesson).) ----
def test_entity_type_abstract_defaults_false():
    et = EntityType("Agreement", None, [], "", None)
    assert et.abstract is False


def test_entity_type_abstract_declared_true_survives_loading():
    p = Pack(name="x", version="1", description="",
             entity_types=[EntityType("Deal", None, [], "", None, abstract=True)], relations=[])
    assert p.entity_types["Deal"].abstract is True


# ---- investment_fibo v2.2 integration (already declares all three) -------
@pytest.mark.ac("KG-AC-69")
def test_investment_fibo_datatype_properties_and_abstract_types_visible():
    p = load_pack("investment_fibo")
    # v2.1 (2026-08-11, same-day extended OWL): 42 datatype properties, up from v2.0's 15 -- fills
    # the CommercialTerms/FeeSchedule/Amendment/SideLetter/InvestmentManager gap the v2.0 pack's own
    # _known_gaps note flagged as missing.
    assert len(p.datatype_properties) == 42
    assert p.datatype_properties["commitmentAmount"].domain == "Commitment"
    assert p.datatype_properties["commitmentAmount"].range == "number"
    abstract_types = {t for t, et in p.entity_types.items() if et.abstract}
    assert abstract_types == {"InvestmentRelationship", "Commitment", "CommercialTerms"}
    # lei: v2.0 deliberately excluded it (owner decision, v13 clarify pass); v2.1 re-added it the
    # SAME day when the extended OWL arrived (owner decision, declaration only -- downstream
    # consumption, e.g. as a canonicalization match key, is a separate, not-yet-made decision).
    assert "lei" in p.datatype_properties
    assert p.datatype_properties["lei"].domain == "LegalEntity"
    assert p.datatype_properties["lei"].range == "identifier"
    # spot-check one property from each newly-filled domain (the v2.0 _known_gaps)
    assert p.datatype_properties["investmentManagerName"].domain == "InvestmentManager"
    assert p.datatype_properties["managementFee"].domain == "CommercialTerms"
    assert p.datatype_properties["feeScheduleId"].domain == "FeeSchedule"
    assert p.datatype_properties["amendmentId"].domain == "Amendment"
    assert p.datatype_properties["sideLetterId"].domain == "SideLetter"


@pytest.mark.ac("KG-AC-88")
def test_investment_fibo_v22_all_three_abstract_types_carry_derived():
    # v2.2 (2026-08-12): each of the pack's own three abstract types gains a `derived` block
    # (KG-AC-88) -- this pack is the real-world regression guard for Q1, not just a synthetic one.
    p = load_pack("investment_fibo")
    ir = p.entity_types["InvestmentRelationship"].derived
    assert ir.identity_from == "Agreement.agreementId"
    # v2.3 (spec v15): mint_when removed, pattern required -- the presence-of-anchor trigger
    # minted contentless hubs. InvestmentRelationship is the pack's only reified_relation.
    assert ir.pattern == "reified_relation"
    assert p.entity_types["Commitment"].derived.pattern == "attribute_bundle"
    assert p.entity_types["CommercialTerms"].derived.pattern == "attribute_bundle"
    assert set(ir.auto_relations) == {
        "hasInvestor", "hasInvestmentManager", "governedBy", "hasSubscription",
    }
    commitment = p.entity_types["Commitment"].derived
    assert commitment.identity_from == "Subscription.subscriptionId"
    assert commitment.auto_relations == ["hasCommitment"]
    terms = p.entity_types["CommercialTerms"].derived
    assert terms.identity_from == "Agreement.agreementId"
    # clarify F1's real-world proof: both of this pack's declared auto_relations carry
    # CommercialTerms in RANGE, never domain -- a domain-only validator would have rejected
    # this exact, already-shipped pack content.
    assert set(terms.auto_relations) == {"hasCommercialTerms", "definedByFeeSchedule"}
    for rel_name in terms.auto_relations:
        rel = p.relations[rel_name]
        assert "CommercialTerms" in rel.domain or "CommercialTerms" in rel.range
