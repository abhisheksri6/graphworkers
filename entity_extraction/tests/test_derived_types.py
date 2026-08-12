"""Q1 (spec v14, KG-AC-88): the pack `derived` block — loader-exposed and fail-loud validated,
same posture `datatype_properties` (KG-AC-69) already has. Declares HOW an abstract type is minted
deterministically after extraction (Q3, not this task), replacing the v13 model-synthesis-and-
position-reference mechanism (KG-AC-72) that dropped four ontology-valid relations as unresolved
references on a real deployed run and wrote `CommercialTerms` as an orphan with zero edges. This
file proves the SCHEMA is correct; Q3/Q4 prove the derivation pass that consumes it."""
import pytest

from ontologies import DatatypeProperty, DerivedSpec, EntityType, OntologyError, Pack
from ontologies import Relation as PackRelation


def _agreement_pack(derived=None):
    return Pack(
        name="x", version="1", description="",
        entity_types=[
            EntityType("InvestmentRelationship", None, [], "", None, abstract=True, derived=derived),
            EntityType("Agreement", None, [], "", None),
            EntityType("Investor", None, [], "", None),
        ],
        relations=[
            PackRelation("hasInvestor", ["InvestmentRelationship"], ["Investor"], ""),
            PackRelation("governedBy", ["InvestmentRelationship"], ["Agreement"], ""),
        ],
        datatype_properties=[DatatypeProperty("agreementId", "Agreement", "identifier", "")],
    )


# ---- exposure --------------------------------------------------------------
@pytest.mark.ac("KG-AC-88")
def test_derived_block_exposed_with_its_three_fields():
    d = DerivedSpec(identity_from="Agreement.agreementId", pattern="reified_relation",
                    auto_relations=["hasInvestor", "governedBy"])
    p = _agreement_pack(derived=d)
    et = p.entity_types["InvestmentRelationship"]
    assert et.derived.identity_from == "Agreement.agreementId"
    assert et.derived.pattern == "reified_relation"
    assert et.derived.auto_relations == ["hasInvestor", "governedBy"]


@pytest.mark.ac("KG-AC-88")
def test_derived_defaults_to_none():
    et = EntityType("X", None, [], "", None, abstract=True)
    assert et.derived is None


@pytest.mark.ac("KG-AC-88")
def test_abstract_type_with_no_derived_block_loads_clean_and_is_inert():
    # an abstract type omitting `derived` must load exactly as under v13 -- no validation error,
    # simply never minted (that's Q3's job to honor; this task only proves loading doesn't reject it).
    p = _agreement_pack(derived=None)
    assert p.entity_types["InvestmentRelationship"].derived is None


# ---- identity_from validation ------------------------------------------------------------------
@pytest.mark.ac("KG-AC-88")
def test_identity_from_missing_dot_fails_loud():
    d = DerivedSpec(identity_from="AgreementId", pattern="reified_relation", auto_relations=[])
    with pytest.raises(OntologyError, match="identity_from must be"):
        _agreement_pack(derived=d)


@pytest.mark.ac("KG-AC-88")
def test_identity_from_undeclared_type_fails_loud_naming_pack_and_type():
    d = DerivedSpec(identity_from="Contract.agreementId", pattern="reified_relation", auto_relations=[])
    with pytest.raises(OntologyError, match=r"pack 'x'.*InvestmentRelationship.*undeclared type 'Contract'"):
        _agreement_pack(derived=d)


@pytest.mark.ac("KG-AC-88")
def test_identity_from_undeclared_property_fails_loud():
    d = DerivedSpec(identity_from="Agreement.referenceNumber", pattern="reified_relation", auto_relations=[])
    with pytest.raises(OntologyError, match="not a declared datatype_properties entry"):
        _agreement_pack(derived=d)


@pytest.mark.ac("KG-AC-88")
def test_identity_from_property_declared_on_a_different_type_fails_loud():
    # agreementId IS declared, but on Agreement -- referencing it via a type that isn't its
    # declared domain must fail, not silently accept a mismatched pairing.
    d = DerivedSpec(identity_from="Investor.agreementId", pattern="reified_relation", auto_relations=[])
    with pytest.raises(OntologyError, match="not a declared datatype_properties entry"):
        _agreement_pack(derived=d)


# ---- pattern validation (v15 — replaces v14's mint_when) ---------------------------------------
@pytest.mark.ac("KG-AC-88")
def test_unknown_pattern_fails_loud_naming_pack_and_type():
    d = DerivedSpec(identity_from="Agreement.agreementId", pattern="magic", auto_relations=[])
    with pytest.raises(OntologyError, match=r"pack 'x'.*InvestmentRelationship.*unknown pattern 'magic'"):
        _agreement_pack(derived=d)


@pytest.mark.ac("KG-AC-88")
def test_both_declared_patterns_are_accepted():
    for pattern in ("reified_relation", "attribute_bundle"):
        d = DerivedSpec(identity_from="Agreement.agreementId", pattern=pattern, auto_relations=[])
        p = _agreement_pack(derived=d)
        assert p.entity_types["InvestmentRelationship"].derived.pattern == pattern


@pytest.mark.ac("KG-AC-88")
def test_mint_when_is_gone_and_a_pack_still_declaring_it_fails_to_load():
    # v15 REMOVED mint_when: its presence-of-anchor trigger minted contentless hubs (an Agreement
    # that merely REFERENCES a fee schedule would mint CommercialTerms with zero attributes). A
    # pack still carrying it must fail loudly rather than silently ignore a stale declaration.
    assert not hasattr(DerivedSpec(identity_from="A.b", pattern="reified_relation",
                                   auto_relations=[]), "mint_when")


# ---- auto_relations validation (clarify F1's regression guard) ----------------------------------
@pytest.mark.ac("KG-AC-88")
def test_auto_relations_entry_not_a_declared_relation_fails_loud():
    d = DerivedSpec(identity_from="Agreement.agreementId", pattern="reified_relation",
                    auto_relations=["notDeclared"])
    with pytest.raises(OntologyError, match="'notDeclared' is not a declared relation"):
        _agreement_pack(derived=d)


@pytest.mark.ac("KG-AC-88")
def test_auto_relations_entry_carrying_neither_domain_nor_range_fails_loud():
    # fundsInto is a genuinely declared relation, but it references neither domain nor range as
    # InvestmentRelationship -- attaching it to that abstract type's auto_relations is meaningless
    # and must fail loud, not silently no-op at derivation time.
    with pytest.raises(OntologyError, match="NEITHER its domain.*nor its range"):
        Pack(
            name="x", version="1", description="",
            entity_types=[
                EntityType("InvestmentRelationship", None, [], "", None, abstract=True,
                          derived=DerivedSpec(identity_from="Agreement.agreementId", pattern="reified_relation",
                                             auto_relations=["fundsInto"])),
                EntityType("Agreement", None, [], "", None),
                EntityType("Fund", None, [], "", None),
                EntityType("Investor", None, [], "", None),
            ],
            relations=[PackRelation("fundsInto", ["Investor"], ["Fund"], "")],
            datatype_properties=[DatatypeProperty("agreementId", "Agreement", "identifier", "")],
        )


@pytest.mark.ac("KG-AC-88")
def test_auto_relations_domain_side_is_accepted():
    d = DerivedSpec(identity_from="Agreement.agreementId", pattern="reified_relation",
                    auto_relations=["hasInvestor"])  # domain=InvestmentRelationship
    p = _agreement_pack(derived=d)
    assert p.entity_types["InvestmentRelationship"].derived.auto_relations == ["hasInvestor"]


@pytest.mark.ac("KG-AC-88")
def test_auto_relations_range_side_is_accepted_never_domain_only():
    # clarify F1: this is the exact regression the finding caught -- a relation carrying the
    # abstract type in RANGE only (not domain) must still validate. hasCommitment/
    # hasCommercialTerms in the real pack are both range-only; without this, both would have
    # been permanently unproducible, including hasCommercialTerms -- one of the four relations
    # whose live failure motivated v14 in the first place.
    p = Pack(
        name="x", version="1", description="",
        entity_types=[
            EntityType("CommercialTerms", None, [], "", None, abstract=True,
                      derived=DerivedSpec(identity_from="Agreement.agreementId", pattern="reified_relation",
                                         auto_relations=["hasCommercialTerms"])),
            EntityType("Agreement", None, [], "", None),
        ],
        relations=[PackRelation("hasCommercialTerms", ["Agreement"], ["CommercialTerms"], "")],
        datatype_properties=[DatatypeProperty("agreementId", "Agreement", "identifier", "")],
    )
    assert p.entity_types["CommercialTerms"].derived.auto_relations == ["hasCommercialTerms"]


@pytest.mark.ac("KG-AC-88")
def test_multiple_auto_relations_mixed_domain_and_range_all_accepted():
    d = DerivedSpec(identity_from="Agreement.agreementId", pattern="reified_relation",
                    auto_relations=["hasInvestor", "governedBy"])
    p = _agreement_pack(derived=d)  # hasInvestor + governedBy both domain=InvestmentRelationship
    assert len(p.entity_types["InvestmentRelationship"].derived.auto_relations) == 2


# ---- error messages name pack + type (KG-AC-88's own fail-loud posture) -------------------------
@pytest.mark.ac("KG-AC-88")
def test_error_messages_name_the_pack_and_the_entity_type():
    d = DerivedSpec(identity_from="Agreement.agreementId", pattern="reified_relation",
                    auto_relations=["notDeclared"])
    with pytest.raises(OntologyError) as exc:
        _agreement_pack(derived=d)
    assert "pack 'x'" in str(exc.value)
    assert "InvestmentRelationship" in str(exc.value)
