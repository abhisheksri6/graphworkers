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
    d = DerivedSpec(identity_from="Agreement.agreementId", mint_when="Agreement",
                    auto_relations=["hasInvestor", "governedBy"])
    p = _agreement_pack(derived=d)
    et = p.entity_types["InvestmentRelationship"]
    assert et.derived.identity_from == "Agreement.agreementId"
    assert et.derived.mint_when == "Agreement"
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
    d = DerivedSpec(identity_from="AgreementId", mint_when="Agreement", auto_relations=[])
    with pytest.raises(OntologyError, match="identity_from must be"):
        _agreement_pack(derived=d)


@pytest.mark.ac("KG-AC-88")
def test_identity_from_undeclared_type_fails_loud_naming_pack_and_type():
    d = DerivedSpec(identity_from="Contract.agreementId", mint_when="Agreement", auto_relations=[])
    with pytest.raises(OntologyError, match=r"pack 'x'.*InvestmentRelationship.*undeclared type 'Contract'"):
        _agreement_pack(derived=d)


@pytest.mark.ac("KG-AC-88")
def test_identity_from_undeclared_property_fails_loud():
    d = DerivedSpec(identity_from="Agreement.referenceNumber", mint_when="Agreement", auto_relations=[])
    with pytest.raises(OntologyError, match="not a declared datatype_properties entry"):
        _agreement_pack(derived=d)


@pytest.mark.ac("KG-AC-88")
def test_identity_from_property_declared_on_a_different_type_fails_loud():
    # agreementId IS declared, but on Agreement -- referencing it via a type that isn't its
    # declared domain must fail, not silently accept a mismatched pairing.
    d = DerivedSpec(identity_from="Investor.agreementId", mint_when="Agreement", auto_relations=[])
    with pytest.raises(OntologyError, match="not a declared datatype_properties entry"):
        _agreement_pack(derived=d)


# ---- mint_when validation ----------------------------------------------------------------------
@pytest.mark.ac("KG-AC-88")
def test_mint_when_undeclared_type_fails_loud():
    d = DerivedSpec(identity_from="Agreement.agreementId", mint_when="Contract", auto_relations=[])
    with pytest.raises(OntologyError, match="mint_when references undeclared type 'Contract'"):
        _agreement_pack(derived=d)


@pytest.mark.ac("KG-AC-88")
def test_mint_when_is_a_plain_type_name_not_an_expression():
    # clarify F4: no expression grammar exists or is parsed -- a plain declared type name is the
    # entire contract. This is a positive-path regression guard against ever adding one.
    d = DerivedSpec(identity_from="Agreement.agreementId", mint_when="Agreement", auto_relations=[])
    p = _agreement_pack(derived=d)
    assert p.entity_types["InvestmentRelationship"].derived.mint_when == "Agreement"


# ---- auto_relations validation (clarify F1's regression guard) ----------------------------------
@pytest.mark.ac("KG-AC-88")
def test_auto_relations_entry_not_a_declared_relation_fails_loud():
    d = DerivedSpec(identity_from="Agreement.agreementId", mint_when="Agreement",
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
                          derived=DerivedSpec(identity_from="Agreement.agreementId", mint_when="Agreement",
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
    d = DerivedSpec(identity_from="Agreement.agreementId", mint_when="Agreement",
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
                      derived=DerivedSpec(identity_from="Agreement.agreementId", mint_when="Agreement",
                                         auto_relations=["hasCommercialTerms"])),
            EntityType("Agreement", None, [], "", None),
        ],
        relations=[PackRelation("hasCommercialTerms", ["Agreement"], ["CommercialTerms"], "")],
        datatype_properties=[DatatypeProperty("agreementId", "Agreement", "identifier", "")],
    )
    assert p.entity_types["CommercialTerms"].derived.auto_relations == ["hasCommercialTerms"]


@pytest.mark.ac("KG-AC-88")
def test_multiple_auto_relations_mixed_domain_and_range_all_accepted():
    d = DerivedSpec(identity_from="Agreement.agreementId", mint_when="Agreement",
                    auto_relations=["hasInvestor", "governedBy"])
    p = _agreement_pack(derived=d)  # hasInvestor + governedBy both domain=InvestmentRelationship
    assert len(p.entity_types["InvestmentRelationship"].derived.auto_relations) == 2


# ---- error messages name pack + type (KG-AC-88's own fail-loud posture) -------------------------
@pytest.mark.ac("KG-AC-88")
def test_error_messages_name_the_pack_and_the_entity_type():
    d = DerivedSpec(identity_from="Agreement.agreementId", mint_when="Agreement",
                    auto_relations=["notDeclared"])
    with pytest.raises(OntologyError) as exc:
        _agreement_pack(derived=d)
    assert "pack 'x'" in str(exc.value)
    assert "InvestmentRelationship" in str(exc.value)
