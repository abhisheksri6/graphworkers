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


# ---- investment_fibo v2.0 integration (already declares all three) -------
@pytest.mark.ac("KG-AC-69")
def test_investment_fibo_datatype_properties_and_abstract_types_visible():
    p = load_pack("investment_fibo")
    assert len(p.datatype_properties) == 15
    assert p.datatype_properties["commitmentAmount"].domain == "Commitment"
    assert p.datatype_properties["commitmentAmount"].range == "number"
    abstract_types = {t for t, et in p.entity_types.items() if et.abstract}
    assert abstract_types == {"InvestmentRelationship", "Commitment", "CommercialTerms"}
    assert "lei" not in p.datatype_properties
