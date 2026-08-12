"""P14 (spec v13, KG-AC-86): nodes and relationships carry their ONTOLOGY-QUALIFIED name from the
pack's declared ``iri`` — verbatim, whatever form the pack author wrote it in (a full IRI like
``https://.../investment#Investor``, or a prefixed form like ``iiro:Investor`` if a future pack
declares it that way; kg_export applies NO transformation of its own — the pack's `iri` string
IS the qualified name to export, as-is). A type/relation with no `iri` exports the BARE name and
is explicitly NOT an error (most packs today declare no `iri` at all)."""
import pytest

from core import CanonicalEdge, CanonicalNode, build_node_statement, build_rel_statement, qualified_name


# ---- qualified_name (pure) -------------------------------------------------------------------
@pytest.mark.ac("KG-AC-86")
def test_qualified_name_uses_the_iri_verbatim_when_present():
    assert qualified_name("https://contextbuilder.ai/ontology/investment#Investor", "Investor") \
        == "https://contextbuilder.ai/ontology/investment#Investor"


@pytest.mark.ac("KG-AC-86")
def test_qualified_name_uses_a_prefixed_iri_verbatim_too():
    # if a pack author declares iri already in prefixed form, that IS the qualified name --
    # kg_export does not parse/rewrite it.
    assert qualified_name("iiro:Investor", "Investor") == "iiro:Investor"


@pytest.mark.ac("KG-AC-86")
def test_qualified_name_falls_back_to_the_bare_name_when_no_iri():
    assert qualified_name(None, "Investor") == "Investor"


@pytest.mark.ac("KG-AC-86")
def test_missing_iri_is_not_an_error():
    # explicit regression guard for the AC's own "not an error" clause.
    try:
        result = qualified_name(None, "Fund")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"missing iri must not raise, got {exc!r}")
    assert result == "Fund"


# ---- node export ----------------------------------------------------------------------------
@pytest.mark.ac("KG-AC-86")
def test_node_carries_ontology_class_from_entity_iri():
    node = CanonicalNode(canonical_id="c1", entity_type="Investor",
                         entity_iri="https://contextbuilder.ai/ontology/investment#Investor")
    _cypher, params = build_node_statement(node)
    assert params["ontology_class"] == "https://contextbuilder.ai/ontology/investment#Investor"


@pytest.mark.ac("KG-AC-86")
def test_node_with_no_entity_iri_exports_bare_name_not_an_error():
    node = CanonicalNode(canonical_id="c1", entity_type="Fund", entity_iri=None)
    cypher, params = build_node_statement(node)
    assert params["ontology_class"] == "Fund"
    assert "ontology_class" in cypher  # the property is still SET, just with the bare value


@pytest.mark.ac("KG-AC-86")
def test_ontology_class_value_is_a_bound_parameter_not_interpolated():
    adversarial_iri = "x'}) DETACH DELETE n //"
    node = CanonicalNode(canonical_id="c1", entity_type="Investor", entity_iri=adversarial_iri)
    cypher, params = build_node_statement(node)
    assert adversarial_iri not in cypher
    assert params["ontology_class"] == adversarial_iri


# ---- relationship export ---------------------------------------------------------------------
@pytest.mark.ac("KG-AC-86")
def test_relationship_carries_ontology_relation_from_relation_iri():
    edge = CanonicalEdge(src_canonical_id="a", relation_type="hasInvestor", dst_canonical_id="b",
                         relation_iri="https://contextbuilder.ai/ontology/investment#hasInvestor")
    _cypher, params = build_rel_statement(edge)
    assert params["ontology_relation"] == "https://contextbuilder.ai/ontology/investment#hasInvestor"


@pytest.mark.ac("KG-AC-86")
def test_relationship_with_no_relation_iri_exports_bare_name_not_an_error():
    edge = CanonicalEdge(src_canonical_id="a", relation_type="issues", dst_canonical_id="b",
                         relation_iri=None)
    cypher, params = build_rel_statement(edge)
    assert params["ontology_relation"] == "issues"
    assert "ontology_relation" in cypher


@pytest.mark.ac("KG-AC-86")
def test_ontology_relation_value_is_a_bound_parameter_not_interpolated():
    adversarial_iri = "'}) MATCH (x) DETACH DELETE x //"
    edge = CanonicalEdge(src_canonical_id="a", relation_type="issues", dst_canonical_id="b",
                         relation_iri=adversarial_iri)
    cypher, params = build_rel_statement(edge)
    assert adversarial_iri not in cypher
    assert params["ontology_relation"] == adversarial_iri
