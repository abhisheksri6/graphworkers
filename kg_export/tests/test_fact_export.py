"""P14 (spec v13, KG-AC-83): merged facts reach Neo4j as node properties. ``kg_canonical_entities.
attributes`` (P11/P13's shape — ``{property: [{value, normalized_value, status, provenance}]}``)
becomes ``n.<property>`` (normalized_value when present else value) + ``n.<property>_status``. A
`conflicting` fact (>=2 distinct values) exports ALL competing values as a LIST, never a silently
chosen winner. Values are ALWAYS bound Cypher parameters, matching the worker's existing injection
posture for ids/properties — never string-interpolated into the query text.
"""
import pytest

from core import CanonicalNode, build_node_statement


def _entry(value, normalized_value=None, status="single_source"):
    return {"value": value, "normalized_value": normalized_value or value, "status": status,
            "provenance": []}


# ---- single-value facts (single_source / consistent) ---------------------------------------------
@pytest.mark.ac("KG-AC-83")
def test_single_source_fact_becomes_a_node_property_with_its_status():
    node = CanonicalNode(canonical_id="c1", entity_type="Agreement",
                         attributes={"governingLaw": [_entry("England and Wales", status="single_source")]})
    _cypher, params = build_node_statement(node)
    assert params["facts"]["governingLaw"] == "England and Wales"
    assert params["facts"]["governingLaw_status"] == "single_source"


@pytest.mark.ac("KG-AC-83")
def test_normalized_value_used_over_raw_value_when_present():
    node = CanonicalNode(canonical_id="c1", entity_type="Agreement", attributes={
        "effectiveDate": [_entry("15 March 2025", normalized_value="2025-03-15", status="consistent")],
    })
    _cypher, params = build_node_statement(node)
    assert params["facts"]["effectiveDate"] == "2025-03-15"
    assert params["facts"]["effectiveDate_status"] == "consistent"


@pytest.mark.ac("KG-AC-83")
def test_raw_value_used_when_no_normalized_value():
    node = CanonicalNode(canonical_id="c1", entity_type="Agreement",
                         attributes={"investmentStrategy": [_entry("Multi-Asset Income Strategy",
                                                                   normalized_value=None)]})
    node.attributes["investmentStrategy"][0]["normalized_value"] = None  # force the fallback path
    _cypher, params = build_node_statement(node)
    assert params["facts"]["investmentStrategy"] == "Multi-Asset Income Strategy"


# ---- conflicting facts: competing values PLUS status, never a chosen winner ----------------------
@pytest.mark.ac("KG-AC-83")
def test_conflicting_fact_exports_all_competing_values_as_a_list():
    node = CanonicalNode(canonical_id="c1", entity_type="Agreement", attributes={
        "effectiveDate": [
            _entry("15 March 2025", normalized_value="2025-03-15", status="conflicting"),
            _entry("20 March 2025", normalized_value="2025-03-20", status="conflicting"),
        ],
    })
    _cypher, params = build_node_statement(node)
    assert set(params["facts"]["effectiveDate"]) == {"2025-03-15", "2025-03-20"}
    assert isinstance(params["facts"]["effectiveDate"], list)
    assert params["facts"]["effectiveDate_status"] == "conflicting"


@pytest.mark.ac("KG-AC-83")
def test_conflicting_fact_never_silently_picks_a_winner():
    # the defect this AC exists to prevent, mirrored from KG-AC-78's own no-last-write-wins guard.
    node = CanonicalNode(canonical_id="c1", entity_type="Agreement", attributes={
        "status": [
            _entry("Active", status="conflicting"),
            _entry("Terminated", status="conflicting"),
            _entry("Suspended", status="conflicting"),
        ],
    })
    _cypher, params = build_node_statement(node)
    assert len(params["facts"]["status"]) == 3  # NONE dropped


# ---- multiple facts, independent handling ---------------------------------------------------------
@pytest.mark.ac("KG-AC-83")
def test_multiple_facts_each_become_their_own_property():
    node = CanonicalNode(canonical_id="c1", entity_type="Agreement", attributes={
        "governingLaw": [_entry("England and Wales", status="single_source")],
        "agreementId": [_entry("IMA-2025-018", status="consistent")],
    })
    _cypher, params = build_node_statement(node)
    assert params["facts"]["governingLaw"] == "England and Wales"
    assert params["facts"]["agreementId"] == "IMA-2025-018"


@pytest.mark.ac("KG-AC-83")
def test_no_attributes_yields_an_empty_facts_map_not_an_error():
    node = CanonicalNode(canonical_id="c1", entity_type="Agreement")
    _cypher, params = build_node_statement(node)
    assert params["facts"] == {}


# ---- injection safety: values are ALWAYS bound parameters -----------------------------------------
@pytest.mark.ac("KG-AC-83")
def test_adversarial_fact_value_cannot_alter_the_cypher_statement():
    adversarial = "x'}) DETACH DELETE n //"
    node = CanonicalNode(canonical_id="c1", entity_type="Agreement",
                         attributes={"governingLaw": [_entry(adversarial, status="single_source")]})
    cypher, params = build_node_statement(node)
    # the malicious string must NEVER appear in the query TEXT itself...
    assert adversarial not in cypher
    assert "DETACH DELETE" not in cypher
    # ...it can only be delivered as a bound parameter VALUE.
    assert params["facts"]["governingLaw"] == adversarial


@pytest.mark.ac("KG-AC-83")
def test_adversarial_value_in_a_conflicting_list_also_stays_bound():
    adversarial = "'}) MATCH (x) DETACH DELETE x //"
    node = CanonicalNode(canonical_id="c1", entity_type="Agreement", attributes={
        "status": [_entry("Active", status="conflicting"), _entry(adversarial, status="conflicting")],
    })
    cypher, params = build_node_statement(node)
    assert adversarial not in cypher
    assert adversarial in params["facts"]["status"]
