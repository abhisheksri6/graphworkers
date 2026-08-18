"""specs/knowledge-graph v16 — S8: the export contract for a long-lived, shared graph database
(KG-AC-104, KG-AC-105 export half).

Three defects the 2026-08-17 review found in the shipped export, all of which only bite once a
database is written continuously by several pipelines rather than dropped and rebuilt:

1. **The MERGE key was the per-type label.** Once KG-AC-103 lets a canonical type sharpen across
   batches (`Organization` → `InvestmentAdviser`), a re-labelled entity MERGEs on a DIFFERENT
   label and Neo4j creates a SECOND node with the same `canonical_id` — the export itself
   manufacturing the duplicates this whole spec exists to remove.
2. **No disambiguating properties reached Neo4j.** `canonical_name`, `aliases`, `reference_only`
   and the derived/abstract markers — everything v13–v15 built so a consumer could tell how a node
   came to be — stopped at Postgres, so an Agreement, its derived hub and a reference stub rendered
   as near-identical nodes carrying the same identifier string.
3. **Nothing was ever removed.** A canonical entity retracted in Postgres (S7) stayed in Neo4j.
"""
import pytest

from core import CanonicalNode, build_node_statement, reconcile_statement


def _node(**kw):
    base = dict(canonical_id="c1", entity_type="InvestmentAdviser")
    base.update(kw)
    return CanonicalNode(**base)


@pytest.mark.ac("KG-AC-104")
def test_merge_keys_on_a_stable_label_never_the_entity_type():
    """The type is DATA on the node, never the MERGE key."""
    cypher, params = build_node_statement(_node())
    assert "MERGE (n:`KgEntity` {canonical_id: $canonical_id})" in cypher, cypher
    assert ":`InvestmentAdviser`" not in cypher, (
        "the per-type label is still the MERGE key — a sharpening type would duplicate the node")
    assert params["entity_type"] == "InvestmentAdviser"


@pytest.mark.ac("KG-AC-104")
def test_a_type_change_merges_onto_the_same_node():
    """The concrete regression: the same canonical_id, seen under two types across batches, must
    produce the SAME MERGE pattern — one node, its type updated."""
    a, _ = build_node_statement(_node(entity_type="Organization"))
    b, _ = build_node_statement(_node(entity_type="InvestmentAdviser"))
    assert a == b, "two batches of one entity would MERGE onto different nodes"


@pytest.mark.ac("KG-AC-104")
def test_display_and_honesty_properties_are_exported_as_bound_parameters():
    """KG-AC-92's derived-vs-grounded distinction and KG-AC-94's stub marker only mean something if
    a consumer can SEE them; KG-AC-76/77's display fields likewise."""
    _cypher, params = build_node_statement(_node(
        canonical_name="Acme Capital Management", aliases=["Acme", "Acme Capital"],
        reference_only=True, is_abstract=True, is_derived=True))
    assert params["canonical_name"] == "Acme Capital Management"
    assert params["aliases"] == ["Acme", "Acme Capital"]
    assert params["reference_only"] is True
    assert params["is_abstract"] is True
    assert params["is_derived"] is True


@pytest.mark.ac("KG-AC-104")
def test_property_values_are_never_string_interpolated():
    """The worker's standing injection posture: only the (pack-vocabulary-closed) label is
    formatted; every value stays a bound parameter."""
    hostile = 'x" }) DETACH DELETE n //'
    cypher, params = build_node_statement(_node(canonical_name=hostile))
    assert hostile not in cypher
    assert params["canonical_name"] == hostile


@pytest.mark.ac("KG-AC-105")
def test_reconcile_deletes_only_canonical_ids_absent_from_the_plane_of_record():
    """Sound ONLY because one database holds exactly one scope (KG-AC-97/98): the set-diff of
    canonical_ids is authoritative there. DETACH so a node's relationships go with it."""
    cypher, params = reconcile_statement(["keep-1", "keep-2"])
    assert "DETACH DELETE" in cypher
    assert "NOT IN $keep" in cypher or "NOT n.canonical_id IN $keep" in cypher, cypher
    assert params["keep"] == ["keep-1", "keep-2"]
    assert ":`KgEntity`" in cypher, "reconcile must be bounded to the label this exporter owns"


class _Record:
    """Stand-in for a neo4j `Record`: a Mapping that is deliberately **NOT a dict**.

    This shape is the whole point of the class. The original fake returned plain dicts, so the
    preflight's `isinstance(row, dict)` guard passed in tests and then rejected a correctly
    provisioned database in production (found live 2026-08-17, on the owner's first pipeline run
    after v16). A fake that is easier to satisfy than the real driver is not a test."""

    def __init__(self, data):
        self._data = dict(data)

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)


class _FakeExp:
    def __init__(self, constraints):
        self._constraints = [_Record(c) for c in constraints]
        self.calls = []

    def execute(self, cypher, params):
        self.calls.append(cypher)
        return self._constraints


@pytest.mark.ac("KG-AC-104")
def test_constraint_preflight_accepts_a_provisioned_database():
    from kg_export_worker import assert_uniqueness_constraint
    exp = _FakeExp([{"labelsOrTypes": ["KgEntity"], "properties": ["canonical_id"]}])
    assert_uniqueness_constraint(exp)  # no raise
    assert exp.calls and "SHOW CONSTRAINTS" in exp.calls[0], (
        "the preflight must be READ-ONLY — the worker issues no schema DDL (KG-AC-52 posture)")


@pytest.mark.ac("KG-AC-104")
def test_constraint_preflight_fails_loud_with_a_runbook_pointer():
    """Without the constraint, concurrent MERGEs on one canonical_id can race into duplicate nodes
    — the very defect this export exists not to create. Better a loud failure than a graph that
    looks fine until it doesn't."""
    from clients import Neo4jConnectionError
    from kg_export_worker import assert_uniqueness_constraint

    with pytest.raises(Neo4jConnectionError, match="provisioning-runbook"):
        assert_uniqueness_constraint(_FakeExp([]))
    # a constraint on some OTHER label/property does not count
    with pytest.raises(Neo4jConnectionError):
        assert_uniqueness_constraint(
            _FakeExp([{"labelsOrTypes": ["Other"], "properties": ["canonical_id"]}]))


@pytest.mark.ac("KG-AC-105")
def test_reconcile_on_an_empty_canonical_set_is_still_bounded_to_the_label():
    """An empty scope legitimately empties its database — but must never touch anything that is
    not a :KgEntity this exporter wrote."""
    cypher, params = reconcile_statement([])
    assert params["keep"] == []
    assert ":`KgEntity`" in cypher
