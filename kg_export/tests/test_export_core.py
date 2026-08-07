"""KG-AC-27 (round-trip: the Postgres canonical graph == the exported graph), KG-AC-28 (rebuildable —
drop + re-export reproduces it), KG-AC-29 (idempotent MERGE — no duplicates on re-run). Pure at the
Cypher-statement level + a stateful fake driver for MERGE semantics."""
import pytest

from clients import Neo4jExporter
from core import CanonicalEdge, CanonicalNode, build_export_statements, run_export, sanitize_ident


def _graph():
    nodes = [
        CanonicalNode("c1", "Bank", "acme", external_id="LEI-1", provenance={"folders": ["f1"]}),
        CanonicalNode("c2", "Bond", "acme 2030", provenance={"folders": ["f1"]}),
        CanonicalNode("c3", "Person", "jane roe", provenance={"folders": ["f1"]}),
    ]
    edges = [
        CanonicalEdge("c1", "issues", "c2"),
        CanonicalEdge("c1", "employs", "c3"),
    ]
    return nodes, edges


class _Recorder:
    def __init__(self):
        self.runs = []

    def __call__(self, cypher, params):
        self.runs.append((cypher, params))


@pytest.mark.ac("KG-AC-27")
def test_roundtrip_one_merge_per_node_and_edge():
    nodes, edges = _graph()
    rec = _Recorder()
    summary = run_export(nodes, edges, rec)
    assert summary == {"node_count": 3, "relationship_count": 2}
    node_runs = [r for r in rec.runs if r[0].startswith("MERGE (n:")]
    rel_runs = [r for r in rec.runs if "MERGE (a)-[" in r[0]]
    assert len(node_runs) == 3 and len(rel_runs) == 2
    # the canonical graph maps 1:1 to the exported statements (round-trip)
    assert {r[1]["canonical_id"] for r in node_runs} == {"c1", "c2", "c3"}
    # node labels come from the reconciled type; the LEI + provenance ride as properties
    bank = next(r for r in node_runs if r[1]["canonical_id"] == "c1")
    assert "`Bank`" in bank[0] and bank[1]["external_id"] == "LEI-1"


@pytest.mark.ac("KG-AC-29")
def test_all_statements_are_idempotent_merge():
    nodes, edges = _graph()
    for cypher, _ in build_export_statements(nodes, edges):
        assert "MERGE" in cypher and "CREATE" not in cypher  # never CREATE -> no duplicates on re-run


@pytest.mark.ac("KG-AC-28")
def test_rebuild_produces_identical_statements():
    nodes, edges = _graph()
    first = build_export_statements(nodes, edges)
    second = build_export_statements(nodes, edges)  # a fresh rebuild from the same Postgres state
    assert first == second


def test_sanitize_label_defends_bad_identifiers():
    assert sanitize_ident("Investment Adviser") == "Investment_Adviser"
    assert sanitize_ident("123bad")[0] == "_"
    assert sanitize_ident("") == "Unknown"


# ---- full exporter path with a stateful fake driver (MERGE dedup) ---------
class _FakeSession:
    def __init__(self, store):
        self._store = store

    def run(self, cypher, params):
        if cypher.startswith("MERGE (n:"):
            self._store["nodes"].add(params["canonical_id"])          # MERGE = upsert (idempotent)
        elif "MERGE (a)-[" in cypher:
            self._store["rels"].add((params["src"], params["dst"]))
        return None

    def close(self):
        pass


class _FakeDriver:
    def __init__(self, store):
        self._store = store

    def session(self, database=None):
        return _FakeSession(self._store)

    def close(self):
        pass


@pytest.mark.ac("KG-AC-29")
@pytest.mark.ac("KG-AC-28")
def test_reexport_is_idempotent_and_rebuildable_via_driver():
    nodes, edges = _graph()
    store = {"nodes": set(), "rels": set()}
    decrypt = lambda ref, expected_category: ("neo4j", {"uri": "bolt://x", "username": "u", "password": "p"})
    factory = lambda uri, user, pw: _FakeDriver(store)

    def _export():
        with Neo4jExporter("kg-conn", decrypt=decrypt, driver_factory=factory) as exp:
            return run_export(nodes, edges, exp.execute)

    _export()
    assert (len(store["nodes"]), len(store["rels"])) == (3, 2)
    # drop + rebuild, or simply re-run: MERGE dedups -> no growth (idempotent + rebuildable)
    _export()
    assert (len(store["nodes"]), len(store["rels"])) == (3, 2)
    # the Neo4jExporter also decrypts with the knowledge_graph category
    seen = {}

    def _capturing_decrypt(ref, expected_category):
        seen["cat"] = expected_category
        return "neo4j", {"uri": "bolt://x", "username": "u", "password": "p"}

    with Neo4jExporter("kg-conn", decrypt=_capturing_decrypt, driver_factory=factory):
        pass
    assert seen["cat"] == "knowledge_graph"


# ---- KG-AC-46/47 (evolve v5): evidence + confidence + support_count on the Neo4j relationship ----
@pytest.mark.ac("KG-AC-47")
def test_rel_statement_carries_support_confidence_and_evidence():
    from core import build_rel_statement

    edge = CanonicalEdge("c1", "issues", "c2", support_count=2, confidence=0.85,
                         evidence_text=["Acme Corp issues the bond.", "A second filing confirms it."])
    cypher, params = build_rel_statement(edge)
    assert params["support_count"] == 2
    assert params["confidence"] == 0.85
    assert params["evidence_text"] == ["Acme Corp issues the bond.", "A second filing confirms it."]
    assert "r.support_count" in cypher and "r.confidence" in cypher and "r.evidence_text" in cypher


@pytest.mark.ac("KG-AC-47")
def test_rel_statement_defaults_support_confidence_and_evidence():
    from core import build_rel_statement

    edge = CanonicalEdge("c1", "issues", "c2")  # nothing supplied
    _cypher, params = build_rel_statement(edge)
    assert params["support_count"] is None
    assert params["confidence"] is None
    assert params["evidence_text"] == []
