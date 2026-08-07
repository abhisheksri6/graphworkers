"""KG-AC-27 (DB round-trip): read_canonical_graph over a seeded canonicalized graph yields one node
per canonical_id (duplicate mentions collapse) and one edge per canonical relationship, and run_export
projects exactly that. Gated on DATABASE_URL; seeds + cleans up."""
import os
import uuid

import pytest

from store import read_canonical_graph
from core import run_export

_DSN = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not _DSN, reason="DATABASE_URL not set (DB-integration test)")


def _norm(dsn):
    for p in ("postgresql+psycopg2://", "postgresql+asyncpg://", "postgresql+psycopg://"):
        if dsn.startswith(p):
            return "postgresql://" + dsn[len(p):]
    return dsn


@pytest.fixture
def conn():
    import psycopg2
    c = psycopg2.connect(_norm(_DSN))
    c.autocommit = False
    yield c
    c.close()


@pytest.mark.ac("KG-AC-47")
def test_db_round_trip_carries_support_confidence_and_evidence(conn):
    """kg_export reads the PRE-AGGREGATED kg_canonical_edges row directly (entity_canonicalization
    writes it, workers/entity_canonicalization/tests/test_edge_aggregation.py proves the aggregation
    itself) — this test proves the read/projection side: support_count, confidence, and the evidence
    array all survive into the CanonicalEdge kg_export builds the Neo4j relationship from."""
    folder = f"kgexpevid-{uuid.uuid4()}"
    cid_bank, cid_bond = str(uuid.uuid4()), str(uuid.uuid4())
    key_bank, key_bond = f"Bank|{folder}", f"Bond|{folder}"
    try:
        with conn.cursor() as cur:
            for cid, key, etype, nf in ((cid_bank, key_bank, "Bank", "acme"), (cid_bond, key_bond, "Bond", "acme 2030")):
                cur.execute(
                    """INSERT INTO public.kg_canonical_entities (canonical_id, canonical_key, entity_type, normalized_form)
                       VALUES (%s,%s,%s,%s)""", (cid, key, etype, nf))
            for uid, cid, etype in (("u1", cid_bank, "Bank"), ("u2", cid_bond, "Bond")):
                cur.execute(
                    """INSERT INTO public.kg_entities
                           (folder_id, entity_uid, entity_type, surface_form, canonical_id,
                            ontology_pack, ontology_version, stage)
                       VALUES (%s,%s,%s,%s,%s,'fibo_core','1.0','canonicalized')""",
                    (folder, uid, etype, "Acme", cid))
            # the pre-aggregated canonical edge, as entity_canonicalization would have written it
            cur.execute(
                """INSERT INTO public.kg_canonical_edges
                       (src_canonical_id, relation_type, dst_canonical_id, support_count, confidence, evidence_text)
                   VALUES (%s,'issues',%s,2,0.82,%s)""",
                (cid_bank, cid_bond, ["Acme Corp issues the bond.", "A second filing confirms it."]))
        conn.commit()

        with conn.cursor() as cur:
            _nodes, edges = read_canonical_graph(cur, [folder])
        assert len(edges) == 1
        assert edges[0].support_count == 2
        assert edges[0].confidence == pytest.approx(0.82)
        assert edges[0].evidence_text == ["Acme Corp issues the bond.", "A second filing confirms it."]
    finally:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM public.kg_canonical_edges WHERE src_canonical_id=%s AND dst_canonical_id=%s",
                (cid_bank, cid_bond),
            )
            cur.execute("DELETE FROM public.kg_edges WHERE folder_id=%s", (folder,))
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (folder,))
            cur.execute("DELETE FROM public.kg_canonical_entities WHERE canonical_key IN (%s,%s)", (key_bank, key_bond))
        conn.commit()


@pytest.mark.ac("KG-AC-27")
def test_db_round_trip_collapses_and_projects(conn):
    """Node collapse (3 mentions -> 2 canonical nodes) is kg_export's own NODE_SQL, unchanged by H4.
    Edge collapse (2 mention-edges -> 1 canonical edge) is now entity_canonicalization's
    responsibility (test_edge_aggregation.py proves that); here the pre-aggregated
    kg_canonical_edges row is seeded directly, and this test proves kg_export projects exactly it."""
    folder = f"kgexp-{uuid.uuid4()}"
    cid_bank, cid_bond = str(uuid.uuid4()), str(uuid.uuid4())
    key_bank, key_bond = f"Bank|{folder}", f"Bond|{folder}"
    try:
        with conn.cursor() as cur:
            for cid, key, etype, nf in ((cid_bank, key_bank, "Bank", "acme"), (cid_bond, key_bond, "Bond", "acme 2030")):
                cur.execute(
                    """INSERT INTO public.kg_canonical_entities (canonical_id, canonical_key, entity_type, normalized_form)
                       VALUES (%s,%s,%s,%s)""", (cid, key, etype, nf))
            # two mentions of the bank (collapse to ONE node) + one bond, all canonicalized
            mentions = [("u1", cid_bank, "Bank"), ("u2", cid_bank, "Bank"), ("u3", cid_bond, "Bond")]
            for uid, cid, etype in mentions:
                cur.execute(
                    """INSERT INTO public.kg_entities
                           (folder_id, entity_uid, entity_type, surface_form, canonical_id,
                            ontology_pack, ontology_version, stage)
                       VALUES (%s,%s,%s,%s,%s,'fibo_core','1.0','canonicalized')""",
                    (folder, uid, etype, "Acme", cid))
            # the ONE canonical edge entity_canonicalization would have aggregated from the batch's
            # (now-collapsed) mention-edges.
            cur.execute(
                """INSERT INTO public.kg_canonical_edges
                       (src_canonical_id, relation_type, dst_canonical_id, support_count, confidence)
                   VALUES (%s,'issues',%s,1,1.0)""",
                (cid_bank, cid_bond),
            )
        conn.commit()

        with conn.cursor() as cur:
            nodes, edges = read_canonical_graph(cur, [folder])
        # 3 mentions collapse to 2 canonical nodes; exactly 1 canonical edge is projected
        assert {n.canonical_id for n in nodes} == {cid_bank, cid_bond}
        assert len(nodes) == 2
        assert len(edges) == 1 and (edges[0].src_canonical_id, edges[0].dst_canonical_id) == (cid_bank, cid_bond)

        runs = []
        summary = run_export(nodes, edges, lambda c, p: runs.append((c, p)))
        assert summary == {"node_count": 2, "relationship_count": 1}
    finally:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM public.kg_canonical_edges WHERE src_canonical_id=%s AND dst_canonical_id=%s",
                (cid_bank, cid_bond),
            )
            cur.execute("DELETE FROM public.kg_edges WHERE folder_id=%s", (folder,))
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (folder,))
            cur.execute("DELETE FROM public.kg_canonical_entities WHERE canonical_key IN (%s,%s)", (key_bank, key_bond))
        conn.commit()
