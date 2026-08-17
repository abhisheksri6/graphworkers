"""specs/knowledge-graph v16 — S10 converge finding: the export read is scope-filtered (KG-AC-97).

Found by the S10 converge pass, not by a failing test: `read_canonical_graph` filtered by
`folder_ids` only, so the **rebuild** path (`folder_ids=None`, KG-AC-28 — "Neo4j is a rebuildable
projection") read EVERY scope's canonical rows and would MERGE another department's graph into this
database. The folder-scoped path was implicitly safe (one batch is one scope, worker-validated);
the rebuild path was not, and a rebuild is precisely the operation an operator reaches for after an
incident, i.e. the worst moment to leak tenants into each other.
"""
import os
import uuid

import pytest

from store import read_canonical_graph

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


def _seed(cur, scope, folder, cid, key, surface):
    cur.execute(
        """INSERT INTO public.kg_canonical_entities
               (canonical_id, canonical_key, entity_type, normalized_form, graph_scope)
           VALUES (%s,%s,'Organization',%s,%s)""",
        (cid, key, surface, scope))
    cur.execute(
        """INSERT INTO public.kg_entities
               (folder_id, graph_scope, entity_uid, entity_type, surface_form, canonical_id,
                ontology_pack, ontology_version, stage)
           VALUES (%s,%s,%s,'Organization',%s,%s,'fibo_core','1.0','canonicalized')""",
        (folder, scope, f"{folder}:0", surface, cid))


@pytest.mark.ac("KG-AC-97")
def test_a_full_rebuild_reads_only_its_own_scope(conn):
    scope_a, scope_b = f"test-a-{uuid.uuid4()}", f"test-b-{uuid.uuid4()}"
    f_a, f_b = f"exp-{uuid.uuid4()}", f"exp-{uuid.uuid4()}"
    cid_a, cid_b = str(uuid.uuid4()), str(uuid.uuid4())
    try:
        with conn.cursor() as cur:
            _seed(cur, scope_a, f_a, cid_a, f"organization:a-{cid_a[:8]}", "acme")
            _seed(cur, scope_b, f_b, cid_b, f"organization:b-{cid_b[:8]}", "zenith")
        conn.commit()

        with conn.cursor() as cur:
            # the REBUILD path: no folder_ids at all
            nodes, _edges = read_canonical_graph(cur, None, graph_scope=scope_a)
        ids = {n.canonical_id for n in nodes}
        assert cid_a in ids, "the scope's own entity is missing from its rebuild"
        assert cid_b not in ids, (
            "a rebuild read another scope's canonical entity — it would be MERGEd into this "
            "department's database")

        # Prove the filter is load-bearing rather than vacuously true: the UNSCOPED read (the
        # pre-fix behaviour) does see both, so the exclusion above is the scope filter doing work
        # and not simply an absence of data to leak.
        with conn.cursor() as cur:
            unscoped, _ = read_canonical_graph(cur, None)
        unscoped_ids = {n.canonical_id for n in unscoped}
        assert {cid_a, cid_b} <= unscoped_ids, (
            "precondition failed — both scopes' rows must be visible unfiltered, else this test "
            "would pass even with the filter removed")
    finally:
        with conn.cursor() as cur:
            for f in (f_a, f_b):
                cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (f,))
            cur.execute(
                "DELETE FROM public.kg_canonical_entities WHERE graph_scope = ANY(%s)",
                ([scope_a, scope_b],))
        conn.commit()
