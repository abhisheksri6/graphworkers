"""specs/knowledge-graph v16 — S7: end-of-batch scope-local retraction (KG-AC-105, Postgres half).

The 2026-08-17 review's finding B1: **nothing ever retracted.** Re-extraction replaces a folder's
mention rows, but no code path removed the canonical rows those mentions used to back — so every
fix-and-re-run cycle left the previous run's canonical entities and edges behind, accumulating
forever. That was survivable only because the graph could be dropped and rebuilt; a shared,
continuously-written graph (G1) cannot rely on that, which is why v16 promotes retraction from
hygiene to a requirement.

Retraction runs INSIDE the KG-AC-40 transaction: a mid-batch failure must roll the deletions back
together with everything else, never leaving a half-retracted graph.
"""
import os
import uuid

import pytest

from ontologies import load_pack
from store import canonicalize_batch

_DSN = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not _DSN, reason="DATABASE_URL not set (DB-integration test)")
FIBO = load_pack("fibo_core")

_SCOPE = {"name": "test-retract"}


@pytest.fixture(autouse=True)
def _isolated_scope():
    _SCOPE["name"] = f"test-{uuid.uuid4()}"
    yield


def _norm(dsn):
    for p in ("postgresql+psycopg2://", "postgresql+asyncpg://", "postgresql+psycopg://"):
        if dsn.startswith(p):
            return "postgresql://" + dsn[len(p):]
    return dsn


class _DbShim:
    def __init__(self, conn):
        self._c = conn

    def connection(self):
        return type("_W", (), {"connection": self._c})()


@pytest.fixture
def conn():
    import psycopg2
    c = psycopg2.connect(_norm(_DSN))
    c.autocommit = False
    yield c
    c.close()


def _seed(cur, folder_id, surfaces):
    for i, surface in enumerate(surfaces):
        cur.execute(
            """INSERT INTO public.kg_entities
                   (folder_id, graph_scope, entity_uid, entity_type, surface_form,
                    ontology_pack, ontology_version, stage)
               VALUES (%s,%s,%s,'Organization',%s,'fibo_core','1.0','staged')
               ON CONFLICT (entity_uid) DO NOTHING""",
            (folder_id, _SCOPE["name"], f"{folder_id}:{i}:{surface}", surface),
        )


def _restage(cur, folder_id, surfaces):
    """What a re-extraction does: the folder's partition is replaced wholesale."""
    cur.execute("DELETE FROM public.kg_edges WHERE folder_id=%s", (folder_id,))
    cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (folder_id,))
    _seed(cur, folder_id, surfaces)


def _canonical_count(cur):
    cur.execute(
        "SELECT count(*) FROM public.kg_canonical_entities WHERE graph_scope=%s", (_SCOPE["name"],))
    return cur.fetchone()[0]


def _cleanup(conn, folders):
    with conn.cursor() as cur:
        for f in folders:
            cur.execute("DELETE FROM public.kg_edges WHERE folder_id=%s", (f,))
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (f,))
        cur.execute("DELETE FROM public.kg_canonical_edges WHERE graph_scope=%s", (_SCOPE["name"],))
        cur.execute(
            "DELETE FROM public.kg_canonical_entities WHERE graph_scope=%s", (_SCOPE["name"],))
    conn.commit()


@pytest.mark.ac("KG-AC-105")
def test_a_superseded_canonical_entity_is_swept_on_reprocess(conn):
    """THE B1 case: a folder re-extracted without an entity it used to yield leaves that entity's
    canonical row backed by nothing. Pre-v16 it stayed forever."""
    f = f"retract-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f, ["Acme Corp", "Zenith Holdings"])
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f], pack=FIBO)
        conn.commit()
        with conn.cursor() as cur:
            assert _canonical_count(cur) == 2

        # re-extraction no longer produces Zenith
        with conn.cursor() as cur:
            _restage(cur, f, ["Acme Corp"])
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            assert _canonical_count(cur) == 1, (
                "the superseded canonical entity is still there — nothing retracted it")
    finally:
        _cleanup(conn, [f])


@pytest.mark.ac("KG-AC-105")
def test_retraction_is_scope_local(conn):
    """A batch retracting its own scope must never reach into another tenant's partition."""
    f_a, f_b = f"retract-{uuid.uuid4()}", f"retract-{uuid.uuid4()}"
    other_scope = f"test-other-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f_a, ["Acme Corp"])
            # a neighbouring scope with a canonical entity whose mentions are then removed —
            # orphaned, but NOT this batch's business
            cur.execute(
                """INSERT INTO public.kg_canonical_entities
                       (graph_scope, canonical_key, entity_type, normalized_form)
                   VALUES (%s,'organization:neighbour','Organization','neighbour')""",
                (other_scope,))
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f_a], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM public.kg_canonical_entities WHERE graph_scope=%s",
                (other_scope,))
            assert cur.fetchone()[0] == 1, "retraction crossed a scope boundary"
            cur.execute(
                "DELETE FROM public.kg_canonical_entities WHERE graph_scope=%s", (other_scope,))
        conn.commit()
    finally:
        _cleanup(conn, [f_a, f_b])


@pytest.mark.ac("KG-AC-40")
def test_a_failure_after_retraction_rolls_the_deletions_back(conn):
    """Retraction lives inside the batch transaction, so an abort must restore what it removed —
    a half-retracted graph is exactly the mixed state KG-AC-40 forbids."""
    f = f"retract-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f, ["Acme Corp", "Zenith Holdings"])
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            _restage(cur, f, ["Acme Corp"])
        conn.commit()

        # canonicalize, then abort the transaction before committing
        canonicalize_batch(_DbShim(conn), [f], pack=FIBO)
        conn.rollback()

        with conn.cursor() as cur:
            assert _canonical_count(cur) == 2, (
                "the rollback did not restore the retracted rows — retraction escaped the "
                "batch transaction")
    finally:
        _cleanup(conn, [f])
