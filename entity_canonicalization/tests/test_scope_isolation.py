"""specs/knowledge-graph v16 — S5: department isolation + per-scope serialization
(KG-AC-97 canon half, KG-AC-100).

The G2 acceptance test: the same entity name in two scopes stays two canonical nodes sharing no
row. Pre-v16 `canonical_key` was globally UNIQUE, so both departments resolved onto ONE
`canonical_id`, the full-recompute merged both tenants' aliases and facts onto it, and each
department's export then carried the other's data into its "private" database.
"""
import os
import uuid

import pytest

from ontologies import load_pack
from store import batch_graph_scope, canonicalize_batch

_DSN = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not _DSN, reason="DATABASE_URL not set (DB-integration test)")
FIBO = load_pack("fibo_core")
LEGAL, FINANCE = "test-legal", "test-finance"


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


def _connect():
    import psycopg2
    c = psycopg2.connect(_norm(_DSN))
    c.autocommit = False
    return c


@pytest.fixture
def conn():
    c = _connect()
    yield c
    c.close()


def _seed(cur, folder_id, scope, surface="Global Endowment Trust", etype="Organization"):
    cur.execute(
        """INSERT INTO public.kg_entities
               (folder_id, graph_scope, entity_uid, entity_type, surface_form,
                ontology_pack, ontology_version, stage)
           VALUES (%s,%s,%s,%s,%s,'fibo_core','1.0','staged')
           ON CONFLICT (entity_uid) DO NOTHING""",
        (folder_id, scope, f"{scope}:{folder_id}:0", etype, surface),
    )


def _cleanup(conn, folders):
    with conn.cursor() as cur:
        for f in folders:
            cur.execute("DELETE FROM public.kg_edges WHERE folder_id=%s", (f,))
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (f,))
        cur.execute(
            """DELETE FROM public.kg_canonical_entities c
                WHERE c.graph_scope = ANY(%s)
                  AND NOT EXISTS (SELECT 1 FROM public.kg_entities e
                                   WHERE e.canonical_id = c.canonical_id)""",
            ([LEGAL, FINANCE],))
    conn.commit()


@pytest.mark.ac("KG-AC-97")
def test_the_same_entity_in_two_scopes_stays_two_canonical_nodes(conn):
    """G2's acceptance test, stated in the v16 freeze as the isolation guarantee."""
    f_legal, f_fin = f"iso-{uuid.uuid4()}", f"iso-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f_legal, LEGAL)
            _seed(cur, f_fin, FINANCE)
        conn.commit()

        canonicalize_batch(_DbShim(conn), [f_legal], pack=FIBO)
        canonicalize_batch(_DbShim(conn), [f_fin], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT canonical_id FROM public.kg_entities WHERE folder_id=%s", (f_legal,))
            legal_ids = {r[0] for r in cur.fetchall()}
            cur.execute(
                "SELECT DISTINCT canonical_id FROM public.kg_entities WHERE folder_id=%s", (f_fin,))
            fin_ids = {r[0] for r in cur.fetchall()}
            assert legal_ids and fin_ids
            assert not (legal_ids & fin_ids), (
                "two scopes share a canonical entity — one department's aliases and facts would "
                "be merged onto the other's node and exported into its database")

            # each canonical row is stamped with its own scope
            cur.execute(
                "SELECT graph_scope FROM public.kg_canonical_entities "
                "WHERE canonical_id::text = ANY(%s)",
                ([str(c) for c in (legal_ids | fin_ids)],))
            assert {r[0] for r in cur.fetchall()} == {LEGAL, FINANCE}
    finally:
        _cleanup(conn, [f_legal, f_fin])


@pytest.mark.ac("KG-AC-97")
def test_a_mixed_scope_batch_fails_loud(conn):
    """A batch spanning scopes has no single tenancy — canonicalizing it would silently write one
    scope's identities into the other's partition, so it must refuse rather than pick one."""
    f_legal, f_fin = f"iso-{uuid.uuid4()}", f"iso-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f_legal, LEGAL)
            _seed(cur, f_fin, FINANCE)
        conn.commit()
        with conn.cursor() as cur:
            with pytest.raises(ValueError, match="mixed"):
                batch_graph_scope(cur, [f_legal, f_fin])
        conn.rollback()
    finally:
        _cleanup(conn, [f_legal, f_fin])


@pytest.mark.ac("KG-AC-100")
def test_same_scope_batches_serialize_and_different_scopes_do_not(conn):
    """The lock is transaction-scoped, so an uncommitted batch still holds it. Probing with
    `pg_try_advisory_xact_lock` from a second session proves the exclusion without blocking the
    test: a same-scope batch cannot proceed concurrently (which is what removes the lost-update on
    the canonical recompute), while a different scope is unaffected."""
    folder = f"iso-{uuid.uuid4()}"
    other = _connect()
    try:
        with conn.cursor() as cur:
            _seed(cur, folder, LEGAL)
        conn.commit()

        canonicalize_batch(_DbShim(conn), [folder], pack=FIBO)  # holds the lock, NOT committed

        with other.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_xact_lock(hashtext(%s))", (f"kg_canon:{LEGAL}",))
            assert cur.fetchone()[0] is False, "a second same-scope batch could run concurrently"
            cur.execute("SELECT pg_try_advisory_xact_lock(hashtext(%s))", (f"kg_canon:{FINANCE}",))
            assert cur.fetchone()[0] is True, "a different scope was needlessly blocked"
        other.rollback()
        conn.commit()
    finally:
        other.close()
        _cleanup(conn, [folder])
