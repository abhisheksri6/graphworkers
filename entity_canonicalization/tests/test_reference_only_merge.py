"""specs/knowledge-graph v16 — S6: the canonical-level `reference_only` merge (KG-AC-94's
cross-document clause).

KG-AC-94 marks an entity the document NAMED with an identifier but never described — a Fee Schedule
cited in a definitions clause whose actual terms live in another document. Extraction has set that
flag on the MENTION rows since v15, but the AC's cross-document half — *"a canonical entity is
`reference_only` iff EVERY contributing mention is"* — was never implemented, so the flag stopped
at the mention plane and the canonical graph (and therefore Neo4j) could not distinguish a stub
from a fully-read entity. The AC assigns this to canonicalization explicitly, and states the rule
in the AC that introduces the flag precisely because, left silent, whichever mention the merge
happened to pick would decide it non-deterministically.

"Described beats referenced": doc A's stub of FS-2025-031 and doc B's fully-read Fee Schedule merge
to `false`.
"""
import os
import uuid

import pytest

from ontologies import load_pack
from store import canonicalize_batch

_DSN = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not _DSN, reason="DATABASE_URL not set (DB-integration test)")
FIBO = load_pack("fibo_core")

_SCOPE = {"name": "test-refonly"}


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


def _seed(cur, folder_id, surface, reference_only, idx=0):
    cur.execute(
        """INSERT INTO public.kg_entities
               (folder_id, graph_scope, entity_uid, entity_type, surface_form, reference_only,
                ontology_pack, ontology_version, stage)
           VALUES (%s,%s,%s,'Organization',%s,%s,'fibo_core','1.0','staged')
           ON CONFLICT (entity_uid) DO NOTHING""",
        (folder_id, _SCOPE["name"], f"{folder_id}:{idx}", surface, reference_only),
    )


def _canonical_reference_only(cur, folder_id):
    cur.execute(
        """SELECT DISTINCT c.reference_only
             FROM public.kg_canonical_entities c
             JOIN public.kg_entities e ON e.canonical_id = c.canonical_id
            WHERE e.folder_id = %s""",
        (folder_id,),
    )
    rows = cur.fetchall()
    assert len(rows) == 1, f"expected one canonical entity, got {rows}"
    return rows[0][0]


def _cleanup(conn, folders):
    with conn.cursor() as cur:
        for f in folders:
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (f,))
        cur.execute("DELETE FROM public.kg_entities WHERE graph_scope=%s", (_SCOPE["name"],))
        cur.execute(
            "DELETE FROM public.kg_canonical_entities WHERE graph_scope=%s", (_SCOPE["name"],))
    conn.commit()


@pytest.mark.ac("KG-AC-94")
def test_a_cluster_of_only_stubs_is_reference_only(conn):
    f = f"refonly-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f, "Fee Schedule FS-2025-031", True)
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f], pack=FIBO)
        conn.commit()
        with conn.cursor() as cur:
            assert _canonical_reference_only(cur, f) is True
    finally:
        _cleanup(conn, [f])


@pytest.mark.ac("KG-AC-94")
def test_one_described_mention_makes_the_canonical_fully_described(conn):
    """Described beats referenced — the direction that must NOT be a majority vote or a
    first-writer-wins."""
    f = f"refonly-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f, "Fee Schedule FS-2025-031", True, idx=0)
            _seed(cur, f, "Fee Schedule FS-2025-031", False, idx=1)
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f], pack=FIBO)
        conn.commit()
        with conn.cursor() as cur:
            assert _canonical_reference_only(cur, f) is False
    finally:
        _cleanup(conn, [f])


@pytest.mark.ac("KG-AC-94")
def test_a_later_document_that_describes_the_entity_clears_the_flag(conn):
    """The cross-RUN case the AC is really about: the stub is the join point a later document
    merges into, so the flag must be recomputed from every row now sharing the canonical_id — not
    frozen at the founding batch."""
    f1, f2 = f"refonly-{uuid.uuid4()}", f"refonly-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f1, "Fee Schedule FS-2025-031", True)
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f1], pack=FIBO)
        conn.commit()
        with conn.cursor() as cur:
            assert _canonical_reference_only(cur, f1) is True  # stub-only so far

        with conn.cursor() as cur:
            _seed(cur, f2, "Fee Schedule FS-2025-031", False)
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f2], pack=FIBO)
        conn.commit()
        with conn.cursor() as cur:
            assert _canonical_reference_only(cur, f2) is False, (
                "the real document arrived but the canonical entity is still marked a stub")
    finally:
        _cleanup(conn, [f1, f2])
