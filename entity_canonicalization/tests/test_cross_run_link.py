"""specs/knowledge-graph v16 — S5: cross-run resolution through the MATCH path (KG-AC-99,
KG-AC-38 as amended).

This is the file KG-AC-38's row has named since v1 and which never existed. The 2026-08-17 review
found why it mattered: cross-run identity was **exact `canonical_key` equality only**. The design's
own promise — that the candidate block includes already-canonicalized rows, which then go through
the three-band match + adjudication — was never built, so ANY surface variation across batches
minted a second node. Within one batch "Acme Capital" and "Acme Capital Management" merge happily;
across batches they could not, because nothing ever compared them.

With several pipelines feeding one scope, cross-batch resolution is the steady state, not an edge
case.
"""
import os
import uuid

import pytest

from ontologies import load_pack
from store import canonicalize_batch

_DSN = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not _DSN, reason="DATABASE_URL not set (DB-integration test)")
FIBO = load_pack("fibo_core")
SCOPE = "test-xrun"


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


def _seed(cur, folder_id, mentions, scope=SCOPE):
    for i, (surface, etype) in enumerate(mentions):
        cur.execute(
            """INSERT INTO public.kg_entities
                   (folder_id, graph_scope, entity_uid, entity_type, surface_form,
                    ontology_pack, ontology_version, stage)
               VALUES (%s,%s,%s,%s,%s,'fibo_core','1.0','staged')
               ON CONFLICT (entity_uid) DO NOTHING""",
            (folder_id, scope, f"{scope}:{folder_id}:{i}", etype, surface),
        )


def _canonical_ids(cur, folder_id):
    cur.execute(
        "SELECT DISTINCT canonical_id FROM public.kg_entities WHERE folder_id=%s", (folder_id,))
    return [r[0] for r in cur.fetchall()]


def _cleanup(conn, folders):
    with conn.cursor() as cur:
        for f in folders:
            cur.execute("DELETE FROM public.kg_edges WHERE folder_id=%s", (f,))
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (f,))
        cur.execute(
            """DELETE FROM public.kg_canonical_entities c
                WHERE c.graph_scope=%s
                  AND NOT EXISTS (SELECT 1 FROM public.kg_entities e
                                   WHERE e.canonical_id = c.canonical_id)""", (SCOPE,))
    conn.commit()


@pytest.mark.ac("KG-AC-99")
def test_a_later_batch_resolves_a_surface_variant_to_the_existing_canonical(conn):
    """The defect this closes: batch 2's variant surface has a DIFFERENT canonical_key, so the
    exact-key fast path misses — and pre-v16 nothing else ran, so it minted a second node for one
    real company."""
    f1, f2 = f"xrun-{uuid.uuid4()}", f"xrun-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f1, [("Acme Capital Management", "Organization")])
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f1], pack=FIBO, adjudicate=lambda a, b: True)
        conn.commit()

        with conn.cursor() as cur:
            _seed(cur, f2, [("Acme Capital Mgmt", "Organization")])
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f2], pack=FIBO, adjudicate=lambda a, b: True)
        conn.commit()

        with conn.cursor() as cur:
            assert _canonical_ids(cur, f1) == _canonical_ids(cur, f2), (
                "the later batch minted a NEW canonical entity instead of resolving to the "
                "existing one — cross-run matching did not run")
    finally:
        _cleanup(conn, [f1, f2])


@pytest.mark.ac("KG-AC-99")
def test_a_genuinely_new_entity_still_mints_its_own_canonical(conn):
    """The rejection direction: cross-run matching must not become a merge-everything bucket."""
    f1, f2 = f"xrun-{uuid.uuid4()}", f"xrun-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f1, [("Acme Capital Management", "Organization")])
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f1], pack=FIBO, adjudicate=lambda a, b: False)
        conn.commit()

        with conn.cursor() as cur:
            _seed(cur, f2, [("Zenith Holdings", "Organization")])
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f2], pack=FIBO, adjudicate=lambda a, b: False)
        conn.commit()

        with conn.cursor() as cur:
            assert _canonical_ids(cur, f1) != _canonical_ids(cur, f2)
    finally:
        _cleanup(conn, [f1, f2])
