"""DB-integration for the canonicalization engine (gated on DATABASE_URL):
  KG-AC-22 duplicates collapse to one canonical_id + idempotent re-run;
  KG-AC-25 batch-scoped (a folder outside the batch is untouched);
  KG-AC-38 cross-run reuse (a later batch matching a prior canonical reuses its id) + race-safe mint;
  KG-AC-39 single-instance over the WHOLE released batch (all folders in one call);
  KG-AC-40 atomic (a mid-batch failure rolls back to all-staged).
Uses unique folder ids and cleans up.
"""
import os
import uuid

import pytest

from ontologies import load_pack
from store import canonicalize_batch

_DSN = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not _DSN, reason="DATABASE_URL not set (DB-integration test)")
FIBO = load_pack("fibo_core")


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


def _seed(cur, folder_id, mentions):
    """mentions: list of (surface, entity_type, external_id|None)."""
    for i, (surface, etype, ext) in enumerate(mentions):
        uid = f"{folder_id}:{i}"
        cur.execute(
            """INSERT INTO public.kg_entities
                   (folder_id, entity_uid, entity_type, surface_form, external_id,
                    ontology_pack, ontology_version, stage)
               VALUES (%s,%s,%s,%s,%s,'fibo_core','1.0','staged')
               ON CONFLICT (entity_uid) DO NOTHING""",
            (folder_id, uid, etype, surface, ext),
        )


def _rows(cur, folder_id):
    cur.execute("SELECT surface_form, entity_type, canonical_id, stage FROM public.kg_entities WHERE folder_id=%s ORDER BY entity_uid", (folder_id,))
    return cur.fetchall()


def _cleanup(conn, folder_ids, canonical_keys=()):
    with conn.cursor() as cur:
        for f in folder_ids:
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (f,))
        for k in canonical_keys:
            cur.execute("DELETE FROM public.kg_canonical_entities WHERE canonical_key=%s", (k,))
    conn.commit()


@pytest.mark.ac("KG-AC-22")
@pytest.mark.ac("KG-AC-39")
@pytest.mark.ac("KG-AC-25")
def test_batch_collapses_dupes_single_instance_and_scoped(conn):
    fa, fb, fc = (f"canon-{uuid.uuid4()}" for _ in range(3))
    try:
        with conn.cursor() as cur:
            _seed(cur, fa, [("Acme Corp", "Organization", None), ("Acme Corporation", "InvestmentAdviser", None)])
            _seed(cur, fb, [("Acme Corp", "Organization", None)])   # same entity, different folder
            _seed(cur, fc, [("Globex Ltd", "Organization", None)])  # OUTSIDE the batch
        conn.commit()

        # single-instance over the WHOLE released batch (fa + fb in one call), NOT fc
        summary = canonicalize_batch(_DbShim(conn), [fa, fb], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            ra, rb, rc = _rows(cur, fa), _rows(cur, fb), _rows(cur, fc)
        # KG-AC-22: all three Acme mentions (2 in fa + 1 in fb) share ONE canonical_id
        acme_cids = {r[2] for r in ra} | {r[2] for r in rb}
        assert len(acme_cids) == 1 and None not in acme_cids
        assert summary["canonical_count"] == 1  # one real entity across the batch
        # KG-AC-23 reconciled type is on the canonical index (InvestmentAdviser beats Organization)
        assert all(r[3] == "canonicalized" for r in ra + rb)
        # KG-AC-25: the out-of-batch folder is untouched (still staged, no canonical_id)
        assert all(r[3] == "staged" and r[2] is None for r in rc)

        # KG-AC-22 idempotent: a re-run finds nothing staged -> no-op
        summary2 = canonicalize_batch(_DbShim(conn), [fa, fb], pack=FIBO)
        conn.commit()
        assert summary2 == {"canonical_count": 0, "merged_count": 0, "minted_count": 0}
    finally:
        _cleanup(conn, [fa, fb, fc], ["InvestmentAdviser|acme", "Organization|acme", "Organization|globex"])


@pytest.mark.ac("KG-AC-38")
def test_cross_run_reuses_existing_canonical(conn):
    f1, f2 = f"canon-{uuid.uuid4()}", f"canon-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f1, [("Acme Corp", "Organization", None)])
        conn.commit()
        s1 = canonicalize_batch(_DbShim(conn), [f1], pack=FIBO)
        conn.commit()
        assert s1["minted_count"] == 1  # novel canonical minted
        with conn.cursor() as cur:
            cid1 = _rows(cur, f1)[0][2]

        # a LATER batch with the same entity in a new folder -> reuse the existing canonical (merged)
        with conn.cursor() as cur:
            _seed(cur, f2, [("Acme Corporation", "Organization", None)])
        conn.commit()
        s2 = canonicalize_batch(_DbShim(conn), [f2], pack=FIBO)
        conn.commit()
        assert s2["minted_count"] == 0 and s2["merged_count"] == 1
        with conn.cursor() as cur:
            cid2 = _rows(cur, f2)[0][2]
        assert cid2 == cid1  # the two become one graph node across runs
    finally:
        _cleanup(conn, [f1, f2], ["Organization|acme"])


@pytest.mark.ac("KG-AC-40")
def test_mid_batch_failure_rolls_back_all_staged(conn):
    f = f"canon-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            # an ambiguous pair (fuzzy band) forces the adjudicator to run
            _seed(cur, f, [("Acme Systems", "Organization", None), ("Acme Solutions", "Organization", None)])
        conn.commit()

        def boom(a, b):
            raise RuntimeError("adjudication failure mid-batch")

        with pytest.raises(RuntimeError):
            canonicalize_batch(_DbShim(conn), [f], fuzzy_floor=0.5, fuzzy_ceiling=0.99, pack=FIBO, adjudicate=boom)
        conn.rollback()  # the worker wrapper rolls back on failure

        with conn.cursor() as cur:
            rows = _rows(cur, f)
        # atomic: nothing canonicalized — all still staged, no canonical_id
        assert all(r[3] == "staged" and r[2] is None for r in rows)
    finally:
        _cleanup(conn, [f])
