"""KG-AC-47 (evolve v5, DB-integration, gated on DATABASE_URL): after canonicalize_batch re-points
entities to canonical_id, duplicate canonical edges collapse to one kg_canonical_edges row —
support_count = distinct source documents, confidence = max of contributors, evidence_text = top-3
sentences. Correct incrementally across batches (a later batch reinforcing the same canonical
relationship updates support_count, not a duplicate row). Runs inside the SAME atomic transaction as
entity resolution (KG-AC-40) — a mid-batch failure leaves no partial aggregation."""
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


def _seed_entity(cur, folder_id, uid, surface, etype):
    cur.execute(
        """INSERT INTO public.kg_entities
               (folder_id, entity_uid, entity_type, surface_form,
                ontology_pack, ontology_version, stage)
           VALUES (%s,%s,%s,%s,'fibo_core','1.0','staged')
           ON CONFLICT (entity_uid) DO NOTHING""",
        (folder_id, uid, etype, surface),
    )


def _seed_edge(cur, folder_id, edge_uid, rtype, src_uid, dst_uid, confidence, evidence_text,
               source_doc_id=None):
    # src/dst_entity_id: the mention-level FK -- reuse kg_entities.id looked up by entity_uid.
    cur.execute("SELECT id FROM public.kg_entities WHERE entity_uid=%s", (src_uid,))
    src_id = cur.fetchone()[0]
    cur.execute("SELECT id FROM public.kg_entities WHERE entity_uid=%s", (dst_uid,))
    dst_id = cur.fetchone()[0]
    cur.execute(
        """INSERT INTO public.kg_edges
               (folder_id, edge_uid, relation_type, src_entity_id, dst_entity_id,
                src_entity_uid, dst_entity_uid, confidence, evidence_text, source_doc_id)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (folder_id, edge_uid, rtype, src_id, dst_id, src_uid, dst_uid, confidence, evidence_text,
         source_doc_id),
    )


def _canonical_edge_row(cur, src_cid, rtype, dst_cid):
    cur.execute(
        """SELECT support_count, confidence, evidence_text FROM public.kg_canonical_edges
            WHERE src_canonical_id=%s AND relation_type=%s AND dst_canonical_id=%s""",
        (src_cid, rtype, dst_cid),
    )
    return cur.fetchone()


def _canonical_edge_source_doc_ids(cur, src_cid, rtype, dst_cid):
    cur.execute(
        """SELECT source_doc_ids FROM public.kg_canonical_edges
            WHERE src_canonical_id=%s AND relation_type=%s AND dst_canonical_id=%s""",
        (src_cid, rtype, dst_cid),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _cleanup(conn, folder_ids, canonical_keys=()):
    with conn.cursor() as cur:
        for f in folder_ids:
            cur.execute("DELETE FROM public.kg_edges WHERE folder_id=%s", (f,))
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (f,))
        for k in canonical_keys:
            cur.execute(
                """DELETE FROM public.kg_canonical_edges WHERE src_canonical_id IN
                       (SELECT canonical_id FROM public.kg_canonical_entities WHERE canonical_key=%s)
                    OR dst_canonical_id IN
                       (SELECT canonical_id FROM public.kg_canonical_entities WHERE canonical_key=%s)""",
                (k, k),
            )
        for k in canonical_keys:
            cur.execute("DELETE FROM public.kg_canonical_entities WHERE canonical_key=%s", (k,))
    conn.commit()


@pytest.mark.ac("KG-AC-47")
def test_duplicate_canonical_edges_collapse_with_aggregation(conn):
    fa, fb = f"canonedge-{uuid.uuid4()}", f"canonedge-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            # fa: Acme Corp -[issues]-> Acme Bond  (one mention-edge)
            _seed_entity(cur, fa, f"{fa}:0", "Acme Corp", "Organization")
            _seed_entity(cur, fa, f"{fa}:1", "Acme Bond", "Bond")
            _seed_edge(cur, fa, f"{fa}:e0", "issues", f"{fa}:0", f"{fa}:1", 0.6, "fa sentence")
            # fb: SAME real-world entities (different surface, same normalized), asserting the SAME
            # relationship again -- a second document supporting it, higher confidence.
            _seed_entity(cur, fb, f"{fb}:0", "Acme Corporation", "Organization")
            _seed_entity(cur, fb, f"{fb}:1", "Acme Bond", "Bond")
            _seed_edge(cur, fb, f"{fb}:e0", "issues", f"{fb}:0", f"{fb}:1", 0.9, "fb sentence")
        conn.commit()

        canonicalize_batch(_DbShim(conn), [fa, fb], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT canonical_id FROM public.kg_entities WHERE entity_uid=%s", (f"{fa}:0",))
            src_cid = cur.fetchone()[0]
            cur.execute("SELECT canonical_id FROM public.kg_entities WHERE entity_uid=%s", (f"{fa}:1",))
            dst_cid = cur.fetchone()[0]
            row = _canonical_edge_row(cur, src_cid, "issues", dst_cid)

        assert row is not None
        support_count, confidence, evidence_text = row
        assert support_count == 2  # two distinct documents (fa, fb) asserted it
        assert float(confidence) == pytest.approx(0.9)  # max
        assert set(evidence_text) == {"fa sentence", "fb sentence"}  # both fit in top-3
    finally:
        _cleanup(conn, [fa, fb], ["organization:acme", "bond:acme-bond"])


@pytest.mark.ac("KG-AC-80")
def test_canonical_edge_exposes_its_contributing_document_set(conn):
    # deliberately on a DIFFERENT basis than support_count (folder_id) -- source_doc_ids groups by
    # the actual document, so two mention-edges from the SAME document but different folders/batches
    # still count as ONE contributing document, alongside support_count's own (unchanged) folder count.
    fa, fb = f"canonedge-{uuid.uuid4()}", f"canonedge-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed_entity(cur, fa, f"{fa}:0", "Acme Corp", "Organization")
            _seed_entity(cur, fa, f"{fa}:1", "Acme Bond", "Bond")
            _seed_edge(cur, fa, f"{fa}:e0", "issues", f"{fa}:0", f"{fa}:1", 0.6, "fa sentence",
                      source_doc_id="docA")
            _seed_entity(cur, fb, f"{fb}:0", "Acme Corporation", "Organization")
            _seed_entity(cur, fb, f"{fb}:1", "Acme Bond", "Bond")
            _seed_edge(cur, fb, f"{fb}:e0", "issues", f"{fb}:0", f"{fb}:1", 0.9, "fb sentence",
                      source_doc_id="docB")
        conn.commit()

        canonicalize_batch(_DbShim(conn), [fa, fb], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT canonical_id FROM public.kg_entities WHERE entity_uid=%s", (f"{fa}:0",))
            src_cid = cur.fetchone()[0]
            cur.execute("SELECT canonical_id FROM public.kg_entities WHERE entity_uid=%s", (f"{fa}:1",))
            dst_cid = cur.fetchone()[0]
            doc_ids = _canonical_edge_source_doc_ids(cur, src_cid, "issues", dst_cid)

        assert doc_ids == ["docA", "docB"]
    finally:
        _cleanup(conn, [fa, fb], ["organization:acme", "bond:acme-bond"])


@pytest.mark.ac("KG-AC-47")
def test_aggregation_is_atomic_with_entity_resolution(conn):
    """A mid-batch failure rolls back the WHOLE transaction -- no orphaned kg_canonical_edges row
    left behind when entity resolution itself fails (KG-AC-40)."""
    f = f"canonedge-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed_entity(cur, f, f"{f}:0", "Acme Systems", "Organization")
            _seed_entity(cur, f, f"{f}:1", "Acme Solutions", "Organization")  # ambiguous band pair
        conn.commit()

        def boom(a, b):
            raise RuntimeError("adjudication failure mid-batch")

        with pytest.raises(RuntimeError):
            canonicalize_batch(_DbShim(conn), [f], fuzzy_floor=0.5, fuzzy_ceiling=0.99,
                              pack=FIBO, adjudicate=boom)
        conn.rollback()

        with conn.cursor() as cur:
            cur.execute("SELECT canonical_id, stage FROM public.kg_entities WHERE folder_id=%s", (f,))
            rows = cur.fetchall()
        assert all(cid is None and stage == "staged" for cid, stage in rows)
    finally:
        _cleanup(conn, [f])
