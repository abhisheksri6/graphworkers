"""specs/knowledge-graph v16 — S3: scope threading through extraction (KG-AC-97, KG-AC-10 amended,
KG-AC-56 closure).

The scope is part of a mention's IDENTITY, not a label added afterwards: two departments extracting
the same document must produce disjoint rows that can never collide on `entity_uid`, and a re-run
must replace only its OWN scope's partition. DB cases follow the gated-on-DATABASE_URL pattern the
canonicalization suite uses.
"""
import os
import uuid

import pytest

from core import (
    Candidate, build_entity_records, compute_derived_entity_uid, compute_entity_uid,
)

_DSN = os.environ.get("DATABASE_URL", "")


# ---- identity (KG-AC-10 as amended) ----------------------------------------
@pytest.mark.ac("KG-AC-10")
def test_entity_uid_is_scope_sensitive():
    """The same mention in two scopes MUST hash differently — otherwise the two departments'
    rows collide on the entity_uid unique index and one silently overwrites the other."""
    args = ("f1", "chunk-1", "Organization", "Acme Corp", 10, 0)
    a = compute_entity_uid("legal", *args)
    b = compute_entity_uid("finance", *args)
    assert a != b, "entity_uid ignores graph_scope — two scopes would collide"
    assert a == compute_entity_uid("legal", *args), "entity_uid is not deterministic"


@pytest.mark.ac("KG-AC-10")
def test_derived_entity_uid_is_scope_sensitive():
    a = compute_derived_entity_uid("legal", "f1", "InvestmentRelationship", "IMA-2025-018")
    b = compute_derived_entity_uid("finance", "f1", "InvestmentRelationship", "IMA-2025-018")
    assert a != b
    assert a == compute_derived_entity_uid("legal", "f1", "InvestmentRelationship", "IMA-2025-018")


@pytest.mark.ac("KG-AC-97")
def test_entity_rows_carry_their_scope():
    cand = Candidate(surface_form="Acme Corp", entity_type="Organization",
                     source_chunk_id="c1", layer="llm", span_start=0, span_end=9)
    rows = build_entity_records("f1", [cand], "generic", "1.0", graph_scope="legal")
    assert rows[0]["graph_scope"] == "legal"


# ---- scoped partition replace + edge extractor persistence -----------------
class _DbShim:
    def __init__(self, conn):
        self._c = conn

    def connection(self):
        return type("_W", (), {"connection": self._c})()


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


def _cand(surface, chunk="c1", start=0):
    return Candidate(surface_form=surface, entity_type="Organization", source_chunk_id=chunk,
                     layer="llm", span_start=start, span_end=start + len(surface))


@pytest.mark.skipif(not _DSN, reason="DATABASE_URL not set (DB-integration test)")
@pytest.mark.ac("KG-AC-97")
def test_partition_replace_is_scoped_so_two_scopes_of_one_folder_coexist(conn):
    """THE department-isolation guarantee at the extraction layer: the same document processed into
    two scopes must NOT clobber — pre-v16 the DELETE was by folder_id alone, so the second scope's
    run would wipe the first's rows."""
    from store import partition_replace

    folder = f"scope-{uuid.uuid4()}"
    try:
        rows_a = build_entity_records(folder, [_cand("Acme Corp")], "generic", "1.0",
                                      graph_scope="legal")
        rows_b = build_entity_records(folder, [_cand("Acme Corp")], "generic", "1.0",
                                      graph_scope="finance")
        partition_replace(_DbShim(conn), folder, "run-1", "dag-1", str(uuid.uuid4()), rows_a, [],
                          graph_scope="legal")
        partition_replace(_DbShim(conn), folder, "run-1", "dag-1", str(uuid.uuid4()), rows_b, [],
                          graph_scope="finance")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT graph_scope, count(*) FROM public.kg_entities WHERE folder_id=%s "
                "GROUP BY graph_scope ORDER BY graph_scope", (folder,))
            assert cur.fetchall() == [("finance", 1), ("legal", 1)], (
                "writing a second scope clobbered the first — the partition replace is not scoped")
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM public.kg_edges WHERE folder_id=%s", (folder,))
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (folder,))
        conn.commit()


@pytest.mark.skipif(not _DSN, reason="DATABASE_URL not set (DB-integration test)")
@pytest.mark.ac("KG-AC-56")
def test_edge_rows_persist_their_extractor_provenance(conn):
    """KG-AC-56 promises EVERY written relation row carries `extractor`. The value was computed in
    the edge dicts all along and silently dropped at the INSERT because the column did not exist
    (2026-08-17 review, finding A6) — so 'derived' vs 'llm' was unauditable in the DB."""
    from core import compute_edge_uid
    from store import partition_replace

    folder = f"scope-{uuid.uuid4()}"
    try:
        cands = [_cand("Acme Corp", start=0), _cand("Beta Bank", start=20)]
        ent_rows = build_entity_records(folder, cands, "generic", "1.0", graph_scope="legal")
        src, dst = ent_rows[0]["entity_uid"], ent_rows[1]["entity_uid"]
        edge_rows = [{
            "edge_uid": compute_edge_uid(folder, "ownedBy", src, dst),
            "relation_type": "ownedBy", "src_entity_uid": src, "dst_entity_uid": dst,
            "confidence": 1.0, "evidence_text": None, "source_doc_id": None, "page": None,
            "extractor": "derived",
        }]
        partition_replace(_DbShim(conn), folder, "run-1", "dag-1", str(uuid.uuid4()),
                          ent_rows, edge_rows, graph_scope="legal")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT extractor, graph_scope FROM public.kg_edges WHERE folder_id=%s",
                        (folder,))
            assert cur.fetchall() == [("derived", "legal")]
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM public.kg_edges WHERE folder_id=%s", (folder,))
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (folder,))
        conn.commit()
