"""KG-AC-10 (partition-replace idempotency), KG-AC-8 (bulk rows written directly to Postgres), and
KG-AC-11 (full provenance on every row) — a DB-integration test against the real kg schema.

Gated on DATABASE_URL (a Postgres DSN). On a box with the dev/test DB it runs; offline it skips. It
uses a unique per-run folder_id and cleans up after itself.
"""
import os
import uuid

import pytest

from core import (
    Candidate, assign_occurrence_indices, build_edge_records, build_entity_records,
    entity_uid_key_map, merge_candidates, Relation,
)
from store import partition_replace

_DSN = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not _DSN, reason="DATABASE_URL not set (DB-integration test)")


def _norm_dsn(dsn: str) -> str:
    for p in ("postgresql+psycopg2://", "postgresql+asyncpg://", "postgresql+psycopg://"):
        if dsn.startswith(p):
            return "postgresql://" + dsn[len(p):]
    return dsn


class _DbShim:
    """Mimics the with_capability `db` session's `.connection().connection` -> raw psycopg2 conn."""
    def __init__(self, conn):
        self._c = conn

    def connection(self):
        return type("_W", (), {"connection": self._c})()


def _build_rows(folder_id):
    cands = [
        Candidate("Acme Corp", "Organization", "c1", "gazetteer", 0, 9,
                  external_id="5493001KJTIIGC8Y1R12", external_id_source="gleif_lei"),
        Candidate("Jane Roe", "Person", "c1", "spacy", 20, 28),
        Candidate("Acme 5% 2030", "Bond", "c1", "llm", 40, 52),
    ]
    assign_occurrence_indices(cands)
    merged = merge_candidates(cands)
    ent_rows = build_entity_records(folder_id, merged, "fibo_core", "1.0")
    rels = [Relation("issues", "Acme Corp", "Organization", "Acme 5% 2030", "Bond", "c1"),
            Relation("employs", "Acme Corp", "Organization", "Jane Roe", "Person", "c1")]
    edge_rows = build_edge_records(folder_id, rels, entity_uid_key_map(ent_rows))
    return ent_rows, edge_rows


def _snapshot(cur, folder_id):
    cur.execute("""SELECT entity_uid, entity_type, surface_form, folder_id, run_id, dag_id,
                          ontology_pack, ontology_version, extractor, external_id, external_id_source, stage
                     FROM public.kg_entities WHERE folder_id=%s ORDER BY entity_uid""", (folder_id,))
    ents = cur.fetchall()
    cur.execute("""SELECT edge_uid, relation_type, src_entity_uid, dst_entity_uid, folder_id
                     FROM public.kg_edges WHERE folder_id=%s ORDER BY edge_uid""", (folder_id,))
    edges = cur.fetchall()
    return ents, edges


@pytest.fixture
def conn():
    import psycopg2
    c = psycopg2.connect(_norm_dsn(_DSN))
    c.autocommit = False
    yield c
    c.close()


@pytest.mark.ac("KG-AC-10")
@pytest.mark.ac("KG-AC-8")
@pytest.mark.ac("KG-AC-11")
def test_partition_replace_idempotent_with_provenance(conn):
    folder_id = f"kgtest-{uuid.uuid4()}"
    task_id = str(uuid.uuid4())
    ent_rows, edge_rows = _build_rows(folder_id)
    try:
        # Run 1
        n_e, n_ed = partition_replace(_DbShim(conn), folder_id, "run-1", "dag-1", task_id, ent_rows, edge_rows)
        conn.commit()
        assert (n_e, n_ed) == (3, 2)
        with conn.cursor() as cur:
            ents1, edges1 = _snapshot(cur, folder_id)
        # KG-AC-8: bulk rows are in Postgres directly
        assert len(ents1) == 3 and len(edges1) == 2
        # KG-AC-11: provenance populated on every row
        for row in ents1:
            (_uid, _t, _s, f, r, d, pack, ver, extractor, *_rest) = row
            assert f == folder_id and r == "run-1" and d == "dag-1"
            assert pack == "fibo_core" and ver == "1.0" and extractor in ("gazetteer", "spacy", "llm")
        # the gazetteer row carries the LEI external_id
        gaz = [row for row in ents1 if row[8] == "gazetteer"][0]
        assert gaz[9] == "5493001KJTIIGC8Y1R12" and gaz[10] == "gleif_lei"

        # Run 2 (re-execute for the SAME folder; run_id differs) -> identical entity/edge SET (KG-AC-10)
        partition_replace(_DbShim(conn), folder_id, "run-2", "dag-1", str(uuid.uuid4()), ent_rows, edge_rows)
        conn.commit()
        with conn.cursor() as cur:
            ents2, edges2 = _snapshot(cur, folder_id)
        assert {e[0] for e in ents2} == {e[0] for e in ents1}  # same entity_uids, no accumulation
        assert {e[0] for e in edges2} == {e[0] for e in edges1}  # same edge_uids
        assert len(ents2) == 3 and len(edges2) == 2
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM public.kg_edges WHERE folder_id=%s", (folder_id,))
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (folder_id,))
        conn.commit()
