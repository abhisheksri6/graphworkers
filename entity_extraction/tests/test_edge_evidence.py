"""KG-AC-46 (evolve v5): a relation carries `evidence_text` (the source sentence it was extracted
from) + `confidence`; a relation for which the model supplies no evidence sentence is dropped —
evidence is mandatory, dropped at parse time in LlmGraphStrategy (not build_edge_records, which stays
a generic row-builder so hand-built Relations, e.g. test_idempotency.py's DB fixture, are unaffected).
`evidence_text` survives into the written kg_edges row (DB-integration, gated on DATABASE_URL)."""
import os
import uuid

import pytest

from core import Relation, build_edge_records, entity_uid_key_map
from ontologies import load_pack
from strategies import Chunk, ExtractionConfig
from strategies.llm_graph import LlmGraphStrategy

FIBO = load_pack("fibo_core")

_DSN = os.environ.get("DATABASE_URL", "")


class _FakeLlmClient:
    """Evolve v6: complete_tool() returns an already-parsed dict, not a JSON string."""

    def __init__(self, response):
        self._response = response
        self.calls = 0
        self.resolved_model = "fake-model"
        self.usage = []

    def complete_tool(self, *, system_text, user_text, tool_name, tool_description, tool_schema):
        self.calls += 1
        return self._response


@pytest.mark.ac("KG-AC-46")
def test_relation_with_evidence_is_kept():
    response = {
        "entities": [
            {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
            {"type": "Bond", "surface": "Acme 5% 2030", "confidence": 0.8},
        ],
        "relations": [
            {"type": "issues", "src": "Acme Corp", "src_type": "Organization",
             "dst": "Acme 5% 2030", "dst_type": "Bond", "confidence": 0.7,
             "evidence": "Acme Corp issues Acme 5% 2030."},
        ],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    strat.extract([Chunk("c1", "Acme Corp issues Acme 5% 2030.")], ExtractionConfig(engine="llm"), FIBO)
    assert len(strat.relations) == 1
    assert strat.relations[0].evidence_text == "Acme Corp issues Acme 5% 2030."


@pytest.mark.ac("KG-AC-46")
def test_relation_without_evidence_is_dropped():
    response = {
        "entities": [
            {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
            {"type": "Bond", "surface": "Acme 5% 2030", "confidence": 0.8},
            {"type": "Person", "surface": "Jane Roe", "confidence": 0.9},
        ],
        "relations": [
            {"type": "issues", "src": "Acme Corp", "src_type": "Organization",
             "dst": "Acme 5% 2030", "dst_type": "Bond", "confidence": 0.7},
            {"type": "employs", "src": "Acme Corp", "src_type": "Organization",
             "dst": "Jane Roe", "dst_type": "Person", "confidence": 0.6,
             "evidence": "Acme Corp employs Jane Roe."},
        ],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    strat.extract([Chunk("c1", "text")], ExtractionConfig(engine="llm"), FIBO)
    # the 'issues' relation had no 'evidence' key -> dropped; 'employs' had one -> kept
    assert len(strat.relations) == 1
    assert strat.relations[0].relation_type == "employs"


@pytest.mark.ac("KG-AC-46")
def test_build_edge_records_carries_evidence_text():
    rel = Relation("issues", "Acme Corp", "Organization", "Acme 5% 2030", "Bond", "c1",
                   confidence=0.7, evidence_text="Acme Corp issues Acme 5% 2030.")
    ent_uid_map = {
        ("c1", "Organization", "Acme Corp"): "uid-src",
        ("c1", "Bond", "Acme 5% 2030"): "uid-dst",
    }
    rows = build_edge_records("f1", [rel], ent_uid_map)
    assert len(rows) == 1
    assert rows[0]["evidence_text"] == "Acme Corp issues Acme 5% 2030."
    assert rows[0]["confidence"] == 0.7


# ---- DB-integration: evidence_text survives the write (gated on DATABASE_URL) ----
# NOTE: a module-level `pytestmark` would skip every test above too -- the skip is applied only to
# the DB test function below.


def _norm_dsn(dsn: str) -> str:
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
    c = psycopg2.connect(_norm_dsn(_DSN))
    c.autocommit = False
    yield c
    c.close()


@pytest.mark.ac("KG-AC-46")
@pytest.mark.skipif(not _DSN, reason="DATABASE_URL not set (DB-integration test)")
def test_partition_replace_writes_evidence_text(conn):
    from core import Candidate, assign_occurrence_indices, build_entity_records, merge_candidates
    from store import partition_replace

    folder_id = f"kgevid-{uuid.uuid4()}"
    task_id = str(uuid.uuid4())
    cands = [
        Candidate("Acme Corp", "Organization", "c1", "llm", 0, 9),
        Candidate("Acme 5% 2030", "Bond", "c1", "llm", 20, 32),
    ]
    assign_occurrence_indices(cands)
    merged = merge_candidates(cands)
    ent_rows = build_entity_records(folder_id, merged, "fibo_core", "1.0")
    rel = Relation("issues", "Acme Corp", "Organization", "Acme 5% 2030", "Bond", "c1",
                   confidence=0.7, evidence_text="Acme Corp issues Acme 5% 2030.")
    edge_rows = build_edge_records(folder_id, [rel], entity_uid_key_map(ent_rows))
    try:
        partition_replace(_DbShim(conn), folder_id, "run-1", "dag-1", task_id, ent_rows, edge_rows)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT evidence_text, confidence FROM public.kg_edges WHERE folder_id=%s", (folder_id,)
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] == "Acme Corp issues Acme 5% 2030."
        assert float(row[1]) == pytest.approx(0.7)
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM public.kg_edges WHERE folder_id=%s", (folder_id,))
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (folder_id,))
        conn.commit()
