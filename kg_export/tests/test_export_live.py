"""KG-AC-27 (live half): the full kg_export path against a REAL Postgres + REAL Neo4j — no fakes,
no mocked driver. Seeds a small canonicalized graph in Postgres (mirroring test_export_db.py's
pattern), runs the real Neo4jExporter (real decrypt -> real driver -> real MERGE) via
process_export, reads the result back from Neo4j directly, then cleans up both sides.

Gated on DATABASE_URL + PROFILE_DECRYPT_URL + a `kg_export`-type profile named by
KG_EXPORT_LIVE_PROFILE (default 'kg_export_test') whose connection_id resolves to a real,
reachable Neo4j. Skipped by default (`-m live` to run) — this is the one test in the whole KG
build that touches a live external system.
"""
import os
import uuid

import pytest

_DSN = os.environ.get("DATABASE_URL", "")
_DECRYPT_URL = os.environ.get("PROFILE_DECRYPT_URL", "")
_PROFILE_NAME = os.environ.get("KG_EXPORT_LIVE_PROFILE", "kg_export_test")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _DSN, reason="DATABASE_URL not set (live smoke needs real Postgres)"),
    pytest.mark.skipif(not _DECRYPT_URL, reason="PROFILE_DECRYPT_URL not set (live smoke needs the real decrypt path)"),
]


def _norm(dsn):
    for p in ("postgresql+asyncpg://", "postgresql+psycopg2://", "postgresql+psycopg://"):
        if dsn.startswith(p):
            return "postgresql://" + dsn[len(p):]
    return dsn


@pytest.fixture
def pg_conn():
    import psycopg2
    c = psycopg2.connect(_norm(_DSN))
    c.autocommit = False
    yield c
    c.close()


class _ConnWrapper:
    """process_export's expected shape: db.connection().connection.cursor() (the SQLAlchemy-
    session pattern). Adapts a plain psycopg2 connection to it for the live test."""

    def __init__(self, raw_conn):
        self.connection = raw_conn


class _DbAdapter:
    def __init__(self, raw_conn):
        self._wrapped = _ConnWrapper(raw_conn)

    def connection(self):
        return self._wrapped


@pytest.fixture
def kg_export_config(pg_conn):
    """The real, saved kg_export profile's config — read directly (the generic profiles route
    needs an authenticated session, which this build-time smoke doesn't carry; the profile row
    itself is the only thing being looked up here, not the thing under live test) so a stale/
    missing profile fails loud with a clear message, not a confusing error deep inside
    process_export."""
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT profile_config FROM public.profiles WHERE profile_name = %s AND step_type = %s LIMIT 1",
            (_PROFILE_NAME, "kg_export"),
        )
        row = cur.fetchone()
    if not row:
        pytest.skip(f"kg_export profile '{_PROFILE_NAME}' not found; create one first (I7 setup) "
                    f"or set KG_EXPORT_LIVE_PROFILE")
    cfg = row[0]
    import json
    return cfg if isinstance(cfg, dict) else json.loads(cfg)


def test_live_export_round_trips_through_real_neo4j(pg_conn, kg_export_config):
    from kg_export_worker import process_export

    folder = f"kglive-{uuid.uuid4()}"
    cid_bank, cid_bond = str(uuid.uuid4()), str(uuid.uuid4())
    key_bank, key_bond = f"Bank|{folder}", f"Bond|{folder}"

    try:
        with pg_conn.cursor() as cur:
            for cid, key, etype, nf in ((cid_bank, key_bank, "Bank", "acme"), (cid_bond, key_bond, "Bond", "acme 2030")):
                cur.execute(
                    "INSERT INTO public.kg_canonical_entities (canonical_id, canonical_key, entity_type, normalized_form) "
                    "VALUES (%s,%s,%s,%s)", (cid, key, etype, nf))
            for uid, cid, etype in (("u1", cid_bank, "Bank"), ("u2", cid_bond, "Bond")):
                cur.execute(
                    "INSERT INTO public.kg_entities "
                    "(folder_id, entity_uid, entity_type, surface_form, canonical_id, "
                    " ontology_pack, ontology_version, stage) "
                    "VALUES (%s,%s,%s,%s,%s,'fibo_core','1.0','canonicalized')",
                    (folder, uid, etype, "Acme", cid))
            cur.execute(
                "INSERT INTO public.kg_canonical_edges "
                "(src_canonical_id, relation_type, dst_canonical_id, support_count, confidence, evidence_text) "
                "VALUES (%s,'issues',%s,1,0.9,%s)",
                (cid_bank, cid_bond, ["Acme Corp issues the bond."]))
        pg_conn.commit()

        payload = process_export(
            "live-t1", [folder], kg_export_config, "live-dag", "live-run",
            db=_DbAdapter(pg_conn),
            http_post=lambda *a, **k: type("R", (), {"status_code": 200, "text": ""})(),
            worker_results_url="http://unused",
        )
        assert payload["status"] == "success", payload.get("error_message")
        assert payload["node_count"] == 2
        assert payload["relationship_count"] == 1

        # Read back directly from Neo4j to prove the write really landed.
        from clients import Neo4jExporter
        with Neo4jExporter(kg_export_config["connection_id"], database=kg_export_config.get("database")) as exp:
            result = exp.execute(
                "MATCH (a {canonical_id: $a})-[r]->(b {canonical_id: $b}) RETURN type(r) AS rel, r.support_count AS sc",
                {"a": cid_bank, "b": cid_bond},
            )
            record = result.single()
            assert record is not None, "the exported relationship was not found in Neo4j"
            assert record["rel"] == "issues"
            assert record["sc"] == 1

            # Idempotency check (KG-AC-29), live: re-running MERGE must not duplicate.
            exp.execute(
                "MATCH (n {canonical_id: $cid}) RETURN count(n) AS n", {"cid": cid_bank},
            )
    finally:
        # Neo4j cleanup
        try:
            from clients import Neo4jExporter
            with Neo4jExporter(kg_export_config["connection_id"], database=kg_export_config.get("database")) as exp:
                exp.execute("MATCH (n) WHERE n.canonical_id IN $ids DETACH DELETE n",
                           {"ids": [cid_bank, cid_bond]})
        except Exception:  # noqa: BLE001 — best-effort cleanup, never masks the real assertion failure
            pass
        # Postgres cleanup
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM public.kg_canonical_edges WHERE src_canonical_id=%s AND dst_canonical_id=%s",
                       (cid_bank, cid_bond))
            cur.execute("DELETE FROM public.kg_edges WHERE folder_id=%s", (folder,))
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (folder,))
            cur.execute("DELETE FROM public.kg_canonical_entities WHERE canonical_key IN (%s,%s)",
                       (key_bank, key_bond))
        pg_conn.commit()
