"""KG-AC-52: the kg_export profile's `database` is the write-target Neo4j database — the worker
opens the session against it (no `database` -> defaults to `'neo4j'`); the worker NEVER issues
`CREATE DATABASE` (the export connection is write-only privilege; a non-existent/unreachable
database fails loud as `Neo4jConnectionError`, never a silent no-op). KG-AC-27: `process_export`
threads the resolved profile's `database` through to the exporter."""
import pytest

from clients import Neo4jConnectionError, Neo4jExporter
from kg_export_worker import process_export


class _FakeSession:
    def __init__(self, store, *, fail=False):
        self._store = store
        self._fail = fail

    def run(self, cypher, params):
        if self._fail:
            raise RuntimeError("database does not exist")  # simulates the driver's real failure
        if cypher.startswith("MERGE (n:"):
            self._store["nodes"].add(params["canonical_id"])
        elif "MERGE (a)-[" in cypher:
            self._store["rels"].add((params["src"], params["dst"]))
        return None

    def close(self):
        pass


class _FakeDriver:
    def __init__(self, store, *, fail=False):
        self._store = store
        self._fail = fail
        self.opened_with_database = "__not_called__"

    def session(self, database=None):
        self.opened_with_database = database
        return _FakeSession(self._store, fail=self._fail)

    def close(self):
        pass


def _decrypt(ref, expected_category):
    assert expected_category == "knowledge_graph"
    return "neo4j", {"uri": "bolt://x", "username": "u", "password": "p"}


@pytest.mark.ac("KG-AC-52")
def test_session_opens_with_the_configured_database():
    store = {"nodes": set(), "rels": set()}
    driver = _FakeDriver(store)
    with Neo4jExporter("c", database="graphdb", decrypt=_decrypt,
                       driver_factory=lambda *a, **k: driver) as exp:
        pass
    assert driver.opened_with_database == "graphdb"


@pytest.mark.ac("KG-AC-52")
def test_session_defaults_to_neo4j_database_when_unset():
    store = {"nodes": set(), "rels": set()}
    driver = _FakeDriver(store)
    with Neo4jExporter("c", database=None, decrypt=_decrypt,
                       driver_factory=lambda *a, **k: driver) as exp:
        pass
    assert driver.opened_with_database == "neo4j"


@pytest.mark.ac("KG-AC-52")
def test_session_defaults_to_neo4j_database_when_empty_string():
    store = {"nodes": set(), "rels": set()}
    driver = _FakeDriver(store)
    with Neo4jExporter("c", database="", decrypt=_decrypt,
                       driver_factory=lambda *a, **k: driver) as exp:
        pass
    assert driver.opened_with_database == "neo4j"


@pytest.mark.ac("KG-AC-52")
def test_no_create_database_statement_is_ever_issued():
    # __enter__ opens the session but must not itself run any Cypher (no ensure/create step).
    store = {"nodes": set(), "rels": set()}
    driver = _FakeDriver(store)
    run_calls = []
    real_session_factory = driver.session

    def _spy_session(database=None):
        sess = real_session_factory(database=database)
        orig_run = sess.run
        sess.run = lambda cypher, params: (run_calls.append(cypher), orig_run(cypher, params))[1]
        return sess

    driver.session = _spy_session
    with Neo4jExporter("c", database="graphdb", decrypt=_decrypt,
                       driver_factory=lambda *a, **k: driver):
        pass  # __enter__/__exit__ only -- no execute() calls made
    assert run_calls == []
    assert not any("CREATE DATABASE" in c for c in run_calls)


@pytest.mark.ac("KG-AC-52")
def test_nonexistent_database_fails_loud_as_neo4j_connection_error():
    store = {"nodes": set(), "rels": set()}
    driver = _FakeDriver(store, fail=True)
    with pytest.raises(Neo4jConnectionError):
        with Neo4jExporter("c", database="ghost-db", decrypt=_decrypt,
                           driver_factory=lambda *a, **k: driver) as exp:
            exp.execute("MERGE (n:`Bank` {canonical_id: $canonical_id})", {"canonical_id": "c1"})


# ---- process_export threads the profile's database through (KG-AC-27) --------------------
class _FakeExporterCM:
    """A no-op context manager standing in for Neo4jExporter."""

    def __init__(self, connection_id, database=None):
        self.connection_id = connection_id
        self.database = database

    def __enter__(self):
        return self

    def execute(self, cypher, params):
        return None

    def __exit__(self, *exc):
        return False


class _FakeDb:
    def connection(self):
        class _Conn:
            def cursor(self):
                class _Cur:
                    def __enter__(self_inner):
                        return self_inner

                    def __exit__(self_inner, *a):
                        return False

                    def execute(self_inner, *a):
                        pass

                    def fetchall(self_inner):
                        return []

                return _Cur()

        return type("_W", (), {"connection": _Conn()})()


def _http_post(url, json=None, timeout=None):
    return type("R", (), {"status_code": 200, "text": ""})()


@pytest.mark.ac("KG-AC-27")
def test_process_export_constructs_exporter_with_database_from_config(monkeypatch):
    import kg_export_worker as worker_mod

    captured = {}

    def _fake_exporter_ctor(connection_id, *, database=None, **kwargs):
        captured["connection_id"] = connection_id
        captured["database"] = database
        return _FakeExporterCM(connection_id, database=database)

    monkeypatch.setattr(worker_mod, "Neo4jExporter", _fake_exporter_ctor)
    process_export(
        "t1", ["f1"], {"connection_id": "neo-conn", "database": "graphdb"}, "d", "r",
        db=_FakeDb(), http_post=_http_post, worker_results_url="http://x",
    )
    assert captured == {"connection_id": "neo-conn", "database": "graphdb"}
