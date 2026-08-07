"""KG-AC-33: a folder whose chunks contain no pack-typed entities SUCCEEDS with entity_count=0 (a
valid empty graph) — distinct from the zero-chunks case (KG-AC-7), which fails loud."""
import pytest

from ontologies import load_pack
from strategies import Chunk, ExtractionConfig, run_pipeline


class _FakeLlm:
    resolved_model = "amazon.nova-pro-v1:0"
    usage: list = []

    def complete_tool(self, **_kwargs):
        return {"entities": [], "relations": []}  # model finds nothing pack-typed


@pytest.mark.ac("KG-AC-33")
def test_zero_entities_is_success_empty_graph():
    pack = load_pack("fibo_core")
    ent, edge, summary, usage, blocked = run_pipeline(
        [Chunk("c1", "nothing notable here")],
        ExtractionConfig(engine="llm", ontology_pack="fibo_core"), pack,
        folder_id="f1", llm_client=_FakeLlm(),
    )
    assert ent == [] and edge == []
    assert summary["entity_count"] == 0 and summary["edge_count"] == 0 and summary["distinct_types"] == 0


class _FakeStorage:
    def read_chunks(self, folder_id):
        return [{"chunk_id": "c1", "content": "nothing notable here"}]


class _FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        return None


class _FakeRawConn:
    def cursor(self):
        return _FakeCursor()


class _FakeDb:
    def connection(self):
        return type("_W", (), {"connection": _FakeRawConn()})()


@pytest.mark.ac("KG-AC-33")
def test_process_folder_zero_entities_posts_success():
    posted = {}

    def http_post(url, json=None, timeout=None):
        posted["payload"] = json
        return type("R", (), {"status_code": 200})()

    import entity_extraction_worker as w
    payload = w.process_folder(
        "t", "f1", {"engine": "llm", "ontology_pack": "fibo_core", "connection_id": "c"}, "d", "r",
        storage=_FakeStorage(), db=_FakeDb(), http_post=http_post, worker_results_url="http://x",
        llm_client=_FakeLlm(),
    )
    assert payload["status"] == "success"
    assert payload["entity_count"] == 0
    assert posted["payload"]["status"] == "success"
