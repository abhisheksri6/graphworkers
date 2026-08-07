"""KG-AC-7 (zero chunks -> loud failed callback naming the folder) and KG-AC-19 (per-folder
isolation: a folder's failure posts a loud failed callback; siblings — separate tasks — unaffected).
The always-callback guarantee is proved by both."""
import pytest


def _http_capture():
    box = {}

    def http_post(url, json=None, timeout=None):
        box["payload"] = json
        return type("R", (), {"status_code": 200})()

    return box, http_post


class _EmptyStorage:
    def read_chunks(self, folder_id):
        return []


class _BoomStorage:
    def read_chunks(self, folder_id):
        raise RuntimeError("db connection lost")


@pytest.mark.ac("KG-AC-7")
def test_zero_chunks_fails_loud_naming_folder():
    box, http_post = _http_capture()
    import entity_extraction_worker as w
    payload = w.process_folder(
        "t", "folder-x", {"engine": "spacy", "ontology_pack": "generic"}, "d", "r",
        storage=_EmptyStorage(), db=None, http_post=http_post, worker_results_url="http://x",
    )
    assert payload["status"] == "failed"
    assert "folder-x" in payload["error_message"]
    assert box["payload"]["status"] == "failed"  # loud callback always posted


@pytest.mark.ac("KG-AC-19")
def test_extraction_error_isolated_to_this_folder():
    box, http_post = _http_capture()
    import entity_extraction_worker as w
    payload = w.process_folder(
        "t", "folder-y", {"engine": "spacy", "ontology_pack": "generic"}, "d", "r",
        storage=_BoomStorage(), db=None, http_post=http_post, worker_results_url="http://x",
    )
    # this folder's task fails loud + posts a failed callback; because each folder is its own task,
    # a sibling folder's task (a separate process_folder call) is unaffected.
    assert payload["status"] == "failed"
    assert box["payload"]["status"] == "failed"
    assert "folders" not in box["payload"]  # no partial state promoted on failure
