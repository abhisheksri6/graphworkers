"""Bug fix (found live 2026-08-11): `entity_extraction_task` constructed `StorageClient` OUTSIDE
`process_folder`'s own try/except, so a setup-phase failure (e.g. a misconfigured
`BLOB_STORAGE_URI` -> `ValueError: StorageClient requires a non-empty blob_uri`) crashed the Celery
task itself with NO callback ever posted -- `process_folder`'s own docstring promise ("ALWAYS posts
a callback... KG-AC-19: any failure here fails only THIS folder's task") does not actually hold for
failures that happen before `process_folder` is ever entered. `classification_worker.py`'s
`classification_task` already wraps its ENTIRE body (including its own `StorageClient(...)`
construction) in one try/except -- this mirrors that existing, working pattern. No new AC: a bug
fix against an already-documented promise, Exempt tier (cb-spec tier rules)."""
import httpx
import pytest

import entity_extraction_worker as eew


class _FakeSession:
    def __enter__(self):
        return "fake-db"

    def __exit__(self, *exc):
        return False


class _FakeResponse:
    status_code = 200
    text = ""


def _raising_storage_client(*args, **kwargs):
    raise ValueError("StorageClient requires a non-empty blob_uri")


@pytest.mark.ac("KG-AC-19")
def test_setup_failure_before_process_folder_still_posts_a_failed_callback(monkeypatch):
    monkeypatch.setattr(eew, "get_worker_session", lambda: _FakeSession())
    monkeypatch.setattr(eew, "StorageClient", _raising_storage_client)
    posted = {}

    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    result = eew.entity_extraction_task("t1", "f1", {"engine": "llm"}, "dag1", "run1")

    assert posted, "no callback was ever posted -- the exact gap this test guards against"
    assert posted["url"] == eew.settings.worker_results_url
    assert posted["payload"]["status"] == "failed"
    assert "blob_uri" in posted["payload"]["error_message"]
    assert result["status"] == "failed"


def test_setup_failure_callback_carries_no_summary_or_usage(monkeypatch):
    # matches process_folder's own failure-branch shape exactly (status/error_message set,
    # summary/usage never populated) -- KG-AC-8's thin-callback contract applies here too.
    monkeypatch.setattr(eew, "get_worker_session", lambda: _FakeSession())
    monkeypatch.setattr(eew, "StorageClient", _raising_storage_client)
    posted = {}
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None:
                        (posted.update({"payload": json}), _FakeResponse())[1])

    eew.entity_extraction_task("t1", "f1", {"engine": "llm"}, "dag1", "run1")

    payload = posted["payload"]
    assert "entities" not in payload and "edges" not in payload
    assert payload.get("usage") in (None, [])


def test_setup_failure_is_logged(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(eew, "get_worker_session", lambda: _FakeSession())
    monkeypatch.setattr(eew, "StorageClient", _raising_storage_client)
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: _FakeResponse())

    with caplog.at_level(logging.ERROR):
        eew.entity_extraction_task("t1", "f1", {"engine": "llm"}, "dag1", "run1")

    assert any("f1" in r.getMessage() for r in caplog.records)
