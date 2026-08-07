"""KG-AC-7 (read side): the worker reads chunks via StorageClient.read_chunks(folder_id), mapping
the chunk `content` to the extraction text. The zero-chunks -> loud-failure behavior is exercised at
the worker level (test_failure_isolation / worker); here we prove the read mapping + the empty case."""
import pytest

from store import read_chunks


class _FakeStorage:
    def __init__(self, rows):
        self._rows = rows

    def read_chunks(self, folder_id):
        return list(self._rows)


@pytest.mark.ac("KG-AC-7")
def test_read_chunks_maps_content_to_text():
    st = _FakeStorage([
        {"chunk_id": "c1", "content": "Acme Corp raised $5M"},
        {"chunk_id": "c2", "content": "Jane Roe is the CEO"},
    ])
    chunks = read_chunks(st, "folder-1")
    assert [c.chunk_id for c in chunks] == ["c1", "c2"]
    assert chunks[0].text == "Acme Corp raised $5M"


@pytest.mark.ac("KG-AC-7")
def test_read_chunks_empty_folder_returns_empty():
    assert read_chunks(_FakeStorage([]), "folder-empty") == []
