"""P3 (spec v13, KG-AC-73): document + page provenance threaded from `chunk_metadata` (already
written by chunking, `{"page": N, "source": {"filename": ...}}`) through to `kg_entities`/
`kg_edges`. Root cause this closes: `read_chunks` built `Chunk(chunk_id, text)` and discarded
`chunk_metadata` wholesale, so `source_doc_id` — present in the schema since the KG tables landed —
was never written and no row could be traced to a document or page.
"""
import pytest

from core import Candidate, Relation, build_edge_records, build_entity_records, build_summary
from strategies.base import Chunk, ExtractionConfig, run_pipeline


# ---- Chunk carries doc_id/page --------------------------------------------
@pytest.mark.ac("KG-AC-73")
def test_chunk_gains_doc_id_and_page_optional_fields():
    ch = Chunk(chunk_id="c1", text="hello", doc_id="Agreement.pdf", page=3)
    assert ch.doc_id == "Agreement.pdf" and ch.page == 3


@pytest.mark.ac("KG-AC-73")
def test_chunk_doc_id_page_default_none():
    # backward compatible -- every existing Chunk(chunk_id, text) construction still works.
    ch = Chunk(chunk_id="c1", text="hello")
    assert ch.doc_id is None and ch.page is None


# ---- build_entity_records writes source_doc_id/page -----------------------
@pytest.mark.ac("KG-AC-73")
def test_build_entity_records_writes_source_doc_id_and_page():
    cand = Candidate(surface_form="Acme Corp", entity_type="Organization", source_chunk_id="c1",
                     layer="llm", span_start=0, span_end=9)
    provenance = {"c1": ("Agreement.pdf", 3)}
    rows = build_entity_records("f1", [cand], "generic", "1.0", chunk_provenance=provenance)
    assert rows[0]["source_doc_id"] == "Agreement.pdf"
    assert rows[0]["page"] == 3


@pytest.mark.ac("KG-AC-73")
def test_build_entity_records_missing_provenance_records_null_not_fabricated():
    cand = Candidate(surface_form="Acme Corp", entity_type="Organization", source_chunk_id="c1",
                     layer="llm", span_start=0, span_end=9)
    rows = build_entity_records("f1", [cand], "generic", "1.0")  # no chunk_provenance at all
    assert rows[0]["source_doc_id"] is None
    assert rows[0]["page"] is None


# ---- build_edge_records writes source_doc_id/page for its evidence --------
@pytest.mark.ac("KG-AC-73")
def test_build_edge_records_writes_source_doc_id_and_page():
    rel = Relation(relation_type="employs", src_surface="Acme Corp", src_type="Organization",
                   dst_surface="Jane Roe", dst_type="Person", source_chunk_id="c1")
    uid_map = {("c1", "Organization", "Acme Corp"): "uid-src", ("c1", "Person", "Jane Roe"): "uid-dst"}
    provenance = {"c1": ("Agreement.pdf", 3)}
    rows = build_edge_records("f1", [rel], uid_map, chunk_provenance=provenance)
    assert rows[0]["source_doc_id"] == "Agreement.pdf"
    assert rows[0]["page"] == 3


# ---- run_pipeline: missing-metadata counting (KG-AC-73's own "counted" clause) ----
class _FakeEntityStrategy:
    layer = "spacy"

    def extract(self, chunks, config, pack):
        return [Candidate(surface_form="Acme Corp", entity_type="Organization",
                          source_chunk_id=chunks[0].chunk_id, layer="spacy", span_start=0, span_end=9)]


class _FakeNlp:
    """A no-op nlp callable — run_pipeline pre-resolves shared_nlp for engine=spacy even when
    run_graph_extraction is mocked out below, so a real (unavailable in this sandbox, no
    SPACY_MODEL_PATH) model load must never be reached. Content is never read by the mock."""
    def __call__(self, text):
        return type("Doc", (), {"ents": [], "sents": []})()


@pytest.mark.ac("KG-AC-73")
def test_run_pipeline_counts_chunks_with_missing_metadata(monkeypatch):
    import strategies.base as base_mod

    def _fake_run_graph_extraction(chunks, config, pack, **kwargs):
        return [Candidate(surface_form="Acme Corp", entity_type="Organization",
                          source_chunk_id=chunks[0].chunk_id, layer="spacy",
                          span_start=0, span_end=9)], []

    monkeypatch.setattr(base_mod, "run_graph_extraction", _fake_run_graph_extraction)

    from ontologies import load_pack
    pack = load_pack("generic")
    chunks = [
        Chunk(chunk_id="c1", text="Acme Corp raised $5M", doc_id="Agreement.pdf", page=1),
        Chunk(chunk_id="c2", text="more text", doc_id=None, page=None),  # missing both
        Chunk(chunk_id="c3", text="more text", doc_id="Agreement.pdf", page=None),  # missing page only
    ]
    config = ExtractionConfig(engine="spacy", ontology_pack="generic")
    _ent, _edge, summary, _usage, _blocked = run_pipeline(
        chunks, config, pack, folder_id="f1", spacy_nlp=_FakeNlp())
    assert summary["chunk_metadata_missing_count"] == 2


@pytest.mark.ac("KG-AC-73")
def test_run_pipeline_zero_missing_when_all_chunks_carry_full_metadata(monkeypatch):
    import strategies.base as base_mod

    def _fake_run_graph_extraction(chunks, config, pack, **kwargs):
        return [], []

    monkeypatch.setattr(base_mod, "run_graph_extraction", _fake_run_graph_extraction)

    from ontologies import load_pack
    pack = load_pack("generic")
    chunks = [Chunk(chunk_id="c1", text="x", doc_id="A.pdf", page=1)]
    config = ExtractionConfig(engine="spacy", ontology_pack="generic")
    _ent, _edge, summary, _usage, _blocked = run_pipeline(
        chunks, config, pack, folder_id="f1", spacy_nlp=_FakeNlp())
    assert summary["chunk_metadata_missing_count"] == 0


# ---- build_summary carries the new scalar, default 0 (backward compatible) ----
@pytest.mark.ac("KG-AC-73")
def test_build_summary_chunk_metadata_missing_count_defaults_zero():
    summary = build_summary([], [], "generic", "1.0", unmapped_type_count=0)
    assert summary["chunk_metadata_missing_count"] == 0


@pytest.mark.ac("KG-AC-73")
def test_build_summary_carries_explicit_chunk_metadata_missing_count():
    summary = build_summary([], [], "generic", "1.0", unmapped_type_count=0,
                            chunk_metadata_missing_count=4)
    assert summary["chunk_metadata_missing_count"] == 4
