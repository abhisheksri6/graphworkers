"""KG-AC-64 (evolve v12 — evidence grounding): a relation's `evidence_text` must occur VERBATIM
(whitespace-normalised comparison; case-sensitive otherwise) in its source chunk text, or the
relation is dropped and counted. Applies to relations from every LLM source (`generate`,
`classify`, and any future mode); deterministic-layer relations (`extractor='rules'`) are EXEMPT —
their evidence is the matched sentence by construction. Strengthens KG-AC-46's evidence-mandatory
rule from *evidence present* to *evidence grounded*. Scope note (from the AC itself): this catches
FABRICATED quotes, not factual misattribution — a verbatim-but-misleading quote still passes.
"""
import pytest

from core import Relation, build_summary
from ontologies import load_pack
from strategies.base import validate_relations

FIBO = load_pack("fibo_core")


def _rel(evidence, *, extractor="llm", chunk_id="c1", relation_type="issues",
        src="Acme Corp", src_type="Organization", dst="Acme 5% 2030", dst_type="Bond"):
    return Relation(relation_type, src, src_type, dst, dst_type, chunk_id,
                    evidence_text=evidence, extractor=extractor)


# ---- core grounding behaviour ----------------------------------------------
@pytest.mark.ac("KG-AC-64")
def test_verbatim_evidence_is_kept():
    rel = _rel("Acme Corp issues Acme 5% 2030.")
    kept, ungrounded = validate_relations([rel], FIBO, {"c1": "Acme Corp issues Acme 5% 2030."})
    assert len(kept) == 1 and ungrounded == 0


@pytest.mark.ac("KG-AC-64")
def test_fabricated_evidence_is_dropped_and_counted():
    rel = _rel("Acme Corp secretly owns Ghost Corp.")
    kept, ungrounded = validate_relations([rel], FIBO, {"c1": "Acme Corp issues Acme 5% 2030."})
    assert kept == [] and ungrounded == 1


@pytest.mark.ac("KG-AC-64")
def test_whitespace_differences_are_normalised():
    rel = _rel("Acme   Corp\nissues Acme 5% 2030.")
    kept, ungrounded = validate_relations([rel], FIBO, {"c1": "Acme Corp issues Acme 5% 2030."})
    assert len(kept) == 1 and ungrounded == 0


@pytest.mark.ac("KG-AC-64")
def test_case_mismatch_is_not_grounded():
    # case-sensitive otherwise, per the AC's own wording
    rel = _rel("acme corp issues acme 5% 2030.")
    kept, ungrounded = validate_relations([rel], FIBO, {"c1": "Acme Corp issues Acme 5% 2030."})
    assert kept == [] and ungrounded == 1


@pytest.mark.ac("KG-AC-64")
def test_deterministic_layer_relations_are_exempt():
    rel = _rel("this sentence does not appear anywhere in the chunk", extractor="rules")
    kept, ungrounded = validate_relations([rel], FIBO, {"c1": "totally different chunk text"})
    assert len(kept) == 1 and ungrounded == 0


@pytest.mark.ac("KG-AC-64")
def test_missing_chunk_text_fails_grounding():
    rel = _rel("Acme Corp issues Acme 5% 2030.", chunk_id="unknown-chunk")
    kept, ungrounded = validate_relations([rel], FIBO, {"c1": "Acme Corp issues Acme 5% 2030."})
    assert kept == [] and ungrounded == 1


@pytest.mark.ac("KG-AC-64")
def test_no_evidence_fails_grounding():
    rel = _rel(None)
    kept, ungrounded = validate_relations([rel], FIBO, {"c1": "Acme Corp issues Acme 5% 2030."})
    assert kept == [] and ungrounded == 1


@pytest.mark.ac("KG-AC-64")
def test_domain_range_and_grounding_are_independent_gates():
    # illegal pairing (a Person cannot issue a Bond per FIBO's domain/range) -> dropped, but NOT
    # counted as ungrounded -- it never reaches the grounding check; the drop reason is domain/range.
    rel = _rel("Jane Roe issues Acme 5% 2030.", src="Jane Roe", src_type="Person")
    kept, ungrounded = validate_relations([rel], FIBO, {"c1": "Jane Roe issues Acme 5% 2030."})
    assert kept == [] and ungrounded == 0


# ---- state-plane reporting --------------------------------------------------
@pytest.mark.ac("KG-AC-64")
def test_build_summary_reports_ungrounded_relation_count():
    summary = build_summary([], [], "fibo_core", "1.0", 0, ungrounded_relation_count=3)
    assert summary["ungrounded_relation_count"] == 3


@pytest.mark.ac("KG-AC-64")
def test_run_pipeline_reports_ungrounded_relations_in_summary():
    from strategies import Chunk, ExtractionConfig, run_pipeline

    class _Client:
        resolved_model = "fake-model"
        usage: list = []

        def complete_tool(self, **_kwargs):
            return {
                "entities": [
                    {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
                    {"type": "Bond", "surface": "Acme 5% 2030", "confidence": 0.8},
                ],
                "relations": [
                    {"type": "issues", "src": "Acme Corp", "src_type": "Organization",
                     "dst": "Acme 5% 2030", "dst_type": "Bond", "confidence": 0.7,
                     "evidence": "Acme Corp secretly issues something else entirely."},
                ],
            }

    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core")
    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [Chunk("c1", "Acme Corp issues Acme 5% 2030.")], cfg, FIBO, folder_id="f1", llm_client=_Client(),
    )
    assert edge_rows == []  # the fabricated relation never becomes an edge
    assert summary["ungrounded_relation_count"] == 1
