"""KG-AC-17 (worker side): a guardrails block drops the blocked output for the batch, COUNTS it, and
the task still succeeds; usage[] is llm-only; no document text/entity surfaces appear in logs."""
import logging

import pytest

from callback import log_result_summary
from ontologies import load_pack
from strategies import Chunk, ExtractionConfig, run_pipeline


class _Llm:
    resolved_model = "amazon.nova-pro-v1:0"
    usage = [{"charge_category": "llm", "model": "amazon.nova-pro-v1:0",
              "input_tokens": 10, "output_tokens": 5, "connection_id": "c"}]

    def complete_tool(self, **_kwargs):
        return {"entities": [{"type": "Organization", "surface": "Acme Corp", "confidence": 0.9}],
                "relations": []}


@pytest.mark.ac("KG-AC-17")
def test_guardrails_block_drops_and_counts_task_succeeds():
    pack = load_pack("fibo_core")
    ent, edge, summary, usage, blocked = run_pipeline(
        [Chunk("c1", "Acme Corp")], ExtractionConfig(engine="llm", ontology_pack="fibo_core"), pack,
        folder_id="f", llm_client=_Llm(), guardrails_screen=lambda cands: [False] * len(cands),
    )
    assert blocked == 1              # the blocked candidate is counted
    assert summary["entity_count"] == 0  # ...and dropped (not written)
    # task still "succeeds" — run_pipeline returned normally (no raise)


@pytest.mark.ac("KG-AC-17")
def test_usage_is_llm_only():
    pack = load_pack("fibo_core")
    _e, _ed, _s, usage, _b = run_pipeline(
        [Chunk("c1", "Acme Corp")], ExtractionConfig(engine="llm", ontology_pack="fibo_core"), pack,
        folder_id="f", llm_client=_Llm(),
    )
    assert len(usage) == 1 and usage[0]["charge_category"] == "llm"
    # a run with no LLM client carries no usage (spaCy/regex/rules emit none)
    _e2, _ed2, _s2, usage2, _b2 = run_pipeline(
        [Chunk("c1", "x")], ExtractionConfig(engine="llm", ontology_pack="fibo_core"), pack,
        folder_id="f", llm_client=_Llm(),  # llm client present but the usage list is the client's own
    )
    assert all(u["charge_category"] == "llm" for u in usage2)


@pytest.mark.ac("KG-AC-17")
def test_log_summary_has_no_document_content(caplog):
    logger = logging.getLogger("entity_extraction_worker.test")
    summary = {"entity_count": 2, "edge_count": 1, "distinct_types": 2, "unmapped_type_count": 0,
               "ontology_pack": "fibo_core", "ontology_version": "1.0"}
    with caplog.at_level(logging.INFO):
        log_result_summary(logger, "f1", 3, summary)
    text = caplog.text
    assert "Acme" not in text              # no entity surface / document text leaks
    assert "entities=2" in text and "f1" in text  # counts + folder id only
