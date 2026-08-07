"""KG-AC-61 (closed-set classification) + KG-AC-62 (batched cost bound) — evolve v10, ADR-0014.

Mirrors llm_graph.py's test approach: a fake client stands in for clients.BedrockLlmClient's
complete_tool(...) -> Dict[str, Any] contract. LlmOutputError propagates uncaught (KG-AC-P3's
one-retry-then-fail already lives INSIDE the real client; this strategy never catches it, same
posture as LlmGraphStrategy)."""
import pytest

from candidate_pairs import CandidatePair
from clients import LlmOutputError
from core import Relation
from strategies.llm_classify import LlmClassifyStrategy


def _pair(chunk_id="c1", src="Acme Corp", src_type="Organization", dst="Jane Roe",
         dst_type="Person", allowed=("employs",), sentence_span=(0, 30)):
    return CandidatePair(
        chunk_id=chunk_id, src_surface=src, src_type=src_type, src_span=(0, 9),
        dst_surface=dst, dst_type=dst_type, dst_span=(20, 28),
        sentence_span=sentence_span, allowed_relation_types=list(allowed),
    )


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.last_user_text = None

    def complete_tool(self, *, system_text, user_text, tool_name, tool_description, tool_schema):
        self.calls += 1
        self.last_user_text = user_text
        return self._responses.pop(0)


@pytest.mark.ac("KG-AC-61")
def test_pair_labeled_with_allowed_relation_is_kept():
    client = _FakeClient([{"labels": [
        {"pair_index": 0, "relation_type": "employs", "evidence": "Acme Corp employs Jane Roe."}]}])
    strat = LlmClassifyStrategy(llm_client=client)
    relations = strat.classify([_pair()], {"c1": "Acme Corp employs Jane Roe. Other text."})
    assert len(relations) == 1
    r = relations[0]
    assert isinstance(r, Relation)
    assert r.relation_type == "employs"
    assert r.src_surface == "Acme Corp" and r.src_type == "Organization"
    assert r.dst_surface == "Jane Roe" and r.dst_type == "Person"
    assert r.evidence_text == "Acme Corp employs Jane Roe."
    assert r.extractor == "llm-classify"
    assert client.calls == 1


@pytest.mark.ac("KG-AC-61")
def test_no_relation_label_dropped():
    client = _FakeClient([{"labels": [{"pair_index": 0, "relation_type": "no_relation"}]}])
    strat = LlmClassifyStrategy(llm_client=client)
    relations = strat.classify([_pair()], {"c1": "text"})
    assert relations == []


@pytest.mark.ac("KG-AC-61")
def test_illegal_relation_type_for_pair_dropped():
    # the model picked a relation_type that is NOT in THIS pair's own allowed set (e.g. legal for a
    # different pair in the same batch, but not this one) -- domain/range illegal, dropped.
    client = _FakeClient([{"labels": [
        {"pair_index": 0, "relation_type": "subsidiaryOf", "evidence": "some sentence"}]}])
    strat = LlmClassifyStrategy(llm_client=client)
    relations = strat.classify([_pair(allowed=("employs",))], {"c1": "text"})
    assert relations == []


@pytest.mark.ac("KG-AC-61")
def test_missing_evidence_dropped():
    # KG-AC-46: evidence is mandatory -- a kept-looking label with no evidence is dropped.
    client = _FakeClient([{"labels": [{"pair_index": 0, "relation_type": "employs"}]}])
    strat = LlmClassifyStrategy(llm_client=client)
    relations = strat.classify([_pair()], {"c1": "text"})
    assert relations == []


@pytest.mark.ac("KG-AC-62")
def test_batching_ceil_pairs_over_n():
    pairs = [_pair(dst=f"Person{i}") for i in range(3)]  # 3 pairs, batch_size=2 -> ceil(3/2)=2 calls
    client = _FakeClient([
        {"labels": [{"pair_index": 0, "relation_type": "employs", "evidence": "e0"}]},
        {"labels": [{"pair_index": 0, "relation_type": "employs", "evidence": "e2"}]},
    ])
    strat = LlmClassifyStrategy(llm_client=client, batch_size=2)
    relations = strat.classify(pairs, {"c1": "text"})
    assert client.calls == 2
    assert len(relations) == 2  # one label per call in this fixture


@pytest.mark.ac("KG-AC-61")
def test_malformed_tool_use_propagates():
    class _FailingClient:
        def complete_tool(self, **_kwargs):
            raise LlmOutputError("schema validation failed after one retry")

    strat = LlmClassifyStrategy(llm_client=_FailingClient())
    with pytest.raises(LlmOutputError):
        strat.classify([_pair()], {"c1": "text"})


@pytest.mark.ac("KG-AC-61")
def test_zero_pairs_returns_no_relations_without_calling_client():
    class _AssertNeverCalled:
        def complete_tool(self, **_kwargs):
            raise AssertionError("must not be called for 0 candidate pairs (KG-AC-61/KG-AC-33 posture)")

    strat = LlmClassifyStrategy(llm_client=_AssertNeverCalled())
    relations = strat.classify([], {})
    assert relations == []
