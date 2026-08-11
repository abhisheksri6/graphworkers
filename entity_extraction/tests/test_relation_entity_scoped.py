"""KG-AC-65 (evolve v12 — entity-scoped relation mode, ADR-0014): one LLM call per chunk over the
chunk's full text + merged typed entity list, no candidate-pair enumeration, no same-sentence gate
— so relations whose endpoints sit in DIFFERENT sentences of the same chunk are reachable, unlike
`classify`'s same-sentence candidate rule (KG-AC-60). A chunk with fewer than 2 entities makes no
LLM call (KG-AC-33 empty-result posture). Mirrors llm_classify.py's test approach: a fake client
stands in for clients.BedrockLlmClient's complete_tool(...) -> Dict[str, Any] contract.
"""
import pytest

from clients import LlmOutputError
from core import Candidate, Relation
from ontologies import load_pack
from strategies.llm_entity_scoped import LlmEntityScopedStrategy
from strategies.llm_graph import LlmConnectionError

FIBO = load_pack("fibo_core")


def _ent(surface, etype, chunk_id="c1"):
    return Candidate(surface_form=surface, entity_type=etype, source_chunk_id=chunk_id, layer="llm")


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.last_user_text = None

    def complete_tool(self, *, system_text, user_text, tool_name, tool_description, tool_schema):
        self.calls += 1
        self.last_user_text = user_text
        return self._responses.pop(0)


@pytest.mark.ac("KG-AC-65")
def test_relation_between_cross_sentence_entities_is_returned():
    # "Acme Corp" and "Jane Roe" are in DIFFERENT sentences -- classify's same-sentence candidate
    # rule (KG-AC-60) could never pair them; entity_scoped has no such gate.
    text = "Acme Corp was founded in 1990. Jane Roe joined as CEO in 2020."
    client = _FakeClient([{"relations": [
        {"type": "employs", "src_id": 0, "dst_id": 1, "confidence": 0.8,
         "evidence": "Jane Roe joined as CEO in 2020."},
    ]}])
    strat = LlmEntityScopedStrategy(llm_client=client)
    relations = strat.extract(
        [_ent("Acme Corp", "Organization"), _ent("Jane Roe", "Person")], {"c1": text}, FIBO,
    )
    assert len(relations) == 1
    r = relations[0]
    assert isinstance(r, Relation)
    assert r.relation_type == "employs"
    assert r.src_surface == "Acme Corp" and r.src_type == "Organization"
    assert r.dst_surface == "Jane Roe" and r.dst_type == "Person"
    assert r.evidence_text == "Jane Roe joined as CEO in 2020."
    assert r.extractor == "llm-entity-scoped"
    assert client.calls == 1
    # the user prompt names BOTH entities, not a candidate pair
    assert "Acme Corp" in client.last_user_text and "Jane Roe" in client.last_user_text


@pytest.mark.ac("KG-AC-65")
def test_missing_evidence_dropped():
    client = _FakeClient([{"relations": [
        {"type": "employs", "src_id": 0, "dst_id": 1},  # no 'evidence' key
    ]}])
    strat = LlmEntityScopedStrategy(llm_client=client)
    relations = strat.extract(
        [_ent("Acme Corp", "Organization"), _ent("Jane Roe", "Person")], {"c1": "text"}, FIBO,
    )
    assert relations == []


@pytest.mark.ac("KG-AC-65")
def test_no_relation_type_dropped():
    client = _FakeClient([{"relations": [
        {"src_id": 0, "dst_id": 1, "evidence": "text"},  # no 'type' key
    ]}])
    strat = LlmEntityScopedStrategy(llm_client=client)
    relations = strat.extract(
        [_ent("Acme Corp", "Organization"), _ent("Jane Roe", "Person")], {"c1": "text"}, FIBO,
    )
    assert relations == []


@pytest.mark.ac("KG-AC-65")
def test_fewer_than_two_entities_in_a_chunk_skips_the_call():
    # c1 has 1 entity (no possible relation); c2 has 2 -> only c2 gets a call.
    client = _FakeClient([{"relations": []}])
    strat = LlmEntityScopedStrategy(llm_client=client)
    entities = [
        _ent("Solo Corp", "Organization", chunk_id="c1"),
        _ent("Acme Corp", "Organization", chunk_id="c2"),
        _ent("Jane Roe", "Person", chunk_id="c2"),
    ]
    strat.extract(entities, {"c1": "Solo Corp exists.", "c2": "Acme Corp employs Jane Roe."}, FIBO)
    assert client.calls == 1  # c1's single entity never triggered a call


@pytest.mark.ac("KG-AC-65")
def test_zero_entities_returns_no_relations_without_calling_client():
    class _AssertNeverCalled:
        def complete_tool(self, **_kwargs):
            raise AssertionError("must not be called for <2 entities (KG-AC-65/KG-AC-33 posture)")

    strat = LlmEntityScopedStrategy(llm_client=_AssertNeverCalled())
    assert strat.extract([], {}, FIBO) == []


@pytest.mark.ac("KG-AC-65")
def test_single_entity_returns_no_relations_without_calling_client():
    class _AssertNeverCalled:
        def complete_tool(self, **_kwargs):
            raise AssertionError("must not be called for <2 entities (KG-AC-65/KG-AC-33 posture)")

    strat = LlmEntityScopedStrategy(llm_client=_AssertNeverCalled())
    relations = strat.extract([_ent("Solo Corp", "Organization")], {"c1": "Solo Corp exists."}, FIBO)
    assert relations == []


@pytest.mark.ac("KG-AC-65")
def test_multiple_chunks_each_get_their_own_call_with_only_their_own_entities():
    client = _FakeClient([{"relations": []}, {"relations": []}])
    strat = LlmEntityScopedStrategy(llm_client=client)
    entities = [
        _ent("Acme Corp", "Organization", chunk_id="c1"),
        _ent("Jane Roe", "Person", chunk_id="c1"),
        _ent("Globex", "Organization", chunk_id="c2"),
        _ent("John Doe", "Person", chunk_id="c2"),
    ]
    strat.extract(entities, {"c1": "Acme text", "c2": "Globex text"}, FIBO)
    assert client.calls == 2


@pytest.mark.ac("KG-AC-65")
def test_no_client_raises_when_a_call_is_actually_needed():
    strat = LlmEntityScopedStrategy(llm_client=None)
    with pytest.raises(LlmConnectionError):
        strat.extract(
            [_ent("Acme Corp", "Organization"), _ent("Jane Roe", "Person")], {"c1": "text"}, FIBO,
        )


@pytest.mark.ac("KG-AC-65")
def test_malformed_tool_use_propagates():
    class _FailingClient:
        def complete_tool(self, **_kwargs):
            raise LlmOutputError("schema validation failed after one retry")

    strat = LlmEntityScopedStrategy(llm_client=_FailingClient())
    with pytest.raises(LlmOutputError):
        strat.extract(
            [_ent("Acme Corp", "Organization"), _ent("Jane Roe", "Person")], {"c1": "text"}, FIBO,
        )


# ---- end-to-end via run_pipeline -------------------------------------------------------------
@pytest.mark.ac("KG-AC-65")
def test_run_pipeline_entity_scoped_end_to_end():
    from strategies.base import Chunk, ExtractionConfig, run_pipeline

    class _Client:
        resolved_model = "fake-model"
        usage: list = []

        def __init__(self):
            self.calls = []

        def complete_tool(self, *, tool_name, **_kwargs):
            self.calls.append(tool_name)
            if tool_name == "extract_graph":
                return {"entities": [
                    {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
                    {"type": "Person", "surface": "Jane Roe", "confidence": 0.8},
                ], "relations": []}  # generate-mode relations intentionally absent
            if tool_name == "extract_relations_for_entities":
                return {"relations": [
                    {"type": "employs", "src_id": 0, "dst_id": 1, "confidence": 0.8,
                     "evidence": "Jane Roe joined Acme Corp."}]}
            raise AssertionError(f"unexpected tool_name {tool_name!r}")

    client = _Client()
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", relation_strategy="entity_scoped")
    ent_rows, edge_rows, summary, _usage, _blocked = run_pipeline(
        [Chunk("c1", "Jane Roe joined Acme Corp.")], cfg, FIBO, folder_id="f1", llm_client=client,
    )
    assert client.calls == ["extract_graph", "extract_relations_for_entities"]
    assert len(ent_rows) == 2
    assert len(edge_rows) == 1
    assert edge_rows[0]["relation_type"] == "employs"
    assert edge_rows[0]["extractor"] == "llm-entity-scoped"
    assert edge_rows[0]["evidence_text"] == "Jane Roe joined Acme Corp."
    assert summary["ungrounded_relation_count"] == 0  # grounded via N1's validator


@pytest.mark.ac("KG-AC-65")
def test_run_pipeline_entity_scoped_drops_ungrounded_and_illegal_relations():
    from strategies.base import Chunk, ExtractionConfig, run_pipeline

    class _Client:
        resolved_model = "fake-model"
        usage: list = []

        def __init__(self):
            self.calls = []

        def complete_tool(self, *, tool_name, **_kwargs):
            self.calls.append(tool_name)
            if tool_name == "extract_graph":
                return {"entities": [
                    {"type": "Person", "surface": "Jane Roe", "confidence": 0.9},
                    {"type": "Bond", "surface": "Acme 5% 2030", "confidence": 0.8},
                ], "relations": []}
            if tool_name == "extract_relations_for_entities":
                return {"relations": [
                    # illegal domain/range: a Person cannot "issue" a Bond per FIBO -- the id itself
                    # resolves fine (isolating this as a domain/range drop, not an unresolved one)
                    {"type": "issues", "src_id": 0, "dst_id": 1, "confidence": 0.5,
                     "evidence": "Jane Roe issues Acme 5% 2030."},
                ]}
            raise AssertionError(tool_name)

    client = _Client()
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", relation_strategy="entity_scoped")
    ent_rows, edge_rows, summary, _usage, _blocked = run_pipeline(
        [Chunk("c1", "Jane Roe issues Acme 5% 2030.")], cfg, FIBO, folder_id="f1", llm_client=client,
    )
    # prove entity_scoped was actually INVOKED (not just vacuously empty because it never ran)
    assert client.calls == ["extract_graph", "extract_relations_for_entities"]
    assert edge_rows == []  # illegal domain/range, dropped
    assert summary["ungrounded_relation_count"] == 0  # dropped for domain/range, NOT grounding


@pytest.mark.ac("KG-AC-65")
def test_dep_matcher_layer_still_runs_alongside_entity_scoped():
    # ADR-0012: the deterministic DependencyMatcher layer runs alongside ANY relation mode.
    import spacy
    from spacy.tokens import Doc

    from strategies.base import Chunk, ExtractionConfig, run_pipeline

    class _FixedDocNlp:
        def __init__(self, real_blank_nlp, doc):
            self._doc = doc
            self.vocab = real_blank_nlp.vocab
            self.pipe_names = ["parser"]

        def __call__(self, text):
            return self._doc

    class _Client:
        resolved_model = "fake-model"
        usage: list = []

        def __init__(self):
            self.calls = []

        def complete_tool(self, *, tool_name, **_kwargs):
            self.calls.append(tool_name)
            if tool_name == "extract_graph":
                return {"entities": [
                    {"type": "Organization", "surface": "Acme", "confidence": 0.9},
                    {"type": "Person", "surface": "Jane", "confidence": 0.8},
                ], "relations": []}
            if tool_name == "extract_relations_for_entities":
                return {"relations": []}  # the LLM itself finds nothing this run
            raise AssertionError(tool_name)

    blank_nlp = spacy.blank("en")
    doc = Doc(blank_nlp.vocab, words=["Acme", "employed", "Jane", "."], heads=[1, 1, 1, 1],
              deps=["nsubj", "ROOT", "dobj", "punct"], lemmas=["Acme", "employ", "Jane", "."])
    client = _Client()
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", relation_strategy="entity_scoped",
                           rules_relations_enabled=True)
    ent_rows, edge_rows, _summary, _usage, _blocked = run_pipeline(
        [Chunk("c1", "Acme employed Jane.")], cfg, FIBO, folder_id="f1",
        llm_client=client, spacy_nlp=_FixedDocNlp(blank_nlp, doc),
    )
    # prove entity_scoped was actually INVOKED alongside the dep-matcher, not skipped
    assert client.calls == ["extract_graph", "extract_relations_for_entities"]
    assert len(edge_rows) == 1
    assert edge_rows[0]["relation_type"] == "employs" and edge_rows[0]["extractor"] == "rules"
