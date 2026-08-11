"""KG-AC-67 (evolve v12 — self-consistency voting): `relation_self_consistency_k` > 1 (default 1 =
off, opt-in) runs the ACTIVE LLM relation mode k times over the same input; only relations
appearing in >= ceil(k/2) runs are kept, each surviving relation's `confidence` set to its vote
fraction. Deterministic layers (rules) run ONCE and are exempt. A hard failure on ANY of the k
runs fails the whole folder (never partial-vote degradation, matching every other LLM-failure
path in this spec). Structural finding recorded at cb-plan: under `generate`, the SAME LLM call
produces entities AND relations — so k-repeat calls that call re-run entity extraction too, but
only run 1's entities are ever used; runs 2..k's entities are discarded, never affecting the
graph.
"""
import pytest

from clients import LlmOutputError
from ontologies import load_pack
from strategies.base import Chunk, ExtractionConfig, run_pipeline

FIBO = load_pack("fibo_core")


class _FakeLlmClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.resolved_model = "fake-model"
        self.usage = []

    def complete_tool(self, *, system_text, user_text, tool_name, tool_description, tool_schema):
        self.calls += 1
        return self._responses.pop(0)


_ONE_RESPONSE = {
    "entities": [
        {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
        {"type": "Bond", "surface": "Acme 5% 2030", "confidence": 0.8},
    ],
    "relations": [
        {"type": "issues", "src_id": 0, "dst_id": 1, "confidence": 0.7,
         "evidence": "Acme Corp issues Acme 5% 2030."},
    ],
}


@pytest.mark.ac("KG-AC-67")
def test_default_k_is_1():
    assert ExtractionConfig().relation_self_consistency_k == 1


@pytest.mark.ac("KG-AC-67")
def test_k1_is_a_noop_exactly_one_call_confidence_unchanged():
    # only ONE response queued -- a second call would raise IndexError (pop from empty)
    client = _FakeLlmClient([_ONE_RESPONSE])
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", relation_self_consistency_k=1)
    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [Chunk("c1", "Acme Corp issues Acme 5% 2030.")], cfg, FIBO, folder_id="f1", llm_client=client,
    )
    assert client.calls == 1
    assert len(edge_rows) == 1
    assert edge_rows[0]["confidence"] == 0.7  # the LLM's OWN reported confidence, untouched
    assert summary["self_consistency_votes"] == 1


@pytest.mark.ac("KG-AC-67")
def test_k3_majority_voting_and_vote_fraction_confidence():
    chunk_text = "Acme Corp issues Acme 5% 2030. It also issues Acme 7% 2035."
    entities = [
        {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
        {"type": "Bond", "surface": "Acme 5% 2030", "confidence": 0.8},
        {"type": "Bond", "surface": "Acme 7% 2035", "confidence": 0.8},
    ]
    majority_rel = {"type": "issues", "src_id": 0, "dst_id": 1,
                    "evidence": "Acme Corp issues Acme 5% 2030."}
    minority_rel = {"type": "issues", "src_id": 0, "dst_id": 2,
                    "evidence": "It also issues Acme 7% 2035."}
    responses = [
        {"entities": entities, "relations": [dict(majority_rel, confidence=0.7), dict(minority_rel, confidence=0.5)]},
        {"entities": entities, "relations": [dict(majority_rel, confidence=0.6)]},  # minority absent
        {"entities": entities, "relations": [dict(majority_rel, confidence=0.8)]},  # minority absent
    ]
    client = _FakeLlmClient(responses)
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", relation_self_consistency_k=3)
    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [Chunk("c1", chunk_text)], cfg, FIBO, folder_id="f1", llm_client=client,
    )
    assert client.calls == 3
    assert len(edge_rows) == 1  # only the 3/3 relation survives -- 1/3 falls below ceil(3/2)=2
    assert edge_rows[0]["dst_entity_uid"]  # sanity: it resolved to the "Acme 5% 2030" bond
    assert edge_rows[0]["confidence"] == pytest.approx(1.0)  # 3 votes / k=3
    assert summary["self_consistency_votes"] == 3


@pytest.mark.ac("KG-AC-67")
def test_hard_failure_on_any_run_fails_the_whole_folder():
    class _FailsOnSecondCall:
        resolved_model = "fake-model"
        usage: list = []

        def __init__(self):
            self.calls = 0

        def complete_tool(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return _ONE_RESPONSE
            raise LlmOutputError("boom on run 2 -- never a partial-vote degradation")

    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", relation_self_consistency_k=3)
    with pytest.raises(LlmOutputError):
        run_pipeline(
            [Chunk("c1", "Acme Corp issues Acme 5% 2030.")], cfg, FIBO, folder_id="f1",
            llm_client=_FailsOnSecondCall(),
        )


@pytest.mark.ac("KG-AC-67")
def test_generate_mode_entities_come_from_run_1_only():
    class _VaryingClient:
        resolved_model = "fake-model"
        usage: list = []

        def __init__(self):
            self.calls = 0

        def complete_tool(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"entities": [{"type": "Organization", "surface": "Acme Corp", "confidence": 0.9}],
                        "relations": []}
            # runs 2/3 "hallucinate" a DIFFERENT entity -- it must NEVER reach the written graph
            return {"entities": [{"type": "Organization", "surface": "Ghost Corp", "confidence": 0.9}],
                    "relations": []}

    client = _VaryingClient()
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", relation_self_consistency_k=3)
    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [Chunk("c1", "Acme Corp exists.")], cfg, FIBO, folder_id="f1", llm_client=client,
    )
    assert client.calls == 3  # all k runs still happened (cost scales linearly, per the AC)
    assert len(ent_rows) == 1
    assert ent_rows[0]["surface_form"] == "Acme Corp"  # NOT "Ghost Corp" from runs 2/3


@pytest.mark.ac("KG-AC-67")
def test_deterministic_rules_layer_runs_once_not_k_times():
    import spacy
    from spacy.tokens import Doc

    class _CountingNlp:
        def __init__(self, real_blank_nlp, doc):
            self._doc = doc
            self.vocab = real_blank_nlp.vocab
            self.pipe_names = ["parser"]
            self.calls = 0

        def __call__(self, text):
            self.calls += 1
            return self._doc

    blank_nlp = spacy.blank("en")
    doc = Doc(blank_nlp.vocab, words=["Acme", "employed", "Jane", "."], heads=[1, 1, 1, 1],
              deps=["nsubj", "ROOT", "dobj", "punct"], lemmas=["Acme", "employ", "Jane", "."])
    nlp = _CountingNlp(blank_nlp, doc)

    llm_client = _FakeLlmClient([
        {"entities": [{"type": "Organization", "surface": "Acme", "confidence": 0.9},
                      {"type": "Person", "surface": "Jane", "confidence": 0.8}], "relations": []},
        {"entities": [], "relations": []},
        {"entities": [], "relations": []},
    ])
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", rules_relations_enabled=True,
                           relation_self_consistency_k=3)
    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [Chunk("c1", "Acme employed Jane.")], cfg, FIBO, folder_id="f1",
        llm_client=llm_client, spacy_nlp=nlp,
    )
    assert llm_client.calls == 3  # the LLM mode still ran k times
    assert nlp.calls == 1  # the dep-matcher's parse ran exactly ONCE, not k times
    assert len(edge_rows) == 1
    assert edge_rows[0]["relation_type"] == "employs" and edge_rows[0]["extractor"] == "rules"


@pytest.mark.ac("KG-AC-67")
def test_k_is_bounded_to_5():
    # a config carrying an out-of-range k is defensively clamped worker-side (the authoritative
    # 1<=k<=5 rejection is backend save-time validation, N5 -- this is defense-in-depth).
    client = _FakeLlmClient([_ONE_RESPONSE] * 5)
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", relation_self_consistency_k=99)
    _ent, _edge, summary, _usage, _blocked = run_pipeline(
        [Chunk("c1", "Acme Corp issues Acme 5% 2030.")], cfg, FIBO, folder_id="f1", llm_client=client,
    )
    assert client.calls == 5
    assert summary["self_consistency_votes"] == 5


@pytest.mark.ac("KG-AC-67")
def test_k_below_1_is_clamped_to_1():
    client = _FakeLlmClient([_ONE_RESPONSE])
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", relation_self_consistency_k=0)
    _ent, _edge, summary, _usage, _blocked = run_pipeline(
        [Chunk("c1", "Acme Corp issues Acme 5% 2030.")], cfg, FIBO, folder_id="f1", llm_client=client,
    )
    assert client.calls == 1
    assert summary["self_consistency_votes"] == 1


# ---- vote_relations (core.py) — pure unit tests ---------------------------------------------
@pytest.mark.ac("KG-AC-67")
def test_vote_relations_pure_function():
    from core import Relation, vote_relations

    def _r(dst, evidence):
        return Relation("issues", "Acme Corp", "Organization", dst, "Bond", "c1", evidence_text=evidence)

    runs = [
        [_r("A", "e1"), _r("B", "e2")],
        [_r("A", "e1b")],
        [_r("A", "e1c")],
    ]
    kept = vote_relations(runs, k=3)
    assert len(kept) == 1
    assert kept[0].dst_surface == "A"
    assert kept[0].confidence == pytest.approx(1.0)


@pytest.mark.ac("KG-AC-67")
def test_vote_relations_duplicate_within_one_run_counts_once():
    from core import Relation, vote_relations

    def _r():
        return Relation("issues", "Acme Corp", "Organization", "A", "Bond", "c1", evidence_text="e")

    # run 1 asserts the SAME triple twice -- must count as ONE vote, not two
    runs = [[_r(), _r()], [_r()]]
    kept = vote_relations(runs, k=2)
    assert len(kept) == 1
    assert kept[0].confidence == pytest.approx(1.0)  # 2/2, not 3/2
