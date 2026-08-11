"""KG-AC-48 (evolve v5 — document-scoped coreference, opt-in): with `coreference_enabled=true` and
`engine=llm`, anaphoric mentions are resolved to their antecedent so relations expressed through them
attach to it; an unresolved anaphor is dropped, never written as a bare-pronoun entity;
`coreference_enabled=false` (default) leaves today's per-chunk behavior untouched (no coref call at
all)."""
import pytest

from core import Candidate, filter_bare_pronouns
from ontologies import load_pack
from strategies import Chunk, ExtractionConfig, run_pipeline
from strategies.coref import build_coref_prompt, resolve_coreferences
from strategies.llm_graph import LlmConnectionError

FIBO = load_pack("fibo_core")


class _FakeLlmClient:
    """Returns queued responses in call order (shared across complete() and complete_tool() --
    coref rewrite calls use complete() with a STRING queued; graph extraction uses complete_tool()
    with a DICT queued, evolve v6). Records every prompt/user_text it was called with, in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
        self.resolved_model = "fake-model"
        self.usage = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def complete_tool(self, *, system_text, user_text, tool_name, tool_description, tool_schema):
        self.prompts.append(user_text)
        return self._responses.pop(0)


# ---- resolve_coreferences (pure-ish, LLM injected) ------------------------
@pytest.mark.ac("KG-AC-48")
def test_resolve_coreferences_rewrites_chunk_text_preserving_chunk_id():
    client = _FakeLlmClient(["Acme Corp is a bank.", "Acme Corp employs Jane Roe."])
    chunks = [Chunk("c1", "Acme Corp is a bank."), Chunk("c2", "It employs Jane Roe.")]
    resolved = resolve_coreferences(chunks, client)
    assert [c.chunk_id for c in resolved] == ["c1", "c2"]
    assert resolved[1].text == "Acme Corp employs Jane Roe."
    # the ORIGINAL chunks are untouched (a new list is returned)
    assert chunks[1].text == "It employs Jane Roe."


@pytest.mark.ac("KG-AC-48")
def test_resolve_coreferences_gives_later_chunks_prior_context():
    client = _FakeLlmClient(["Acme Corp is a bank.", "Acme Corp employs Jane Roe."])
    chunks = [Chunk("c1", "Acme Corp is a bank."), Chunk("c2", "It employs Jane Roe.")]
    resolve_coreferences(chunks, client)
    assert len(client.prompts) == 2
    # the 2nd call's prompt carries the 1st chunk's ORIGINAL text as antecedent context
    assert "Acme Corp is a bank." in client.prompts[1]


@pytest.mark.ac("KG-AC-48")
def test_resolve_coreferences_requires_llm_client():
    with pytest.raises(LlmConnectionError):
        resolve_coreferences([Chunk("c1", "text")], None)


@pytest.mark.ac("KG-AC-48")
def test_resolve_coreferences_empty_rewrite_falls_back_to_original():
    client = _FakeLlmClient(["   "])  # degenerate empty rewrite
    chunks = [Chunk("c1", "Acme Corp is a bank.")]
    resolved = resolve_coreferences(chunks, client)
    assert resolved[0].text == "Acme Corp is a bank."  # falls back, never loses content


# ---- filter_bare_pronouns (pure, core.py) ----------------------------------
@pytest.mark.ac("KG-AC-48")
def test_filter_bare_pronouns_drops_pronoun_surfaces_case_insensitive():
    cands = [
        Candidate("Acme Corp", "Organization", "c1", "llm"),
        Candidate("It", "Organization", "c1", "llm"),
        Candidate("the Company", "Organization", "c1", "llm"),
        Candidate("Jane Roe", "Person", "c1", "llm"),
        Candidate("THEY", "Organization", "c1", "llm"),
    ]
    kept = filter_bare_pronouns(cands)
    assert {c.surface_form for c in kept} == {"Acme Corp", "Jane Roe"}


# ---- end-to-end via run_pipeline -------------------------------------------
@pytest.mark.ac("KG-AC-48")
def test_coreference_enabled_resolves_anaphor_so_relation_attaches_to_antecedent():
    # call 1: coref rewrite of the (only) chunk -- "It" -> "Acme Corp"
    # call 2: graph extraction over the REWRITTEN text
    graph_response = {
        "entities": [
            {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
            {"type": "Person", "surface": "Jane Roe", "confidence": 0.9},
        ],
        "relations": [
            {"type": "employs", "src_id": 0, "dst_id": 1, "confidence": 0.7,
             "evidence": "Acme Corp employs Jane Roe."},
        ],
    }
    client = _FakeLlmClient(["Acme Corp employs Jane Roe.", graph_response])
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", coreference_enabled=True)
    chunks = [Chunk("c1", "It employs Jane Roe.")]
    ent_rows, edge_rows, _summary, _usage, _blocked = run_pipeline(
        chunks, cfg, FIBO, folder_id="f1", llm_client=client,
    )
    assert len(client.prompts) == 2  # exactly one coref call + one graph-extraction call
    assert len(ent_rows) == 2
    assert len(edge_rows) == 1  # the relation attaches -- not dropped as dangling


@pytest.mark.ac("KG-AC-48")
def test_coreference_disabled_by_default_makes_no_extra_call():
    graph_response = {
        "entities": [{"type": "Organization", "surface": "Acme Corp", "confidence": 0.9}],
        "relations": [],
    }
    client = _FakeLlmClient([graph_response])
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core")  # coreference_enabled defaults False
    chunks = [Chunk("c1", "Acme Corp exists.")]
    run_pipeline(chunks, cfg, FIBO, folder_id="f1", llm_client=client)
    assert len(client.prompts) == 1  # ONLY the graph-extraction call -- no coref call at all


@pytest.mark.ac("KG-AC-48")
def test_bare_pronoun_entity_dropped_even_if_coref_rewrite_misses_it():
    # coref call "succeeds" (returns something) but the graph-extraction step STILL yields a bare
    # pronoun entity (imperfect upstream LLM behavior) -- the deterministic safety net catches it.
    graph_response = {
        "entities": [
            {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
            {"type": "Organization", "surface": "It", "confidence": 0.5},
        ],
        "relations": [],
    }
    client = _FakeLlmClient(["Acme Corp exists. It is large.", graph_response])
    cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", coreference_enabled=True)
    chunks = [Chunk("c1", "Acme Corp exists. It is large.")]
    ent_rows, _edge_rows, _summary, _usage, _blocked = run_pipeline(
        chunks, cfg, FIBO, folder_id="f1", llm_client=client,
    )
    assert {e["surface_form"] for e in ent_rows} == {"Acme Corp"}  # "It" never written
