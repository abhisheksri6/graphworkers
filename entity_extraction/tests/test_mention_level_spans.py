"""KG-AC-63 (evolve v12 — mention-level emission, re-scoped at clarify 2026-08-11): one `Candidate`
per entity OCCURRENCE, not one per distinct surface. Grounding at clarify found this is a
**regression guard on an already-correct code contract** — `llm_graph.py`'s `seen_surface` dict is
an occurrence-COUNTER, not a dedup gate, so the entity loop has never collapsed repeated surfaces;
every item the LLM's tool-call response contains already becomes its own `Candidate` with its own
disambiguated span. The measured shortfall (19 entities -> 17 pairs -> 1 relation, 2026-08-08) is
the MODEL's own generation behavior — it tends to return one item per distinct name, not one per
mention — a prompt-instruction gap. This file therefore has two, deliberately different kinds of
proof: a regression guard on the code contract (expected to ALREADY pass — nothing to fix there,
recorded at freeze so a future change cannot silently reintroduce a collapse), and a real red-first
assertion that the prompt now asks for the behavior the model has been under-producing. Whether the
strengthened prompt actually changes model behavior is measured by KG-AC-P1/P2's existing F1 evals,
not here."""
import pytest

from ontologies import load_pack
from strategies.base import Chunk, ExtractionConfig
from strategies.llm_graph import LlmGraphStrategy, build_graph_system_prompt

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


# ---- regression guard: the code contract already works, prove it stays that way -----------
@pytest.mark.ac("KG-AC-63")
def test_repeated_surface_items_each_get_their_own_candidate_and_span():
    text = "Acme Corp announced today. Later, Acme Corp confirmed the news. Acme Corp closed higher."
    response = {
        "entities": [
            {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
            {"type": "Organization", "surface": "Acme Corp", "confidence": 0.85},
            {"type": "Organization", "surface": "Acme Corp", "confidence": 0.8},
        ],
        "relations": [],
    }
    client = _FakeLlmClient([response])
    strat = LlmGraphStrategy(llm_client=client)
    out = strat.extract([Chunk("c1", text)], ExtractionConfig(engine="llm", ontology_pack="fibo_core"), FIBO)

    assert len(out) == 3  # one Candidate per item -- never collapsed to one

    spans = [(c.span_start, c.span_end) for c in out]
    assert len(set(spans)) == 3  # each occurrence gets its OWN, distinct span

    # the three spans are exactly the 1st, 2nd, 3rd occurrence of "Acme Corp" in the text
    expected = []
    start = -1
    for _ in range(3):
        start = text.find("Acme Corp", start + 1)
        expected.append((start, start + len("Acme Corp")))
    assert sorted(spans) == sorted(expected)


@pytest.mark.ac("KG-AC-63")
def test_mixed_surfaces_interleave_correctly():
    # occurrence indexing is per-surface, not global -- a second entity interleaved between two
    # occurrences of the first must not perturb either one's occurrence index.
    text = "Acme Corp met Jane Roe. Acme Corp later met Jane Roe again."
    response = {
        "entities": [
            {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
            {"type": "Person", "surface": "Jane Roe", "confidence": 0.9},
            {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
            {"type": "Person", "surface": "Jane Roe", "confidence": 0.9},
        ],
        "relations": [],
    }
    client = _FakeLlmClient([response])
    strat = LlmGraphStrategy(llm_client=client)
    out = strat.extract([Chunk("c1", text)], ExtractionConfig(engine="llm", ontology_pack="fibo_core"), FIBO)
    assert len(out) == 4
    acme_spans = sorted(c.span_start for c in out if c.surface_form == "Acme Corp")
    jane_spans = sorted(c.span_start for c in out if c.surface_form == "Jane Roe")
    assert acme_spans == sorted([text.find("Acme Corp"), text.rfind("Acme Corp")])
    assert jane_spans == sorted([text.find("Jane Roe"), text.rfind("Jane Roe")])


# ---- the real fix: the prompt must ask for the behavior the model has under-produced -------
@pytest.mark.ac("KG-AC-63")
def test_system_prompt_asks_for_one_item_per_occurrence():
    prompt = build_graph_system_prompt(FIBO).lower()
    assert "per occurrence" in prompt or "each occurrence" in prompt or "every occurrence" in prompt
    assert "not one per distinct" in prompt or "do not deduplicate" in prompt or "not just once" in prompt
