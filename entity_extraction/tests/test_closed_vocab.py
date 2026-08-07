"""KG-AC-14 (worker/strategy side): the runtime vocabulary is CLOSED — a type outside the pack
(e.g. an invented LLM type) is dropped and counted in unmapped_type_count, never written. Also
covers the entity_types subset filter, the confidence threshold, and spaCy label mapping."""
import pytest

from core import Candidate
from ontologies import load_pack
from strategies import ExtractionConfig, Chunk, SpacyNerStrategy, filter_closed_vocab

FIBO = load_pack("fibo_core")


def _cfg(**kw):
    kw.setdefault("engine", "llm")
    kw.setdefault("ontology_pack", "fibo_core")
    return ExtractionConfig(**kw)


@pytest.mark.ac("KG-AC-14")
def test_unknown_type_dropped_and_counted():
    cands = [
        Candidate("Acme Corp", "Organization", "c1", "llm"),
        Candidate("DogeCoin", "Cryptocurrency", "c1", "llm"),  # invented / out-of-vocab
        Candidate("Jane Roe", "Person", "c1", "llm"),
    ]
    kept, unmapped = filter_closed_vocab(cands, _cfg(), FIBO)
    assert {c.entity_type for c in kept} == {"Organization", "Person"}
    assert unmapped == 1  # Cryptocurrency counted, never written


@pytest.mark.ac("KG-AC-14")
def test_entity_types_subset_filter_not_counted_unmapped():
    cands = [
        Candidate("Acme Corp", "Organization", "c1", "llm"),
        Candidate("Jane Roe", "Person", "c1", "llm"),
    ]
    kept, unmapped = filter_closed_vocab(cands, _cfg(entity_types=["Organization"]), FIBO)
    assert {c.entity_type for c in kept} == {"Organization"}
    assert unmapped == 0  # Person is a known type, just not selected -> not "unmapped"


@pytest.mark.ac("KG-AC-14")
def test_confidence_threshold_drops_low_confidence():
    cands = [
        Candidate("Acme Corp", "Organization", "c1", "llm", confidence=0.9),
        Candidate("Beta LLC", "Organization", "c1", "llm", confidence=0.3),
    ]
    kept, _ = filter_closed_vocab(cands, _cfg(confidence_threshold=0.5), FIBO)
    assert [c.surface_form for c in kept] == ["Acme Corp"]


class _Ent:
    def __init__(self, text, label, start, end):
        self.text, self.label_, self.start_char, self.end_char = text, label, start, end


class _Doc:
    def __init__(self, ents):
        self.ents = ents


class _FakeNlp:
    def __init__(self, ents_by_text):
        self._m = ents_by_text

    def __call__(self, text):
        return _Doc(self._m.get(text, []))


@pytest.mark.ac("KG-AC-14")
def test_spacy_label_mapping_drops_unmapped_labels():
    text = "Acme Corp raised 5 million dollars"
    fake = _FakeNlp({text: [
        _Ent("Acme Corp", "ORG", 0, 9),          # -> Organization
        _Ent("5", "CARDINAL", 17, 18),           # unmapped label -> not produced
        _Ent("5 million dollars", "MONEY", 17, 34),  # -> MonetaryAmount
    ]})
    strat = SpacyNerStrategy(nlp=fake)
    cands = strat.extract([Chunk("c1", text)], _cfg(engine="spacy"), FIBO)
    assert {(c.entity_type, c.surface_form) for c in cands} == {
        ("Organization", "Acme Corp"), ("MonetaryAmount", "5 million dollars")
    }
    org = next(c for c in cands if c.entity_type == "Organization")
    assert (org.span_start, org.span_end) == (0, 9) and org.layer == "spacy"
