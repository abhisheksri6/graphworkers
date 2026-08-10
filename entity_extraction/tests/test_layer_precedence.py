"""KG-AC-12: layer-precedence span merge — regex > spaCy > LLM on overlap within a chunk
(*amended v11: the gazetteer tier is removed with that capability*); non-overlapping candidates
from every layer unioned; deterministic for fixed input."""
import pytest

from core import Candidate, merge_candidates, spans_overlap


def _c(surface, layer, chunk, start, end, etype="Organization"):
    return Candidate(surface_form=surface, entity_type=etype, source_chunk_id=chunk,
                     layer=layer, span_start=start, span_end=end)


@pytest.mark.ac("KG-AC-12")
def test_highest_precedence_wins_on_overlap():
    cands = [
        _c("Acme", "spacy", "c1", 0, 4),          # overlaps the regex span
        _c("Acme Corp", "regex", "c1", 0, 9),      # highest precedence -> kept whole
        _c("Acme Corporation", "llm", "c1", 0, 16),  # overlaps -> dropped
    ]
    kept = merge_candidates(cands)
    assert len(kept) == 1
    assert kept[0].layer == "regex"
    assert kept[0].surface_form == "Acme Corp"


@pytest.mark.ac("KG-AC-12")
def test_non_overlapping_candidates_unioned_across_layers():
    cands = [
        _c("Acme Corp", "regex", "c1", 0, 9),
        _c("Jane Roe", "spacy", "c1", 20, 28, etype="Person"),  # no overlap -> kept
        _c("2023", "llm", "c1", 40, 44, etype="Date"),          # no overlap -> kept
    ]
    kept = merge_candidates(cands)
    assert {k.surface_form for k in kept} == {"Acme Corp", "Jane Roe", "2023"}


@pytest.mark.ac("KG-AC-12")
def test_same_span_different_chunk_does_not_conflict():
    cands = [
        _c("Acme", "spacy", "c1", 0, 4),
        _c("Beta", "spacy", "c2", 0, 4),  # same offsets, different chunk -> both kept
    ]
    kept = merge_candidates(cands)
    assert len(kept) == 2


@pytest.mark.ac("KG-AC-12")
def test_transitive_overlap_keeps_disjoint_survivor():
    # A(regex)[0,9] overlaps B(spacy)[5,14]; C(llm)[15,20] overlaps neither A nor B.
    cands = [
        _c("A Corp", "regex", "c1", 0, 9),
        _c("Corp B", "spacy", "c1", 5, 14),
        _c("D Inc", "llm", "c1", 15, 20),
    ]
    kept = merge_candidates(cands)
    surfaces = {k.surface_form for k in kept}
    assert surfaces == {"A Corp", "D Inc"}  # B dropped (overlaps the regex winner)


@pytest.mark.ac("KG-AC-12")
def test_deterministic_repeatable():
    cands = [
        _c("Acme", "spacy", "c1", 0, 4),
        _c("Acme Corp", "regex", "c1", 0, 9),
        _c("Jane", "llm", "c1", 20, 24, etype="Person"),
    ]
    first = [(k.layer, k.surface_form) for k in merge_candidates(list(cands))]
    second = [(k.layer, k.surface_form) for k in merge_candidates(list(cands))]
    assert first == second


@pytest.mark.ac("KG-AC-12")
def test_regex_beats_spacy_on_overlap():
    # v11: the top of the stack is regex > spaCy > LLM (the gazetteer tier that used to sit above
    # regex is withdrawn with that capability).
    cands = [
        _c("Acme", "spacy", "c1", 0, 4),           # overlaps the regex span -> dropped
        _c("Acme Corp", "regex", "c1", 0, 9),      # beats spaCy -> kept whole
        _c("Acme Corporation", "llm", "c1", 0, 16),  # overlaps -> dropped
    ]
    kept = merge_candidates(cands)
    assert len(kept) == 1
    assert kept[0].layer == "regex"
    assert kept[0].surface_form == "Acme Corp"


@pytest.mark.ac("KG-AC-12")
def test_spanless_candidates_never_conflict():
    cands = [
        Candidate("Acme", "Organization", "c1", "llm"),
        Candidate("Acme", "Organization", "c1", "llm"),  # spanless repeat -> both kept (occurrence disambiguates)
    ]
    assert not spans_overlap(cands[0], cands[1])
    assert len(merge_candidates(cands)) == 2
