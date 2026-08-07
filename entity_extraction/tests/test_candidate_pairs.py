"""KG-AC-60 (evolve v10 — candidate-pair enumeration, deterministic): a pair is a candidate iff its
(src_type, dst_type) is domain/range-compatible with >=1 declared pack relation AND both entities
co-occur in the same sentence; direction is assigned by the relation's declared domain/range; a
type-incompatible or cross-sentence pair is never a candidate; enumeration is deterministic.

Pure module (design.md's core.py purity rule extends here — no spaCy/DB): sentence boundaries are
handed in as plain (start, end) character-offset tuples per chunk, computed elsewhere (L3's
run_graph_extraction wiring) -- this module never derives them itself.
"""
import pytest

from candidate_pairs import enumerate_candidate_pairs
from core import Candidate
from ontologies import load_pack

_PACK = load_pack("fibo_core")


def _acme():
    return Candidate("Acme Corp", "Organization", "c1", "spacy", span_start=0, span_end=9)


def _jane():
    return Candidate("Jane Roe", "Person", "c1", "spacy", span_start=20, span_end=28)


@pytest.mark.ac("KG-AC-60")
def test_type_valid_same_sentence_pair_emitted_with_orientation():
    # fibo_core's ONLY Organization->Person relation is 'employs' -- unambiguous.
    pairs = enumerate_candidate_pairs([_acme(), _jane()], {"c1": [(0, 50)]}, _PACK)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.src_type == "Organization" and p.src_surface == "Acme Corp"
    assert p.dst_type == "Person" and p.dst_surface == "Jane Roe"
    assert p.allowed_relation_types == ["employs"]
    assert p.chunk_id == "c1"


@pytest.mark.ac("KG-AC-60")
def test_reverse_direction_not_emitted_when_no_relation_declares_it():
    # Person -> Organization has no declared relation in fibo_core (only the forward direction
    # 'employs' exists), so only ONE CandidatePair comes out, not two.
    pairs = enumerate_candidate_pairs([_acme(), _jane()], {"c1": [(0, 50)]}, _PACK)
    assert not any(p.src_type == "Person" for p in pairs)


@pytest.mark.ac("KG-AC-60")
def test_type_incompatible_pair_not_emitted():
    person_a = Candidate("Jane Roe", "Person", "c1", "spacy", span_start=0, span_end=8)
    person_b = Candidate("John Smith", "Person", "c1", "spacy", span_start=20, span_end=30)
    # no fibo_core relation has domain=Person in either direction
    pairs = enumerate_candidate_pairs([person_a, person_b], {"c1": [(0, 50)]}, _PACK)
    assert pairs == []


@pytest.mark.ac("KG-AC-60")
def test_cross_sentence_pair_not_emitted():
    # same types as the positive case, but the two sentence spans put Acme in sentence 0 and
    # Jane in sentence 1 -- type-compatible but NOT same-sentence.
    pairs = enumerate_candidate_pairs([_acme(), _jane()], {"c1": [(0, 15), (15, 50)]}, _PACK)
    assert pairs == []


@pytest.mark.ac("KG-AC-60")
def test_missing_sentence_info_yields_no_pairs():
    # a chunk with no entry in sentence_spans fails CLOSED (no pairs), never an unbounded
    # whole-chunk fallback -- sentence boundaries are always the caller's (L3's) responsibility.
    pairs = enumerate_candidate_pairs([_acme(), _jane()], {}, _PACK)
    assert pairs == []


@pytest.mark.ac("KG-AC-60")
def test_enumeration_deterministic_across_runs():
    entities = [_acme(), _jane()]
    spans = {"c1": [(0, 50)]}
    first = enumerate_candidate_pairs(entities, spans, _PACK)
    second = enumerate_candidate_pairs(entities, spans, _PACK)
    assert first == second
