"""KG-AC-54/55: proves the SHIPPED fibo_custom v1.1 rules-based patterns (not a hand-built clone)
actually match real constructed text — regex entities (LEI checksum, Email) and dependency
relations (worksFor via two distinct phrasings, controls). Uses the same hand-built-Doc technique
as test_rules_relations.py (no trained model on this dev machine)."""
import pytest
import spacy
from spacy.tokens import Doc

from ontologies import load_pack
from strategies.base import Chunk, ExtractionConfig
from strategies.rules_entities import RulesEntitiesStrategy
from strategies.rules_relations import RulesRelationsStrategy

# Independently-derived valid LEI (same technique as test_rules_entities.py — not imported from
# the module under test).
def _valid_lei(prefix18: str) -> str:
    def letters_to_digits(s):
        return "".join(ch if ch.isdigit() else str(ord(ch.upper()) - ord("A") + 10) for ch in s)
    numeric = letters_to_digits(prefix18 + "00")
    remainder = 0
    for ch in numeric:
        remainder = (remainder * 10 + int(ch)) % 97
    check = 98 - remainder
    return f"{prefix18}{check:02d}"


_VALID_LEI = _valid_lei("529900HNOAA1KXQJUQ")


@pytest.mark.ac("KG-AC-54")
def test_shipped_lei_pattern_matches_valid_lei():
    pack = load_pack("fibo_custom")
    out = RulesEntitiesStrategy().extract(
        [Chunk("c1", f"The counterparty's LEI is {_VALID_LEI}.")], ExtractionConfig(), pack)
    assert any(c.surface_form == _VALID_LEI and c.entity_type == "Organization" for c in out)


@pytest.mark.ac("KG-AC-54")
def test_shipped_email_pattern_matches():
    pack = load_pack("fibo_custom")
    out = RulesEntitiesStrategy().extract(
        [Chunk("c1", "Contact jane.roe@acme-corp.example for details.")], ExtractionConfig(), pack)
    assert any(c.surface_form == "jane.roe@acme-corp.example" and c.entity_type == "Email" for c in out)


class _FixedDocNlp:
    def __init__(self, vocab, doc):
        self.vocab = vocab
        self.pipe_names = ["parser"]
        self._doc = doc

    def __call__(self, text):
        return self._doc


@pytest.mark.ac("KG-AC-55")
def test_shipped_worksfor_pattern_matches_join_phrasing():
    pack = load_pack("fibo_custom")
    nlp = spacy.blank("en")
    doc = Doc(nlp.vocab, words=["Jane", "joined", "Acme", "."], heads=[1, 1, 1, 1],
             deps=["nsubj", "ROOT", "dobj", "punct"], lemmas=["Jane", "join", "Acme", "."])
    out = RulesRelationsStrategy(nlp=_FixedDocNlp(nlp.vocab, doc)).extract(
        [Chunk("c1", "x")], ExtractionConfig(), pack)
    matches = [r for r in out if r.relation_type == "worksFor"]
    assert len(matches) == 1
    assert matches[0].src_surface == "Jane" and matches[0].dst_surface == "Acme"
    assert matches[0].src_type == "Person" and matches[0].dst_type == "Organization"


@pytest.mark.ac("KG-AC-55")
def test_shipped_worksfor_pattern_matches_prep_phrasing():
    pack = load_pack("fibo_custom")
    nlp = spacy.blank("en")
    words = ["Jane", "works", "for", "Beta", "."]
    doc = Doc(nlp.vocab, words=words, heads=[1, 1, 1, 2, 1],
             deps=["nsubj", "ROOT", "prep", "pobj", "punct"], lemmas=["Jane", "work", "for", "Beta", "."])
    out = RulesRelationsStrategy(nlp=_FixedDocNlp(nlp.vocab, doc)).extract(
        [Chunk("c1", "x")], ExtractionConfig(), pack)
    matches = [r for r in out if r.relation_type == "worksFor"]
    assert len(matches) == 1
    assert matches[0].src_surface == "Jane" and matches[0].dst_surface == "Beta"


@pytest.mark.ac("KG-AC-55")
def test_shipped_controls_pattern_matches():
    pack = load_pack("fibo_custom")
    nlp = spacy.blank("en")
    doc = Doc(nlp.vocab, words=["Acme", "controls", "Beta", "."], heads=[1, 1, 1, 1],
             deps=["nsubj", "ROOT", "dobj", "punct"], lemmas=["Acme", "control", "Beta", "."])
    out = RulesRelationsStrategy(nlp=_FixedDocNlp(nlp.vocab, doc)).extract(
        [Chunk("c1", "x")], ExtractionConfig(), pack)
    matches = [r for r in out if r.relation_type == "controls"]
    assert len(matches) == 1
    assert matches[0].src_surface == "Acme" and matches[0].dst_surface == "Beta"
    assert matches[0].src_type == "Party" and matches[0].dst_type == "Organization"
