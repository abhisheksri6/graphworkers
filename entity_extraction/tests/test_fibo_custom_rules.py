"""KG-AC-54/55: proves the SHIPPED fibo_custom v1.1 rules-based patterns (not a hand-built clone)
actually match real constructed text — regex entities (Email) and dependency relations (worksFor
via two distinct phrasings, controls). Uses the same hand-built-Doc technique as
test_rules_relations.py (no trained model on this dev machine). The LEI identifier pattern this
pack used to ship is withdrawn at spec v11 (2026-08-08, KG-AC-54 amended) with the gazetteer/
external-id plane — see test_ontology_pack.py's regex_patterns count assertion for that removal."""
import pytest
import spacy
from spacy.tokens import Doc

from ontologies import load_pack
from strategies.base import Chunk, ExtractionConfig
from strategies.rules_entities import RulesEntitiesStrategy
from strategies.rules_relations import RulesRelationsStrategy


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
