"""KG-AC-55: proves the SHIPPED fibo_core dep_patterns match real constructed text. Expanded
2026-08-07 from just `employs` to 7 of the pack's 10 relations (item #2 — deterministic relation
floor). Same hand-built-Doc technique as test_rules_relations.py (no trained model on this dev box).
The three relations left LLM-only (`hasExecutive`, `hasRating`, `denominatedIn`) have no clean
single dependency-pattern phrasing — recorded as a deliberate coverage gap, not an omission."""
import pytest
import spacy
from spacy.tokens import Doc

from ontologies import load_pack
from strategies.base import Chunk, ExtractionConfig
from strategies.rules_relations import RulesRelationsStrategy


class _FixedDocNlp:
    def __init__(self, vocab, doc):
        self.vocab = vocab
        self.pipe_names = ["parser"]
        self._doc = doc

    def __call__(self, text):
        return self._doc


def _match(doc, relation_type):
    pack = load_pack("fibo_core")
    nlp_stub = _FixedDocNlp(doc.vocab, doc)
    out = RulesRelationsStrategy(nlp=nlp_stub).extract([Chunk("c1", "x")], ExtractionConfig(), pack)
    return [r for r in out if r.relation_type == relation_type]


@pytest.fixture()
def vocab():
    return spacy.blank("en").vocab


def _doc(vocab, words, heads, deps, lemmas):
    return Doc(vocab, words=words, heads=heads, deps=deps, lemmas=lemmas)


@pytest.mark.ac("KG-AC-55")
def test_employs(vocab):
    d = _doc(vocab, ["Microsoft", "employed", "Satya", "."], [1, 1, 1, 1],
             ["nsubj", "ROOT", "dobj", "punct"], ["Microsoft", "employ", "Satya", "."])
    m = _match(d, "employs")
    assert len(m) == 1 and m[0].src_surface == "Microsoft" and m[0].dst_surface == "Satya"
    assert m[0].src_type == "Organization" and m[0].dst_type == "Person"


@pytest.mark.ac("KG-AC-55")
def test_issues(vocab):
    d = _doc(vocab, ["Acme", "issued", "bonds", "."], [1, 1, 1, 1],
             ["nsubj", "ROOT", "dobj", "punct"], ["Acme", "issue", "bond", "."])
    m = _match(d, "issues")
    assert len(m) == 1 and m[0].src_surface == "Acme" and m[0].dst_surface == "bonds"
    assert m[0].src_type == "Organization" and m[0].dst_type == "FinancialInstrument"


@pytest.mark.ac("KG-AC-55")
def test_advises(vocab):
    d = _doc(vocab, ["Blackrock", "advises", "the", "fund", "."], [1, 1, 3, 1, 1],
             ["nsubj", "ROOT", "det", "dobj", "punct"], ["Blackrock", "advise", "the", "fund", "."])
    m = _match(d, "advises")
    assert len(m) == 1 and m[0].src_surface == "Blackrock" and m[0].dst_surface == "fund"
    assert m[0].src_type == "InvestmentAdviser" and m[0].dst_type == "InvestmentFund"


@pytest.mark.ac("KG-AC-55")
def test_manages(vocab):
    d = _doc(vocab, ["Vanguard", "manages", "the", "fund", "."], [1, 1, 3, 1, 1],
             ["nsubj", "ROOT", "det", "dobj", "punct"], ["Vanguard", "manage", "the", "fund", "."])
    m = _match(d, "manages")
    assert len(m) == 1 and m[0].src_surface == "Vanguard" and m[0].dst_surface == "fund"
    assert m[0].src_type == "Organization" and m[0].dst_type == "InvestmentFund"


@pytest.mark.ac("KG-AC-55")
def test_regulates(vocab):
    d = _doc(vocab, ["SEC", "regulates", "Acme", "."], [1, 1, 1, 1],
             ["nsubj", "ROOT", "dobj", "punct"], ["SEC", "regulate", "Acme", "."])
    m = _match(d, "regulates")
    assert len(m) == 1 and m[0].src_surface == "SEC" and m[0].dst_surface == "Acme"
    assert m[0].src_type == "RegulatoryAgency" and m[0].dst_type == "Organization"


@pytest.mark.ac("KG-AC-55")
def test_subsidiary_of(vocab):
    d = _doc(vocab, ["Nuance", "is", "a", "subsidiary", "of", "Microsoft", "."],
             [1, 1, 3, 1, 3, 4, 1],
             ["nsubj", "ROOT", "det", "attr", "prep", "pobj", "punct"],
             ["Nuance", "be", "a", "subsidiary", "of", "Microsoft", "."])
    m = _match(d, "subsidiaryOf")
    assert len(m) == 1 and m[0].src_surface == "Nuance" and m[0].dst_surface == "Microsoft"
    assert m[0].src_type == "Organization" and m[0].dst_type == "Organization"


@pytest.mark.ac("KG-AC-55")
def test_located_in(vocab):
    d = _doc(vocab, ["Acme", "is", "located", "in", "Redmond", "."], [2, 2, 2, 2, 3, 2],
             ["nsubjpass", "auxpass", "ROOT", "prep", "pobj", "punct"],
             ["Acme", "be", "locate", "in", "Redmond", "."])
    m = _match(d, "locatedIn")
    assert len(m) == 1 and m[0].src_surface == "Acme" and m[0].dst_surface == "Redmond"
    assert m[0].src_type == "Organization" and m[0].dst_type == "Location"


@pytest.mark.ac("KG-AC-55")
def test_located_in_based_variant(vocab):
    # 'based in' shares the pattern (LEMMA in {locate, base, headquarter}).
    d = _doc(vocab, ["Acme", "is", "based", "in", "Redmond", "."], [2, 2, 2, 2, 3, 2],
             ["nsubjpass", "auxpass", "ROOT", "prep", "pobj", "punct"],
             ["Acme", "be", "base", "in", "Redmond", "."])
    m = _match(d, "locatedIn")
    assert len(m) == 1 and m[0].dst_surface == "Redmond"


@pytest.mark.ac("KG-AC-55")
def test_fibo_core_dep_pattern_coverage():
    # 7 of the pack's 10 relations now have a deterministic pattern; the 3 gaps are deliberate.
    pack = load_pack("fibo_core")
    covered = {dp.relation_type for dp in pack.dep_patterns}
    assert covered == {"employs", "issues", "advises", "manages", "regulates", "subsidiaryOf", "locatedIn"}
    assert {"hasExecutive", "hasRating", "denominatedIn"} & covered == set()  # LLM-only, by design
