"""KG-AC-55: the DependencyMatcher relation layer (ADR-0012 §3) — pack-authored `dep_patterns`
yield typed relations at confidence 1.0, each carrying the matched sentence as `evidence_text`
(KG-AC-46). Uses a manually-constructed spaCy Doc (spacy.blank + explicit heads/deps/lemmas) so
these tests run WITHOUT the ~600MB by-copy en_core_web_lg model, which is not present on this dev
machine (only on the deployed worker) -- the real trained-parser integration is covered by C12/the
worker's own offline-load contract, not here."""
import pytest
import spacy
from spacy.tokens import Doc

from core import Relation
from ontologies import DepPattern, EntityType, Pack, Relation as PackRelation
from strategies.base import Chunk, ExtractionConfig
from strategies.rules_relations import RulesRelationsStrategy

_EMPLOYS_PATTERN = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["employ", "hire"]}}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubj"}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "dobj"}},
]


def _pack(dep_patterns):
    return Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("Organization", None, [], "", None), EntityType("Person", None, [], "", None)],
        relations=[PackRelation("employs", ["Organization"], ["Person"], "")],
        gazetteer_link_types=[], dep_patterns=dep_patterns,
    )


def _worksfor_pack(dep_patterns):
    return Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("Organization", None, [], "", None), EntityType("Person", None, [], "", None)],
        relations=[PackRelation("worksFor", ["Person"], ["Organization"], "")],
        gazetteer_link_types=[], dep_patterns=dep_patterns,
    )


def _employs_pack():
    return _pack([DepPattern(relation_type="employs", pattern=_EMPLOYS_PATTERN,
                              src_node="subject", dst_node="object")])


@pytest.fixture()
def blank_nlp():
    return spacy.blank("en")


def _svo_doc(nlp, *, subject="Microsoft", verb="employed", verb_lemma="employ", obj="Satya"):
    words = [subject, verb, obj]
    heads = [1, 1, 1]
    deps = ["nsubj", "ROOT", "dobj"]
    lemmas = [subject, verb_lemma, obj]
    return Doc(nlp.vocab, words=words, heads=heads, deps=deps, lemmas=lemmas)


@pytest.mark.ac("KG-AC-55")
def test_svo_match_yields_typed_relation_with_evidence(blank_nlp):
    doc = _svo_doc(blank_nlp)
    strategy = RulesRelationsStrategy(nlp=_FixedDocNlp(blank_nlp, doc))
    out = strategy.extract([Chunk("c1", "irrelevant text, doc is injected")], ExtractionConfig(), _employs_pack())
    assert len(out) == 1
    r = out[0]
    assert isinstance(r, Relation)
    assert r.relation_type == "employs"
    assert r.src_surface == "Microsoft" and r.src_type == "Organization"
    assert r.dst_surface == "Satya" and r.dst_type == "Person"
    assert r.confidence == 1.0
    assert r.source_chunk_id == "c1"
    assert r.evidence_text == "Microsoft employed Satya"


@pytest.mark.ac("KG-AC-55")
def test_multiword_object_via_compound_head_structure(blank_nlp):
    # "Microsoft employed Satya Nadella." -- "Satya" is a compound modifier of "Nadella" (the
    # dobj head); the DependencyMatcher's "object" node binds to "Nadella" per the pattern.
    words = ["Microsoft", "employed", "Satya", "Nadella", "."]
    heads = [1, 1, 3, 1, 1]
    deps = ["nsubj", "ROOT", "compound", "dobj", "punct"]
    lemmas = ["Microsoft", "employ", "Satya", "Nadella", "."]
    doc = Doc(blank_nlp.vocab, words=words, heads=heads, deps=deps, lemmas=lemmas)
    strategy = RulesRelationsStrategy(nlp=_FixedDocNlp(blank_nlp, doc))
    out = strategy.extract([Chunk("c1", "x")], ExtractionConfig(), _employs_pack())
    assert len(out) == 1
    assert out[0].dst_surface == "Nadella"  # the dobj head token; "Satya" is its compound modifier


@pytest.mark.ac("KG-AC-55")
def test_no_dep_patterns_is_a_noop(blank_nlp):
    strategy = RulesRelationsStrategy(nlp=blank_nlp)
    out = strategy.extract([Chunk("c1", "text")], ExtractionConfig(), _pack([]))
    assert out == []


@pytest.mark.ac("KG-AC-55")
def test_no_match_in_sentence_yields_nothing(blank_nlp):
    words = ["Microsoft", "reported", "earnings", "."]
    heads = [1, 1, 1, 1]
    deps = ["nsubj", "ROOT", "dobj", "punct"]
    lemmas = ["Microsoft", "report", "earnings", "."]
    doc = Doc(blank_nlp.vocab, words=words, heads=heads, deps=deps, lemmas=lemmas)
    strategy = RulesRelationsStrategy(nlp=_FixedDocNlp(blank_nlp, doc))
    out = strategy.extract([Chunk("c1", "x")], ExtractionConfig(), _employs_pack())
    assert out == []


@pytest.mark.ac("KG-AC-55")
def test_missing_nlp_fails_loud():
    strategy = RulesRelationsStrategy(nlp=None)
    with pytest.raises(RuntimeError):
        strategy.extract([Chunk("c1", "x")], ExtractionConfig(), _employs_pack())


@pytest.mark.ac("KG-AC-55")
def test_missing_parser_pipe_fails_loud(blank_nlp):
    # spacy.blank("en") has NO parser pipe -- this is the exact silent-failure class KG-AC-42
    # guards against: never a silent zero-relation result when the dependency parse is unavailable.
    assert "parser" not in blank_nlp.pipe_names
    strategy = RulesRelationsStrategy(nlp=blank_nlp)
    with pytest.raises(RuntimeError, match="parser"):
        strategy.extract([Chunk("c1", "x")], ExtractionConfig(), _employs_pack())


@pytest.mark.ac("KG-AC-55")
def test_relations_tagged_with_correct_chunk_across_multiple_chunks(blank_nlp):
    doc1 = _svo_doc(blank_nlp, subject="Microsoft", obj="Satya")
    doc2 = _svo_doc(blank_nlp, subject="Google", obj="Sundar")
    strategy = RulesRelationsStrategy(nlp=_SequentialDocNlp(blank_nlp, [doc1, doc2]))
    out = strategy.extract(
        [Chunk("c1", "x"), Chunk("c2", "y")], ExtractionConfig(), _employs_pack())
    by_chunk = {r.source_chunk_id: r for r in out}
    assert by_chunk["c1"].src_surface == "Microsoft"
    assert by_chunk["c2"].src_surface == "Google"


@pytest.mark.ac("KG-AC-55")
def test_multiple_patterns_for_the_same_relation_type_all_match(blank_nlp):
    # 2026-08-07 bugfix: registering a SECOND dep_pattern under the same relation_type used to
    # silently overwrite the first in spaCy's DependencyMatcher (matcher.add() replaces, not
    # accumulates) -- so only the LAST-registered phrasing for a relation ever matched, with no
    # error. Two patterns for "worksFor": a plain transitive verb (join, dobj-shaped) and a
    # preposition-chain verb (work + "for" + pobj-shaped) -- genuinely different node structures.
    join_pattern = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "join"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubj"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "dobj"}},
    ]
    works_for_pattern = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "work"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubj"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": "for"}},
        {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "pobj"}},
    ]
    pack = _worksfor_pack([
        DepPattern(relation_type="worksFor", pattern=join_pattern, src_node="subject", dst_node="object"),
        DepPattern(relation_type="worksFor", pattern=works_for_pattern, src_node="subject", dst_node="object"),
    ])

    # doc 1: "Jane joined Acme." -- only the FIRST-registered pattern shape.
    doc1 = Doc(blank_nlp.vocab, words=["Jane", "joined", "Acme", "."], heads=[1, 1, 1, 1],
              deps=["nsubj", "ROOT", "dobj", "punct"], lemmas=["Jane", "join", "Acme", "."])
    strategy = RulesRelationsStrategy(nlp=_FixedDocNlp(blank_nlp, doc1))
    out1 = strategy.extract([Chunk("c1", "x")], ExtractionConfig(), pack)
    assert len(out1) == 1 and out1[0].src_surface == "Jane" and out1[0].dst_surface == "Acme"

    # doc 2: "Jane works for Beta." -- only the SECOND-registered pattern shape (prep-chain).
    # "Beta" (pobj) attaches to "for" (the preposition), NOT directly to the verb.
    words = ["Jane", "works", "for", "Beta", "."]
    doc2 = Doc(blank_nlp.vocab, words=words, heads=[1, 1, 1, 2, 1],
              deps=["nsubj", "ROOT", "prep", "pobj", "punct"], lemmas=["Jane", "work", "for", "Beta", "."])
    strategy2 = RulesRelationsStrategy(nlp=_FixedDocNlp(blank_nlp, doc2))
    out2 = strategy2.extract([Chunk("c1", "x")], ExtractionConfig(), pack)
    assert len(out2) == 1 and out2[0].src_surface == "Jane" and out2[0].dst_surface == "Beta"


class _FixedDocNlp:
    """Test seam: an nlp-like object whose __call__ ignores its text argument and returns a
    pre-built Doc, and whose .vocab/.pipe_names mirror a real parser-bearing pipeline. Lets the
    DependencyMatcher machinery run for real against a hand-built Doc, without a trained model."""
    def __init__(self, real_blank_nlp, doc):
        self._doc = doc
        self.vocab = real_blank_nlp.vocab
        self.pipe_names = ["parser"]  # simulate the parser pipe being present

    def __call__(self, text):
        return self._doc


class _SequentialDocNlp(_FixedDocNlp):
    def __init__(self, real_blank_nlp, docs):
        super().__init__(real_blank_nlp, docs[0])
        self._docs = iter(docs)

    def __call__(self, text):
        return next(self._docs)
