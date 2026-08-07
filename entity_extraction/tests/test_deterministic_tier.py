"""KG-AC-58 (evolve v8): given a fixed input and a config whose enabled layers are all
deterministic (spacy engine + rules_entities + rules_relations, no llm), extraction output is
byte-identical across repeated runs of `run_pipeline`. Uses a hand-built spaCy Doc (no trained
model on this dev machine, same approach as J4's test_rules_relations.py) that carries BOTH NER
`.ents` (for SpacyNerStrategy) and a dependency parse (for RulesRelationsStrategy) simultaneously.
"""
import pytest
import spacy
from spacy.tokens import Doc

from ontologies import DepPattern, EntityType, Pack, RegexPattern
from ontologies import Relation as PackRelation
from strategies.base import Chunk, ExtractionConfig, run_pipeline

_EMPLOYS_DEP_PATTERN = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["employ", "hire"]}}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubj"}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "dobj"}},
]


def _pack():
    return Pack(
        name="x", version="1", description="",
        entity_types=[
            EntityType("Organization", None, ["ORG"], "", None),
            EntityType("Person", None, ["PERSON"], "", None),
        ],
        relations=[PackRelation("employs", ["Organization"], ["Person"], "")],
        gazetteer_link_types=[],
        regex_patterns=[RegexPattern(entity_type="Person", pattern=r"\bSatya\b")],
        dep_patterns=[DepPattern(relation_type="employs", pattern=_EMPLOYS_DEP_PATTERN,
                                  src_node="subject", dst_node="object")],
    )


class _FixedDocNlp:
    """A parsed-and-typed Doc for 'Microsoft employed Satya.', reused for every call regardless of
    the chunk text -- this only needs to prove the pipeline is deterministic given fixed
    strategy output, not exercise a real tokenizer/parser/NER model."""
    def __init__(self):
        blank = spacy.blank("en")
        self.vocab = blank.vocab
        self.pipe_names = ["parser", "ner"]
        words = ["Microsoft", "employed", "Satya", "."]
        heads = [1, 1, 1, 1]
        deps = ["nsubj", "ROOT", "dobj", "punct"]
        lemmas = ["Microsoft", "employ", "Satya", "."]
        ents = ["B-ORG", "O", "B-PERSON", "O"]
        self._doc = Doc(self.vocab, words=words, heads=heads, deps=deps, lemmas=lemmas, ents=ents)

    def __call__(self, text):
        return self._doc


def _run():
    chunks = [Chunk("c1", "Microsoft employed Satya.")]
    config = ExtractionConfig(
        engine="spacy", ontology_pack="x",
        rules_entities_enabled=True, rules_relations_enabled=True,
    )
    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        chunks, config, _pack(), folder_id="f1", spacy_nlp=_FixedDocNlp())
    return ent_rows, edge_rows, summary


@pytest.mark.ac("KG-AC-58")
def test_deterministic_tier_is_byte_identical_across_runs():
    ent1, edge1, summary1 = _run()
    ent2, edge2, summary2 = _run()
    ent3, edge3, summary3 = _run()
    assert ent1 == ent2 == ent3
    assert edge1 == edge2 == edge3
    assert summary1 == summary2 == summary3


@pytest.mark.ac("KG-AC-58")
def test_deterministic_tier_produces_the_expected_composed_result():
    ent_rows, edge_rows, summary = _run()
    # spaCy NER finds Microsoft(Organization)/Satya(Person); regex ALSO matches "Satya" (Person) on
    # the identical span -- regex(3) beats spacy(2) in LAYER_PRECEDENCE, so the regex-sourced
    # candidate wins the merge (verified: extractor='regex' on the surviving Satya row); Microsoft
    # has no regex pattern, so spaCy's candidate survives unopposed.
    by_surface = {r["surface_form"]: r for r in ent_rows}
    assert {s: r["entity_type"] for s, r in by_surface.items()} == {"Microsoft": "Organization", "Satya": "Person"}
    assert by_surface["Satya"]["extractor"] == "regex"
    assert by_surface["Microsoft"]["extractor"] == "spacy"
    assert len(edge_rows) == 1
    assert edge_rows[0]["relation_type"] == "employs"
    assert edge_rows[0]["extractor"] == "rules"
    assert summary["entity_count"] == 2 and summary["edge_count"] == 1
