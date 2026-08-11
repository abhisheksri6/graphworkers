"""C5 ontology packs — KG-AC-14 (closed vocabulary + fail-loud on unknown pack) and KG-AC-23
(most-specific type per the declared parent hierarchy). Pure — no spaCy/DB.

Owner GATE-2 review (deferred from A1): the shipped sample pack DEFINITIONS (`generic`, `fibo_core`
type list + parent hierarchy + guidance) are reviewed here — see the C5 Done writeup for the summary
surfaced to the owner. `iri` refs are illustrative/inert (F-IRI-1), not a freeze gate.
"""
import pytest

from ontologies import (
    DepPattern, EntityType, OntologyError, Pack, Relation, RegexPattern, available_packs, load_pack,
)


# ---- relation `iri` retention (P1, spec v13, KG-AC-85) --------------------
@pytest.mark.ac("KG-AC-85")
def test_relation_iri_survives_loading():
    # today Relation is (type, domain, range, guidance) -- iri is silently discarded at parse time.
    p = Pack(name="x", version="1", description="",
             entity_types=[EntityType("Investor", None, [], "", None),
                           EntityType("InvestmentRelationship", None, [], "", None)],
             relations=[Relation("hasInvestor", ["InvestmentRelationship"], ["Investor"], "",
                                 iri="https://contextbuilder.ai/ontology/investment#hasInvestor")])
    assert p.relations["hasInvestor"].iri == "https://contextbuilder.ai/ontology/investment#hasInvestor"


@pytest.mark.ac("KG-AC-85")
def test_relation_iri_optional_pack_unaffected():
    p = Pack(name="x", version="1", description="",
             entity_types=[EntityType("A", None, [], "", None), EntityType("B", None, [], "", None)],
             relations=[Relation("rel", ["A"], ["B"], "")])
    assert p.relations["rel"].iri is None


@pytest.mark.ac("KG-AC-85")
def test_investment_fibo_relations_carry_iri():
    p = load_pack("investment_fibo")
    r = p.relations["hasInvestor"]
    assert r.iri == "https://contextbuilder.ai/ontology/investment#hasInvestor"


# ---- pack loading + validation (KG-AC-14) --------------------------------
@pytest.mark.ac("KG-AC-14")
def test_shipped_packs_load_and_validate():
    assert set(available_packs()) >= {"generic", "fibo_core"}
    generic = load_pack("generic")
    fibo = load_pack("fibo_core")
    assert generic.version and fibo.version
    assert len(fibo.entity_types) == 24
    assert len(fibo.relations) == 10


@pytest.mark.ac("KG-AC-54")
@pytest.mark.ac("KG-AC-55")
def test_fibo_custom_shipped_patterns_load_and_validate():
    # 2026-08-07: fibo_custom v1.1 gains rules-based entity/relation patterns (was v1.0, none).
    # v11 (2026-08-08): the LEI identifier pattern is withdrawn with the gazetteer/external-id
    # plane (KG-AC-54 amended) — Email (not an identifier) is unaffected and stays.
    fibo_custom = load_pack("fibo_custom")
    assert fibo_custom.version == "1.1"
    assert len(fibo_custom.regex_patterns) == 1  # Email only
    assert {rp.entity_type for rp in fibo_custom.regex_patterns} == {"Email"}
    assert len(fibo_custom.dep_patterns) == 3  # worksFor (join/found + works-for), controls
    assert {dp.relation_type for dp in fibo_custom.dep_patterns} == {"worksFor", "controls"}
    # 'owns' is deliberately NOT covered (multi-type range, out of scope — see rules_relations.py).
    assert "owns" not in {dp.relation_type for dp in fibo_custom.dep_patterns}


@pytest.mark.ac("KG-AC-14")
def test_unknown_pack_fails_loud():
    with pytest.raises(OntologyError):
        load_pack("does_not_exist")


@pytest.mark.ac("KG-AC-14")
def test_closed_vocabulary_membership():
    fibo = load_pack("fibo_core")
    assert fibo.is_known_type("Bank") is True
    assert fibo.is_known_type("InvestmentAdviser") is True
    # an invented / out-of-vocab type is NOT known -> strategies drop + count it
    assert fibo.is_known_type("Cryptocurrency") is False


@pytest.mark.ac("KG-AC-14")
def test_spacy_label_mapping():
    fibo = load_pack("fibo_core")
    assert fibo.map_spacy_label("ORG") == "Organization"
    assert fibo.map_spacy_label("GPE") == "Country"
    assert fibo.map_spacy_label("CARDINAL") is None  # unmapped -> dropped


@pytest.mark.ac("KG-AC-14")
def test_version_mandatory():
    with pytest.raises(OntologyError):
        Pack(name="x", version="", description="", entity_types=[EntityType("A", None, [], "", None)],
             relations=[])


@pytest.mark.ac("KG-AC-14")
def test_hierarchy_must_be_a_dag():
    # A -> B -> A cycle must be rejected by the loader/validator.
    with pytest.raises(OntologyError):
        Pack(name="x", version="1", description="",
             entity_types=[EntityType("A", "B", [], "", None), EntityType("B", "A", [], "", None)],
             relations=[])


@pytest.mark.ac("KG-AC-14")
def test_parent_must_be_declared():
    with pytest.raises(OntologyError):
        Pack(name="x", version="1", description="",
             entity_types=[EntityType("A", "Ghost", [], "", None)], relations=[])


# ---- type reconciliation hierarchy (KG-AC-23) ----------------------------
@pytest.mark.ac("KG-AC-23")
def test_descendant_beats_ancestor():
    fibo = load_pack("fibo_core")
    # the F-CHUNK-2 case: InvestmentAdviser (specific) vs Organization (generic) -> specific wins
    assert fibo.most_specific_type(["Organization", "InvestmentAdviser"]) == "InvestmentAdviser"
    assert fibo.most_specific_type(["InvestmentAdviser", "Organization"]) == "InvestmentAdviser"


@pytest.mark.ac("KG-AC-23")
def test_cross_branch_tie_breaks_by_declaration_order():
    fibo = load_pack("fibo_core")
    # Bank and Equity are on different branches; Bank is declared before Equity -> Bank wins.
    assert fibo.most_specific_type(["Equity", "Bank"]) == "Bank"
    assert fibo.most_specific_type(["Bank", "Equity"]) == "Bank"


@pytest.mark.ac("KG-AC-23")
def test_most_specific_ignores_unknown_types():
    fibo = load_pack("fibo_core")
    assert fibo.most_specific_type(["Cryptocurrency", "Bond"]) == "Bond"
    assert fibo.most_specific_type(["Cryptocurrency"]) is None


@pytest.mark.ac("KG-AC-23")
def test_three_level_chain_most_specific():
    # deepest descendant wins across a 3-level chain
    p = Pack(name="x", version="1", description="",
             entity_types=[EntityType("A", None, [], "", None), EntityType("B", "A", [], "", None),
                           EntityType("C", "B", [], "", None)],
             relations=[])
    assert p.most_specific_type(["A", "B", "C"]) == "C"
    assert p.most_specific_type(["A", "C"]) == "C"


# ---- J1 (spec v8): regex_patterns / dep_patterns — ADR-0012/0013 -----------
@pytest.mark.ac("KG-AC-54")
def test_regex_patterns_load_and_validate():
    # v11 (2026-08-08): fibo_core's only regex_pattern was the LEI identifier, withdrawn with the
    # gazetteer/external-id plane — fibo_custom's non-identifier Email pattern proves the shipped
    # regex_patterns machinery still loads and validates (KG-AC-54's amended "mechanism survives").
    fibo_custom = load_pack("fibo_custom")
    assert len(fibo_custom.regex_patterns) >= 1
    rp = fibo_custom.regex_patterns[0]
    assert rp.entity_type in fibo_custom.entity_types
    import re
    re.compile(rp.pattern)  # must be a valid regex


@pytest.mark.ac("KG-AC-54")
def test_regex_pattern_undeclared_type_fails_loud():
    with pytest.raises(OntologyError):
        Pack(name="x", version="1", description="",
             entity_types=[EntityType("A", None, [], "", None)], relations=[],
             regex_patterns=[RegexPattern(entity_type="Ghost", pattern=r"\d+")])


@pytest.mark.ac("KG-AC-54")
def test_regex_pattern_invalid_regex_fails_loud():
    with pytest.raises(OntologyError):
        Pack(name="x", version="1", description="",
             entity_types=[EntityType("A", None, [], "", None)], relations=[],
             regex_patterns=[RegexPattern(entity_type="A", pattern="(unclosed")])


@pytest.mark.ac("KG-AC-54")
def test_regex_pattern_unknown_checksum_fails_loud():
    with pytest.raises(OntologyError):
        Pack(name="x", version="1", description="",
             entity_types=[EntityType("A", None, [], "", None)], relations=[],
             regex_patterns=[RegexPattern(entity_type="A", pattern=r"\d+", checksum="bogus")])


@pytest.mark.ac("KG-AC-55")
def test_dep_patterns_load_and_validate():
    fibo = load_pack("fibo_core")
    assert len(fibo.dep_patterns) >= 1
    dp = fibo.dep_patterns[0]
    assert dp.relation_type in fibo.relations
    node_ids = {tok["RIGHT_ID"] for tok in dp.pattern}
    assert dp.src_node in node_ids and dp.dst_node in node_ids


def _dep_pattern(relation_type="employs", **overrides):
    pattern = overrides.pop("pattern", [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "employ"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubj"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "dobj"}},
    ])
    kwargs = dict(relation_type=relation_type, pattern=pattern, src_node="subject", dst_node="object")
    kwargs.update(overrides)
    return DepPattern(**kwargs)


def _pack_with_relation(dep_patterns):
    return Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("Organization", None, [], "", None), EntityType("Person", None, [], "", None)],
        relations=[Relation("employs", ["Organization"], ["Person"], "")],
        dep_patterns=dep_patterns,
    )


@pytest.mark.ac("KG-AC-55")
def test_dep_pattern_undeclared_relation_fails_loud():
    with pytest.raises(OntologyError):
        _pack_with_relation([_dep_pattern(relation_type="acquires")])


@pytest.mark.ac("KG-AC-55")
def test_dep_pattern_malformed_structure_fails_loud():
    # first token must be the anchor: RIGHT_ID + RIGHT_ATTRS, no LEFT_ID/REL_OP
    bad = [{"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubj"}}]
    with pytest.raises(OntologyError):
        _pack_with_relation([_dep_pattern(pattern=bad)])


@pytest.mark.ac("KG-AC-55")
def test_dep_pattern_undefined_node_reference_fails_loud():
    # "subject" references LEFT_ID "ghost", never introduced by an earlier RIGHT_ID
    bad = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "employ"}},
        {"LEFT_ID": "ghost", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubj"}},
    ]
    with pytest.raises(OntologyError):
        _pack_with_relation([_dep_pattern(pattern=bad, dst_node="subject")])


@pytest.mark.ac("KG-AC-55")
def test_dep_pattern_src_dst_must_be_declared_nodes():
    with pytest.raises(OntologyError):
        _pack_with_relation([_dep_pattern(dst_node="not_a_node")])


# ---- relation domain/range (KG-AC-16 pack-level; enforced by strategies in C4) ----
@pytest.mark.ac("KG-AC-16")
def test_relation_domain_range_with_subtypes():
    fibo = load_pack("fibo_core")
    # Bond is a FinancialInstrument subtype -> issues(Organization, Bond) is legal
    assert fibo.relation_allowed("issues", "Organization", "Bond") is True
    # Bank is an Organization subtype -> still a legal issuer
    assert fibo.relation_allowed("issues", "Bank", "Equity") is True
    # a Person cannot issue an instrument
    assert fibo.relation_allowed("issues", "Person", "Bond") is False
    # unknown relation type -> not allowed
    assert fibo.relation_allowed("bffWith", "Person", "Person") is False
