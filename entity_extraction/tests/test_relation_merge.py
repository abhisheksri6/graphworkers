"""KG-AC-56 (evolve v8): relation merge + provenance. Relations from multiple sources (rules
DependencyMatcher and/or the one-pass LLM extractor) are unioned and deduped on
(src_entity_uid, relation_type, dst_entity_uid), order-preserving; a triple asserted by both
collapses to one edge (extractor='rules+llm', confidence=max, evidence unioned); every entity and
relation row carries an `extractor` provenance value (entity-side already existed pre-v8; this
file proves the relation/edge-row side, new in v8)."""
import pytest

from core import (
    Candidate, Relation, build_edge_records, build_entity_records, entity_uid_key_map,
    merge_candidates, merge_edge_records,
)


def _edge(src, rtype, dst, *, confidence, evidence, extractor):
    return {
        "edge_uid": f"uid-{src}-{rtype}-{dst}",  # in production all contributors share one edge_uid
        "relation_type": rtype, "src_entity_uid": src, "dst_entity_uid": dst,
        "confidence": confidence, "evidence_text": evidence, "extractor": extractor,
    }


@pytest.mark.ac("KG-AC-56")
def test_single_source_relation_passes_through_unchanged():
    rows = [_edge("e1", "employs", "e2", confidence=1.0, evidence="Acme employs Jane.", extractor="rules")]
    out = merge_edge_records(rows)
    assert out == rows


@pytest.mark.ac("KG-AC-56")
def test_same_triple_from_rules_and_llm_collapses_to_one_edge():
    rows = [
        _edge("e1", "employs", "e2", confidence=1.0, evidence="Acme employs Jane.", extractor="rules"),
        _edge("e1", "employs", "e2", confidence=0.7, evidence="Jane works for Acme.", extractor="llm"),
    ]
    out = merge_edge_records(rows)
    assert len(out) == 1
    merged = out[0]
    assert merged["extractor"] == "rules+llm"
    assert merged["confidence"] == 1.0  # max of contributors
    assert "Acme employs Jane." in merged["evidence_text"]
    assert "Jane works for Acme." in merged["evidence_text"]  # union, not dropped


@pytest.mark.ac("KG-AC-56")
def test_different_triples_stay_separate():
    rows = [
        _edge("e1", "employs", "e2", confidence=1.0, evidence="a", extractor="rules"),
        _edge("e1", "manages", "e2", confidence=1.0, evidence="b", extractor="rules"),  # different relation_type
        _edge("e3", "employs", "e2", confidence=1.0, evidence="c", extractor="rules"),  # different src
    ]
    out = merge_edge_records(rows)
    assert len(out) == 3


@pytest.mark.ac("KG-AC-56")
def test_merge_is_order_preserving_and_deterministic():
    rows = [
        _edge("e1", "employs", "e2", confidence=1.0, evidence="a", extractor="rules"),
        _edge("e3", "manages", "e4", confidence=1.0, evidence="b", extractor="llm"),
        _edge("e1", "employs", "e2", confidence=0.5, evidence="a2", extractor="llm"),
    ]
    first = merge_edge_records(list(rows))
    second = merge_edge_records(list(rows))
    assert first == second
    # the surviving 2 groups appear in FIRST-ENCOUNTER order: (e1,employs,e2) group before (e3,manages,e4)
    assert (first[0]["src_entity_uid"], first[0]["relation_type"]) == ("e1", "employs")
    assert (first[1]["src_entity_uid"], first[1]["relation_type"]) == ("e3", "manages")


@pytest.mark.ac("KG-AC-56")
def test_duplicate_from_the_same_single_extractor_does_not_get_a_plus_label():
    # two rules-layer matches for the identical triple (e.g. two overlapping dep patterns) should
    # collapse without a spurious 'rules+rules' label.
    rows = [
        _edge("e1", "employs", "e2", confidence=1.0, evidence="a", extractor="rules"),
        _edge("e1", "employs", "e2", confidence=1.0, evidence="a", extractor="rules"),
    ]
    out = merge_edge_records(rows)
    assert len(out) == 1
    assert out[0]["extractor"] == "rules"


@pytest.mark.ac("KG-AC-56")
def test_relation_extractor_field_defaults_llm_for_backward_compat():
    # pre-v8 call sites construct Relation(...) positionally with no extractor arg -- must still work.
    r = Relation("issues", "Acme", "Organization", "Bond A", "Bond", "c1")
    assert r.extractor == "llm"


@pytest.mark.ac("KG-AC-56")
def test_build_edge_records_carries_extractor_from_relation():
    merged = merge_candidates([
        Candidate("Acme", "Organization", "c1", "regex", 0, 4),
        Candidate("Jane", "Person", "c1", "regex", 10, 14),
    ])
    ent_rows = build_entity_records("f1", merged, "generic", "1.0")
    key_map = entity_uid_key_map(ent_rows)
    rel = Relation("employs", "Acme", "Organization", "Jane", "Person", "c1",
                    evidence_text="Acme employs Jane.", extractor="rules")
    edges = build_edge_records("f1", [rel], key_map)
    assert len(edges) == 1
    assert edges[0]["extractor"] == "rules"
