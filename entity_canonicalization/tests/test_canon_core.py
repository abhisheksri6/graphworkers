"""Pure canonicalization core — KG-AC-22 (cluster to one identity), KG-AC-23 (type reconciliation via
the pack hierarchy), KG-AC-24 (LEI-equal anchors a match regardless of string distance). No DB/LLM."""
import pytest

from core import (
    ACCEPT, AMBIGUOUS, REJECT, Mention, aggregate_edge_group, canonical_key, cluster_mentions,
    match_band, normalize_surface, reconcile_type,
)
from ontologies import load_pack

FIBO = load_pack("fibo_core")


def _m(uid, etype, surface, ext=None):
    m = Mention(entity_uid=uid, entity_type=etype, surface_form=surface, external_id=ext)
    m.normalized_form = normalize_surface(surface)
    return m


# ---- normalization + canonical_key ---------------------------------------
@pytest.mark.ac("KG-AC-22")
def test_normalize_strips_legal_suffixes_and_punct():
    assert normalize_surface("Acme Corp.") == "acme"
    assert normalize_surface("Acme Corporation") == "acme"
    assert normalize_surface("J.P. Morgan & Co.") == "j p morgan"


@pytest.mark.ac("KG-AC-22")
def test_canonical_key_lei_beats_type_normalized():
    assert canonical_key("Bank", "acme", external_id="LEI123") == "lei:LEI123"
    assert canonical_key("Bank", "acme") == "Bank|acme"


# ---- three-band match (KG-AC-24 LEI anchor) ------------------------------
@pytest.mark.ac("KG-AC-24")
def test_lei_equal_accepts_regardless_of_string_distance():
    a = _m("1", "Bank", "Acme Bank NA", ext="LEI-X")
    b = _m("2", "Organization", "ACME BANK, National Association", ext="LEI-X")
    assert match_band(a, b, fuzzy_floor=0.8, fuzzy_ceiling=0.95) == ACCEPT


@pytest.mark.ac("KG-AC-24")
def test_different_lei_rejects():
    a = _m("1", "Bank", "Acme Bank", ext="LEI-X")
    b = _m("2", "Bank", "Acme Bank", ext="LEI-Y")  # same name, different LEI = different entities
    assert match_band(a, b, fuzzy_floor=0.8, fuzzy_ceiling=0.95) == REJECT


@pytest.mark.ac("KG-AC-24")
def test_exact_normalized_accepts_and_fuzzy_bands():
    a = _m("1", "Organization", "Acme Corp")
    b = _m("2", "Organization", "Acme Corporation")  # both normalize to "acme"
    assert match_band(a, b, fuzzy_floor=0.8, fuzzy_ceiling=0.95) == ACCEPT
    c = _m("3", "Organization", "Globex")
    assert match_band(a, c, fuzzy_floor=0.8, fuzzy_ceiling=0.95) == REJECT


# ---- clustering (KG-AC-22) -----------------------------------------------
@pytest.mark.ac("KG-AC-22")
def test_cluster_collapses_duplicates_one_identity():
    mentions = [
        _m("1", "Organization", "Acme Corp"),
        _m("2", "InvestmentAdviser", "Acme Corporation"),  # same entity, finer type
        _m("3", "Person", "Jane Roe"),
    ]
    clusters = cluster_mentions(mentions, fuzzy_floor=0.8, fuzzy_ceiling=0.95)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]  # the two Acme mentions collapse; Jane stands alone


@pytest.mark.ac("KG-AC-22")
def test_ambiguous_not_merged_without_adjudicator():
    # two names in the fuzzy band, no adjudicator -> not merged (deferred to LLM)
    a = _m("1", "Organization", "Acme Systems")
    b = _m("2", "Organization", "Acme Solutions")
    verdict = match_band(a, b, fuzzy_floor=0.5, fuzzy_ceiling=0.99)
    assert verdict == AMBIGUOUS
    clusters = cluster_mentions([a, b], fuzzy_floor=0.5, fuzzy_ceiling=0.99, adjudicate=None)
    assert len(clusters) == 2
    # ...but an adjudicator that says yes merges them
    clusters2 = cluster_mentions([a, b], fuzzy_floor=0.5, fuzzy_ceiling=0.99, adjudicate=lambda x, y: True)
    assert len(clusters2) == 1


# ---- type reconciliation (KG-AC-23) --------------------------------------
@pytest.mark.ac("KG-AC-23")
def test_reconcile_type_most_specific_wins():
    # F-CHUNK-2: InvestmentAdviser (specific) beats Organization (ancestor)
    assert reconcile_type(["Organization", "InvestmentAdviser"], FIBO) == "InvestmentAdviser"


@pytest.mark.ac("KG-AC-23")
def test_reconcile_type_cross_branch_declaration_order():
    # Bank and Equity are on different branches -> earlier-declared (Bank) wins
    assert reconcile_type(["Equity", "Bank"], FIBO) == "Bank"


# ---- canonical edge aggregation (KG-AC-47, evolve v5) ---------------------
@pytest.mark.ac("KG-AC-47")
def test_aggregate_edge_group_support_is_distinct_documents():
    # 2 mention-edges from the SAME document -> support_count counts the DOCUMENT once, not the edge
    rows = [
        {"folder_id": "f1", "confidence": 0.6, "evidence_text": "sentence A"},
        {"folder_id": "f1", "confidence": 0.7, "evidence_text": "sentence B"},
        {"folder_id": "f2", "confidence": 0.5, "evidence_text": "sentence C"},
    ]
    agg = aggregate_edge_group(rows)
    assert agg["support_count"] == 2  # distinct folder_ids: f1, f2


@pytest.mark.ac("KG-AC-47")
def test_aggregate_edge_group_confidence_is_max():
    rows = [
        {"folder_id": "f1", "confidence": 0.4, "evidence_text": "low"},
        {"folder_id": "f2", "confidence": 0.9, "evidence_text": "high"},
        {"folder_id": "f3", "confidence": 0.6, "evidence_text": "mid"},
    ]
    agg = aggregate_edge_group(rows)
    assert agg["confidence"] == 0.9


@pytest.mark.ac("KG-AC-47")
def test_aggregate_edge_group_evidence_is_top_3_by_confidence():
    rows = [
        {"folder_id": "f1", "confidence": 0.1, "evidence_text": "e1"},
        {"folder_id": "f2", "confidence": 0.9, "evidence_text": "e2"},
        {"folder_id": "f3", "confidence": 0.5, "evidence_text": "e3"},
        {"folder_id": "f4", "confidence": 0.7, "evidence_text": "e4"},
        {"folder_id": "f5", "confidence": 0.3, "evidence_text": "e5"},
    ]
    agg = aggregate_edge_group(rows)
    assert agg["evidence_text"] == ["e2", "e4", "e3"]  # top 3 by confidence, descending


@pytest.mark.ac("KG-AC-47")
def test_aggregate_edge_group_empty_input():
    agg = aggregate_edge_group([])
    assert agg == {"support_count": 0, "confidence": None, "evidence_text": []}
