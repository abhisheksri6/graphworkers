"""KG-AC-24/79 amended (P25, 2026-08-12): **identifier-bearing identity resolution** — Tier 1 of the
identity hierarchy, deterministic, ahead of blocking/fuzzy/LLM.

**The production bug this exists to prevent** (job `manual__2026-08-12T17:27:55`, folder
`36e398eb`): ONE agreement was written to Neo4j as **four separate nodes** — `"Investment Advisory
and Fiduciary Management Agreement"`, `"Agreement"`, `"This Agreement"`, `"...IMA-GUE-2026-101"` —
even though **three of them had already extracted the identical `agreementId: IMA-GUE-2026-101`**.
The proof they were one entity was sitting in `kg_entities.attributes` on the very rows being
canonicalized, and the algorithm never looked at it: `block_key` puts them in different blocks
(`tok:investment` / `tok:agreement` / `tok:this`) so they are never even COMPARED, and their fuzzy
scores (0.15-0.38) are far below the 0.80 floor, so they would be rejected even if they were.

Identifiers are **opaque tokens, not names** — `normalize_identifier` deliberately does NOT reuse
`normalize_surface` (which strips punctuation and legal suffixes: it would mangle `LP-2026-001`
into `2026 001` by treating the `LP` prefix as a company suffix).
"""
import pytest

from core import (
    Mention, cluster_identifier, cluster_mentions, identifier_values, normalize_identifier,
    normalize_surface,
)
from ontologies import load_pack

PACK = load_pack("investment_fibo")


def _m(uid, etype, surface, attributes=None):
    m = Mention(entity_uid=uid, entity_type=etype, surface_form=surface)
    m.normalized_form = normalize_surface(surface)
    m.identifiers = identifier_values(attributes or [], etype, PACK)
    return m


def _fact(prop, value, normalized=None):
    return {"property": prop, "value": value, "normalized_value": normalized or value,
            "evidence": f"...{value}...", "source_doc_id": "d1", "page": 1}


# ---- normalize_identifier: opaque-token treatment, NOT name normalization -----------------------
@pytest.mark.ac("KG-AC-24")
def test_normalize_identifier_casefolds_and_collapses_whitespace_only():
    assert normalize_identifier("IMA-GUE-2026-101") == "ima-gue-2026-101"
    assert normalize_identifier("  IMA-GUE-2026-101  ") == "ima-gue-2026-101"
    assert normalize_identifier("IMA  GUE\t101") == "ima gue 101"


@pytest.mark.ac("KG-AC-24")
def test_normalize_identifier_does_not_strip_punctuation_or_legal_suffixes():
    # normalize_surface would destroy both of these -- an identifier is an opaque token, and its
    # punctuation IS significant. `LP-2026-001` is the load-bearing case: normalize_surface treats
    # a leading `lp` as a company suffix and drops it entirely.
    assert normalize_identifier("LP-2026-001") == "lp-2026-001"
    assert normalize_surface("LP-2026-001") != normalize_identifier("LP-2026-001")
    assert normalize_identifier("NA.123") == "na.123"


# ---- identifier_values: pack-declared, domain-scoped -------------------------------------------
@pytest.mark.ac("KG-AC-24")
def test_identifier_values_extracts_only_identifier_range_properties():
    attrs = [
        _fact("agreementId", "IMA-GUE-2026-101"),           # range=identifier -> kept
        _fact("agreementType", "Investment Advisory ..."),   # range=string     -> ignored
        _fact("effectiveDate", "15 March 2026", "2026-03-15"),  # range=date    -> ignored
    ]
    assert identifier_values(attrs, "Agreement", PACK) == {"agreementId": "ima-gue-2026-101"}


@pytest.mark.ac("KG-AC-24")
def test_identifier_values_is_domain_scoped():
    # `subscriptionId` is declared with domain=Subscription. A fact claiming it on an Agreement is
    # not identity-bearing FOR an Agreement -- prevents cross-type merges on a shared value.
    attrs = [_fact("subscriptionId", "SUB-1")]
    assert identifier_values(attrs, "Subscription", PACK) == {"subscriptionId": "sub-1"}
    assert identifier_values(attrs, "Agreement", PACK) == {}


@pytest.mark.ac("KG-AC-24")
def test_identifier_values_empty_when_no_pack_no_attributes_or_pack_declares_none():
    assert identifier_values([], "Agreement", PACK) == {}
    assert identifier_values([_fact("agreementId", "X")], "Agreement", None) == {}
    # `generic` declares no datatype_properties at all -- no identifiers, never an error.
    assert identifier_values([_fact("agreementId", "X")], "Agreement", load_pack("generic")) == {}


# ---- cluster_mentions: the Tier-1 union (THE regression guard) ----------------------------------
@pytest.mark.ac("KG-AC-24")
def test_shared_identifier_merges_across_blocks_and_below_the_fuzzy_floor():
    # The EXACT production case. These four surfaces are in four different blocks and score
    # 0.15-0.38 against a 0.80 floor -- unreachable by every pre-P25 mechanism -- but they carry
    # the same agreementId, so they are ONE entity.
    aid = [_fact("agreementId", "IMA-GUE-2026-101")]
    ms = [
        _m("1", "Agreement", "Investment Advisory and Fiduciary Management Agreement", aid),
        _m("2", "Agreement", "Agreement", aid),
        _m("3", "Agreement", "This Agreement", aid),
        _m("4", "Agreement", "Investment Advisory and Fiduciary Management Agreement IMA-GUE-2026-101", aid),
    ]
    clusters = cluster_mentions(ms, fuzzy_floor=0.80, fuzzy_ceiling=0.95, adjudicate=None)
    assert len(clusters) == 1, [[m.surface_form for m in c] for c in clusters]
    assert {m.entity_uid for m in clusters[0]} == {"1", "2", "3", "4"}


@pytest.mark.ac("KG-AC-24")
def test_different_identifier_values_are_never_merged():
    # Same type, near-identical surfaces (fuzzy would ACCEPT), but DIFFERENT identifiers -> two
    # distinct real agreements. The identifier tier must not merge them, and must not let the
    # fuzzy tier merge them either.
    a = _m("1", "Agreement", "Investment Management Agreement", [_fact("agreementId", "IMA-2026-001")])
    b = _m("2", "Agreement", "Investment Management Agreement", [_fact("agreementId", "IMA-2026-002")])
    clusters = cluster_mentions([a, b], fuzzy_floor=0.80, fuzzy_ceiling=0.95, adjudicate=None)
    assert len(clusters) == 2, [[m.surface_form for m in c] for c in clusters]


@pytest.mark.ac("KG-AC-24")
def test_a_corpus_of_identically_titled_agreements_stays_distinct():
    # The scale version of the test above, and the reason the identifier-CONFLICT rule exists at
    # all (found by that test during P25's own implementation). "Investment Management Agreement"
    # is a generic title every document in this domain shares; on exact surface equality alone,
    # all 50 would collapse into ONE canonical entity, each one's own agreementId silently
    # overruled by its title. Identifier mismatch is decisive negative evidence.
    ms = [
        _m(str(i), "Agreement", "Investment Management Agreement",
           [_fact("agreementId", f"IMA-2026-{i:03d}")])
        for i in range(50)
    ]
    clusters = cluster_mentions(ms, fuzzy_floor=0.80, fuzzy_ceiling=0.95, adjudicate=None)
    assert len(clusters) == 50, f"identically-titled distinct agreements collapsed: {len(clusters)}"


@pytest.mark.ac("KG-AC-24")
def test_identifier_conflict_beats_the_llm_adjudicator_too():
    # An identifier mismatch must short-circuit BEFORE the ambiguous band, so a "yes" from the LLM
    # can never override it -- the deterministic tier outranks the probabilistic one by design.
    a = _m("1", "Agreement", "Investment Management Agreement", [_fact("agreementId", "A-1")])
    b = _m("2", "Agreement", "Investment Managment Agreement", [_fact("agreementId", "A-2")])  # typo -> fuzzy band
    always_yes = lambda x, y: True  # noqa: E731
    clusters = cluster_mentions([a, b], fuzzy_floor=0.80, fuzzy_ceiling=0.95, adjudicate=always_yes)
    assert len(clusters) == 2


@pytest.mark.ac("KG-AC-24")
def test_identifier_union_never_crosses_entity_type():
    # A shared raw value on two DIFFERENT types must not merge them (belt-and-braces alongside the
    # domain-scoping in identifier_values above).
    a = _m("1", "Agreement", "The Agreement", [_fact("agreementId", "X-1")])
    b = _m("2", "Subscription", "The Subscription", [_fact("subscriptionId", "X-1")])
    clusters = cluster_mentions([a, b], fuzzy_floor=0.80, fuzzy_ceiling=0.95, adjudicate=None)
    assert len(clusters) == 2


@pytest.mark.ac("KG-AC-24")
def test_mentions_without_identifiers_still_cluster_by_the_existing_rules():
    # Regression guard: the pre-P25 path is untouched for identifier-less entities (Investor
    # declares only `name`, range=string -- no identifier anywhere in the pack for it).
    a = _m("1", "Investor", "Acme Capital Partners")
    b = _m("2", "Investor", "Acme Capital Partners")
    c = _m("3", "Investor", "Globex Holdings")
    clusters = cluster_mentions([a, b, c], fuzzy_floor=0.80, fuzzy_ceiling=0.95, adjudicate=None)
    assert len(clusters) == 2
    assert {m.entity_uid for m in clusters[0]} == {"1", "2"}


# ---- cluster_identifier: the canonical_key / normalized_form basis ------------------------------
@pytest.mark.ac("KG-AC-79")
def test_cluster_identifier_returns_the_shared_value_as_the_identity_basis():
    aid = [_fact("agreementId", "IMA-GUE-2026-101")]
    cluster = [_m("1", "Agreement", "This Agreement", aid), _m("2", "Agreement", "Agreement", aid)]
    assert cluster_identifier(cluster) == "ima-gue-2026-101"


@pytest.mark.ac("KG-AC-79")
def test_cluster_identifier_is_none_when_no_mention_carries_one():
    cluster = [_m("1", "Investor", "Acme Capital Partners")]
    assert cluster_identifier(cluster) is None


@pytest.mark.ac("KG-AC-79")
def test_cluster_identifier_is_deterministic_under_input_reordering():
    # Cross-run identity depends on this being order-independent: the same real cluster must
    # produce the same canonical_key basis no matter what order its mentions were read in.
    aid = [_fact("agreementId", "IMA-GUE-2026-101")]
    a = _m("1", "Agreement", "This Agreement", aid)
    b = _m("2", "Agreement", "Agreement", aid)
    assert cluster_identifier([a, b]) == cluster_identifier([b, a])


@pytest.mark.ac("KG-AC-79")
def test_cluster_identifier_picks_deterministically_when_a_cluster_carries_several():
    # Defensive: a cluster SHOULD carry one identifier value, but if fuzzy/LLM merged two
    # identifier-bearing mentions, the basis must still be stable rather than order-dependent.
    a = _m("1", "Agreement", "Agreement A", [_fact("agreementId", "B-2")])
    b = _m("2", "Agreement", "Agreement B", [_fact("agreementId", "A-1")])
    assert cluster_identifier([a, b]) == cluster_identifier([b, a]) == "a-1"
