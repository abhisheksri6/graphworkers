"""specs/knowledge-graph v16 — S4: ONE type-compatibility rule across every identity tier
(KG-AC-102, KG-AC-24 as amended).

The 2026-08-17 review found three different type rules in one file, producing opposite errors from
the same evidence: identifier COLLECTION demanded an exact `domain` match (so an inherited `lei`
was invisible), Tier-1 GROUPING demanded exact `entity_type` equality (so the same company typed
`Bank` in one chunk and `Organization` in another never unioned on its shared identifier), while
`match_band` applied NO type check at all (so two same-block mentions of different types merged
outright). v16 replaces all three with: compatible iff identical or ancestor/descendant.
"""
import pytest

from core import Mention, cluster_mentions, identifier_values, match_band, ACCEPT, AMBIGUOUS
from ontologies import load_pack

FIBO = load_pack("fibo_core")
BANDS = {"fuzzy_floor": 0.80, "fuzzy_ceiling": 0.95}


def _m(uid, etype, surface, identifiers=None):
    m = Mention(entity_uid=uid, entity_type=etype, surface_form=surface)
    m.normalized_form = surface.lower()
    m.identifiers = identifiers or {}
    return m


# ---- the shared rule -------------------------------------------------------
@pytest.mark.ac("KG-AC-102")
def test_pack_exposes_compatibility_and_root():
    assert FIBO.is_compatible("Bank", "Bank")
    assert FIBO.is_compatible("Bank", "Organization"), "a descendant must be compatible with its ancestor"
    assert FIBO.is_compatible("Organization", "Bank"), "compatibility is symmetric"
    assert not FIBO.is_compatible("Bank", "Person"), "unrelated branches are not compatible"
    assert FIBO.root_of("Bank") == "Organization"
    assert FIBO.root_of("Organization") == "Organization"


# ---- tier 1: identifier collection + grouping ------------------------------
@pytest.mark.ac("KG-AC-24")
def test_identifier_collection_accepts_an_ancestor_domain():
    """KG-AC-70 parity: extraction's own domain gate accepts ancestors, so a property declared on
    `Organization` is legitimately extracted for a `Bank` — and canonicalization must then SEE it."""
    pack = _pack_with_org_identifier()
    attrs = [{"property": "orgId", "value": "X-1", "normalized_value": "X-1"}]
    assert identifier_values(attrs, "Bank", pack) == {"orgId": "x-1"}, (
        "an inherited identifier was invisible to clustering")


@pytest.mark.ac("KG-AC-24")
def test_tier1_unions_compatible_types_sharing_an_identifier():
    pack = _pack_with_org_identifier()
    ids = {"orgId": "x-1"}
    mentions = [
        _m("u1", "Bank", "First National", ids),
        _m("u2", "Organization", "First Natl", ids),
    ]
    clusters = cluster_mentions(mentions, pack=pack, **BANDS)
    assert len(clusters) == 1, "chunk-boundary typing variance split one identifier-bearing entity"


@pytest.mark.ac("KG-AC-102")
def test_tier1_does_not_union_incompatible_types_sharing_a_value():
    """A shared string across unrelated types is a collision, not an identity claim."""
    pack = _pack_with_org_identifier()
    ids = {"orgId": "x-1"}
    mentions = [_m("u1", "Bank", "Acme", ids), _m("u2", "Person", "Acme", ids)]
    assert len(cluster_mentions(mentions, pack=pack, **BANDS)) == 2


# ---- tier 3: the surface bands are type-gated ------------------------------
@pytest.mark.ac("KG-AC-102")
def test_exact_surface_equality_across_incompatible_types_is_never_an_accept():
    a, b = _m("u1", "Bank", "Jordan"), _m("u2", "Person", "Jordan")
    assert match_band(a, b, **BANDS) != ACCEPT, "two different things merged on a shared name"


@pytest.mark.ac("KG-AC-102")
def test_exact_surface_equality_within_compatible_types_still_accepts():
    """The pack is REQUIRED for a cross-type accept: without a hierarchy to consult, `match_band`
    degrades safely to exact-type-equality and returns AMBIGUOUS rather than guessing — it declines
    a merge it cannot justify, never invents one."""
    a, b = _m("u1", "Bank", "First National"), _m("u2", "Organization", "First National")
    assert match_band(a, b, pack=FIBO, **BANDS) == ACCEPT
    assert match_band(a, b, **BANDS) == AMBIGUOUS, "no pack must degrade safely, not over-merge"


# ---- clarify F7: identifier-less generic titles ----------------------------
@pytest.mark.ac("KG-AC-102")
def test_identifier_less_generic_title_is_adjudicated_not_auto_accepted():
    """Two distinct agreements both surfacing only as their type name must not silently collapse —
    the failure the review found for 'Investment Management Agreement'."""
    a = _m("u1", "Organization", "organization")
    b = _m("u2", "Organization", "organization")
    assert match_band(a, b, pack=FIBO, **BANDS) == AMBIGUOUS, (
        "a generic type-name surface auto-merged two identifier-less entities")


@pytest.mark.ac("KG-AC-102")
def test_a_distinctive_shared_surface_still_auto_accepts():
    a = _m("u1", "Organization", "acme capital partners")
    b = _m("u2", "Organization", "acme capital partners")
    assert match_band(a, b, pack=FIBO, **BANDS) == ACCEPT


def _pack_with_org_identifier():
    """A pack declaring an identifier property on the ANCESTOR type (Organization), so the
    inheritance behaviour above is exercised against a real loaded pack."""
    from ontologies.loader import DatatypeProperty
    pack = load_pack("fibo_core")
    pack.datatype_properties["orgId"] = DatatypeProperty(
        property="orgId", domain="Organization", range="identifier", guidance="")
    return pack
