"""specs/knowledge-graph v16 — S4: normalization guards (KG-AC-101).

Two defects found by the 2026-08-17 review, both of which no threshold tuning could ever reach:

1. **Determiner asymmetry.** `normalize_alias` strips leading determiners (its docstring records
   the rule failing twice on real documents), but `normalize_surface` did not — so "The Acme Fund"
   blocked under `tok:the` and "Acme Fund" under `tok:acme`, were never even COMPARED, and keyed
   differently. Contractual English writes both forms interchangeably.

2. **Degenerate normalization.** Legal-suffix stripping turns "Ltd" into `""`, and
   `SequenceMatcher("", "").ratio()` is **1.0** — so every suffix-only surface of a type auto-merged
   into one junk canonical node, across all documents, via the fuzzy band and a shared `tok:` block.
"""
import pytest

from core import (
    ACCEPT, Mention, block_key, canonical_key, fuzzy_score, match_band, normalize_surface,
)

BANDS = {"fuzzy_floor": 0.80, "fuzzy_ceiling": 0.95}


def _m(uid, surface, etype="Organization"):
    m = Mention(entity_uid=uid, entity_type=etype, surface_form=surface)
    m.normalized_form = normalize_surface(surface)
    return m


# ---- 1. determiners --------------------------------------------------------
@pytest.mark.ac("KG-AC-101")
@pytest.mark.parametrize("pair", [
    ("The Acme Fund", "Acme Fund"),
    ("this Agreement", "Agreement"),
    ("such Party", "Party"),
])
def test_determiner_variants_normalize_and_block_identically(pair):
    a, b = pair
    assert normalize_surface(a) == normalize_surface(b), f"{a!r} and {b!r} still differ"
    assert block_key(_m("u1", a)) == block_key(_m("u2", b)), (
        "the pair lands in different blocks, so it is never compared at all")


@pytest.mark.ac("KG-AC-101")
def test_determiner_variants_reach_an_accept():
    assert match_band(_m("u1", "The Acme Fund"), _m("u2", "Acme Fund"), **BANDS) == ACCEPT


@pytest.mark.ac("KG-AC-101")
def test_a_determiner_is_only_stripped_when_leading():
    # "The" inside a name is part of it, not a determiner to discard
    assert normalize_surface("Bank of The West") == normalize_surface("Bank of The West")
    assert "west" in normalize_surface("Bank of The West")


# ---- 2. degenerate normalization ------------------------------------------
@pytest.mark.ac("KG-AC-101")
def test_empty_normalization_never_scores_as_a_match():
    assert normalize_surface("Ltd") == "", "precondition: a suffix-only surface normalizes empty"
    assert fuzzy_score("", "") == 0.0, "empty-vs-empty scored as a perfect match"
    assert fuzzy_score("", "acme") == 0.0


@pytest.mark.ac("KG-AC-101")
def test_two_suffix_only_surfaces_do_not_merge():
    """Pre-v16 these shared block `tok:` and scored 1.0 — collapsing every such surface in the
    corpus into a single nonsense canonical entity."""
    assert match_band(_m("u1", "Ltd"), _m("u2", "Inc"), **BANDS) != ACCEPT


@pytest.mark.ac("KG-AC-101")
def test_canonical_key_falls_back_when_the_normalized_form_is_empty():
    """An empty slug must never be keyed — `organization:` would be one shared key for every
    degenerate surface. The fallback basis is the plain lowercased surface."""
    key = canonical_key("Organization", "", fallback_surface="Ltd")
    assert key.endswith(":ltd"), key
    assert not key.endswith(":"), "an empty slug was keyed"
