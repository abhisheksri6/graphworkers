"""P12 (spec v13, KG-AC-79): ``canonical_key`` becomes a deterministic, human-readable
``<entity-type-slug>:<canonical-name-slug>`` identifier — the addressable, exportable identity
(``canonical_id`` UUID remains the actual storage key).

**Wording clarification, resolved and documented, not guessed at:** the AC's "canonical-name-slug"
component is built from ``normalized_form`` — the stable match key — NOT the `canonical_name`
COLUMN P10 writes. Reading it literally as `slug(canonical_name)` would contradict the SAME
sentence's own "stable across runs for the same cluster" requirement: P10 (already shipped,
tested) makes `canonical_name` deliberately MUTABLE across cross-run merges (a later document's
longer surface can promote it — see `test_aliases_grow_across_a_cross_run_merge`). It would also
create a chicken-and-egg problem: `_resolve_or_mint` needs a lookup key BEFORE `canonical_name` is
computed (which itself depends on the row `_resolve_or_mint` is about to create/find).
`normalized_form` has neither problem — stable per cluster, available pre-mint — and matches how
`canonical_key` already worked before this task (same basis, new readable format)."""
import pytest

from core import canonical_key, slugify


# ---- slugify ---------------------------------------------------------------------------------
@pytest.mark.ac("KG-AC-79")
def test_slugify_lowercases_and_collapses_non_alnum():
    assert slugify("Acme Corp.") == "acme-corp"


@pytest.mark.ac("KG-AC-79")
def test_slugify_collapses_runs_of_non_alnum_to_one_dash():
    assert slugify("Acme   &&&   Co.") == "acme-co"


@pytest.mark.ac("KG-AC-79")
def test_slugify_strips_leading_and_trailing_dashes():
    assert slugify("---Acme---") == "acme"


@pytest.mark.ac("KG-AC-79")
def test_slugify_empty_string():
    assert slugify("") == ""


# ---- canonical_key ------------------------------------------------------------------------------
@pytest.mark.ac("KG-AC-79")
def test_canonical_key_format_is_type_colon_name():
    key = canonical_key("Organization", "acme corp")
    assert key == "organization:acme-corp"


@pytest.mark.ac("KG-AC-79")
def test_canonical_key_stable_across_repeated_calls():
    a = canonical_key("Organization", "acme corp")
    b = canonical_key("Organization", "acme corp")
    assert a == b


@pytest.mark.ac("KG-AC-79")
def test_canonical_key_never_pipe_delimited_the_old_v11_format():
    # regression guard: the pre-v13 format was "<type>|<normalized_form>" -- must be gone.
    key = canonical_key("Organization", "acme corp")
    assert "|" not in key
    assert ":" in key


@pytest.mark.ac("KG-AC-79")
def test_canonical_key_collision_suffix_is_deterministic():
    base = canonical_key("Organization", "acme corp")
    suffixed = canonical_key("Organization", "acme corp", suffix=2)
    assert suffixed == f"{base}-2"
    # calling again with the same suffix must reproduce the SAME string (deterministic, not random)
    assert canonical_key("Organization", "acme corp", suffix=2) == suffixed
