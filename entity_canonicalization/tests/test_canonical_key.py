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


# ---- v16 (KG-AC-103): root-type slug + declaration-order identifier basis ---------------------
@pytest.mark.ac("KG-AC-103")
def test_key_type_half_is_the_hierarchy_root_not_the_specific_type():
    """One real entity legitimately sharpens its type across batches (`Organization` in a document
    that names it plainly, `InvestmentAdviser` in one that does not). Keying on the SPECIFIC type
    mints a second canonical node every time that happens - F-CHUNK-2 resurfacing at the batch
    boundary. The root keeps both batches on one key; the specific type stays on the row as data."""
    from ontologies import load_pack
    pack = load_pack("fibo_core")
    assert pack.root_of("InvestmentAdviser") == "Organization"
    specific = canonical_key("InvestmentAdviser", "acme corp", pack=pack)
    generic = canonical_key("Organization", "acme corp", pack=pack)
    assert specific == generic == "organization:acme-corp"
    # without a pack the bare type is used, unchanged from the pre-v16 format
    assert canonical_key("InvestmentAdviser", "acme corp") == "investmentadviser:acme-corp"


@pytest.mark.ac("KG-AC-103")
def test_identifier_basis_follows_pack_declaration_order_not_value_order():
    """Two runs extracting different identifier SUBSETS of one entity must still key identically.
    Picking `min` over raw values makes the basis depend on WHICH identifiers a run happened to
    see, so run 1 (agreementId only) and run 2 (agreementId + a lexically smaller lei) would mint
    two canonical keys for one real entity - the cross-run split KG-AC-38 exists to prevent."""
    from core import Mention, cluster_identifier
    from ontologies import load_pack
    from ontologies.loader import DatatypeProperty

    pack = load_pack("fibo_core")
    pack.datatype_properties.clear()
    for name in ("agreementId", "lei"):  # declaration order: agreementId wins
        pack.datatype_properties[name] = DatatypeProperty(
            property=name, domain="Organization", range="identifier", guidance="")

    def _m(uid, ids):
        m = Mention(entity_uid=uid, entity_type="Organization", surface_form="x")
        m.identifiers = ids
        return m

    run1 = [_m("u1", {"agreementId": "ima-2026-101"})]
    run2 = [_m("u2", {"agreementId": "ima-2026-101", "lei": "aaa-smaller-than-ima"})]
    assert cluster_identifier(run1, pack) == cluster_identifier(run2, pack) == "ima-2026-101"
    # the pre-v16 min-over-values basis is what would have split them
    assert cluster_identifier(run2) == "aaa-smaller-than-ima"
