"""P10 (spec v13, KG-AC-76): the canonical entity's display name — deterministic, chosen from the
cluster's own mentions: the longest complete surface form, tie-broken by mention frequency, then by
earliest ``(source_doc_id, source_chunk_id, span_start)``. ``normalized_form`` remains the match
key (core.canonical_key) and is never repurposed as a display value.

Abstract instances (KG-AC-90, v15 derivation) need no special-case code: their ``surface_form`` IS
already the document-printed identity value, never a composed/synthesised name — the general
algorithm preserves it naturally, since an abstract cluster's mentions share the identical surface
by construction (exact-normalized match is required to cluster identity values in the first place).
KG-AC-76's "model-synthesised name" language describes the v13 mechanism v15 superseded; corrected
in place alongside this task (same posture as the KG-AC-72 correction)."""
import pytest

from core import Mention, choose_canonical_name


def _m(surface, doc=None, chunk=None, span=None, abstract=False):
    return Mention(entity_uid=f"u-{surface}-{doc}-{chunk}-{span}", entity_type="Organization",
                   surface_form=surface, source_doc_id=doc, source_chunk_id=chunk,
                   span_start=span, is_abstract=abstract)


@pytest.mark.ac("KG-AC-76")
def test_longest_surface_wins():
    name, _ = choose_canonical_name([_m("Acme"), _m("Acme Corporation"), _m("Acme Corp")])
    assert name == "Acme Corporation"


@pytest.mark.ac("KG-AC-76")
def test_tie_broken_by_mention_frequency():
    # "Acme Corp." and "Acme Corpn" are both 10 chars -- same length, frequency decides.
    mentions = [_m("Acme Corp."), _m("Acme Corp."), _m("Acme Corp."), _m("Acme Corpn")]
    name, _ = choose_canonical_name(mentions)
    assert name == "Acme Corp."


@pytest.mark.ac("KG-AC-76")
def test_tie_broken_by_earliest_doc_then_chunk_then_span():
    # same length, same frequency (1 each) -- earliest (doc, chunk, span) decides.
    mentions = [
        _m("Acme Corpn", doc="d2", chunk="c1", span=0),
        _m("Acme Corp.", doc="d1", chunk="c1", span=5),  # earliest doc
    ]
    name, _ = choose_canonical_name(mentions)
    assert name == "Acme Corp."


@pytest.mark.ac("KG-AC-76")
def test_deterministic_under_reordering():
    mentions = [_m("Acme"), _m("Acme Corporation"), _m("Acme Corp"), _m("Acme Corp")]
    a, _ = choose_canonical_name(mentions)
    b, _ = choose_canonical_name(list(reversed(mentions)))
    assert a == b == "Acme Corporation"


@pytest.mark.ac("KG-AC-76")
def test_single_mention_cluster():
    name, aliases = choose_canonical_name([_m("Solo Corp")])
    assert name == "Solo Corp" and aliases == []


@pytest.mark.ac("KG-AC-76")
def test_normalized_form_never_used_as_the_display_value():
    # normalize_surface would lowercase + strip the "Corp" suffix -- canonical_name must be the
    # SURFACE form, never that normalized string.
    name, _ = choose_canonical_name([_m("ACME CORP")])
    assert name == "ACME CORP"


@pytest.mark.ac("KG-AC-76")
def test_empty_cluster_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        choose_canonical_name([])


# ---- abstract instances (KG-AC-90 derived identity) ----------------------------------------------
@pytest.mark.ac("KG-AC-76")
def test_abstract_cluster_canonical_name_is_the_identity_value():
    # the derived hub's surface_form IS the document-printed identity (KG-AC-90) -- never a
    # composed name -- so it is already the correct display value with no special-casing needed.
    mentions = [_m("IMA-2025-018", abstract=True), _m("IMA-2025-018", abstract=True)]
    name, aliases = choose_canonical_name(mentions)
    assert name == "IMA-2025-018"
    assert aliases == []
