"""KG-AC-96 (P26) — **document-declared aliases**, canonicalization side: Tier 2 of the identity
hierarchy, deterministic, between the identifier tier (Tier 1) and the surface tiers (3–5).

**The production case this closes** (folder `36e398eb`): the document declares
`GLOBAL UNIVERSITY ENDOWMENT (hereinafter referred to as the "Investor")`. Coreference rewrote 31 of
33 party mentions (~94 percent), and the 2 leaked mentions — surface `"Investor"` and
`"Investment Manager"` — each minted a SPURIOUS canonical node. No amount of tuning the
probabilistic tiers can reach them: `"global university endowment"` and `"investor"` sit in
different `block_key` blocks (so are never compared) and score 0.23 fuzzy against a 0.80 floor.

**Document-scoping is semantic, not defensive.** A defined term binds only inside the document that
declares it — two documents may each define "the Investor" as a DIFFERENT party, and merging those
would be data corruption, not a recall win. Cross-document identity still flows through Tier 1
(identifier) and Tier 3 (normalised surface), reaching alias-only mentions transitively via
union-find.
"""
import pytest

from core import Mention, cluster_mentions, normalize_surface


def _m(uid, etype, surface, folder_id="f1", declared_aliases=None):
    m = Mention(entity_uid=uid, entity_type=etype, surface_form=surface, folder_id=folder_id)
    m.normalized_form = normalize_surface(surface)
    m.declared_aliases = list(declared_aliases or [])
    return m


def _cluster(mentions):
    return cluster_mentions(mentions, fuzzy_floor=0.80, fuzzy_ceiling=0.95, adjudicate=None)


# ---- the production regression ------------------------------------------------------------------
@pytest.mark.ac("KG-AC-96")
def test_declared_alias_bridges_a_surface_the_probabilistic_tiers_cannot_reach():
    full = _m("1", "Investor", "Global University Endowment", declared_aliases=["Investor", "the Investor"])
    leaked = _m("2", "Investor", "Investor")
    clusters = _cluster([full, leaked])
    assert len(clusters) == 1, [[m.surface_form for m in c] for c in clusters]


@pytest.mark.ac("KG-AC-96")
def test_alias_match_is_normalised_not_literal():
    # "the Investor" vs "Investor" must resolve. This is exactly what `normalize_alias` exists for
    # and what this test caught during P26: `normalize_surface` does NOT strip articles ("the" is
    # not a legal suffix), so the bridge silently failed on its own production case until
    # leading-article removal was added -- applied symmetrically to both sides of the lookup.
    full = _m("1", "Investor", "Global University Endowment", declared_aliases=["the Investor"])
    for variant in ("Investor", "INVESTOR", "  the   investor "):
        assert len(_cluster([full, _m("2", "Investor", variant)])) == 1, variant


@pytest.mark.ac("KG-AC-96")
def test_leading_determiners_not_just_articles_are_stripped():
    # Contractual English cites a defined term with any leading determiner: "the Investor",
    # "this Agreement", "such Party". Article-only stripping left "this Agreement" unmatched
    # against the declared term "Agreement" -- found on the real document during P26's end-to-end
    # replay, AFTER the article-only version had already fixed the party case.
    agreement = _m("1", "Agreement", "Investment Advisory and Fiduciary Management Agreement",
                   declared_aliases=["Agreement"])
    for variant in ("This Agreement", "this Agreement", "such Agreement", "the Agreement",
                    "said Agreement"):
        assert len(_cluster([agreement, _m("2", "Agreement", variant)])) == 1, variant


@pytest.mark.ac("KG-AC-96")
def test_determiner_stripping_cannot_merge_what_the_document_never_bound():
    # The guard on the rule above: stripping determiners only ever matters INSIDE a declared
    # binding. With no alias declared, "this Agreement" and the full title stay separate exactly
    # as before -- determiner handling widens the bridge, never the surface tiers.
    a = _m("1", "Agreement", "Investment Advisory and Fiduciary Management Agreement")
    b = _m("2", "Agreement", "This Agreement")
    assert len(_cluster([a, b])) == 2


# ---- document scoping ---------------------------------------------------------------------------
@pytest.mark.ac("KG-AC-96")
def test_an_alias_binds_only_inside_the_document_that_declares_it():
    # Two documents in one batch, each defining "the Investor" as a DIFFERENT party. Merging them
    # would be data corruption -- the whole reason the bridge is folder-scoped.
    doc1 = _m("1", "Investor", "Global University Endowment", folder_id="f1",
              declared_aliases=["the Investor"])
    doc2_leak = _m("2", "Investor", "Investor", folder_id="f2")
    assert len(_cluster([doc1, doc2_leak])) == 2


@pytest.mark.ac("KG-AC-96")
def test_cross_document_identity_still_reaches_alias_only_mentions_transitively():
    # doc1: "Investor" --(alias bridge, same doc)--> "Global University Endowment"
    # doc2: "Global University Endowment" --(Tier 3 exact surface)--> doc1's
    # Union-find composes the two, so doc2's mentions and doc1's alias-only leak land together
    # WITHOUT the alias ever crossing a document boundary itself.
    d1_full = _m("1", "Investor", "Global University Endowment", folder_id="f1",
                 declared_aliases=["the Investor"])
    d1_leak = _m("2", "Investor", "Investor", folder_id="f1")
    d2_full = _m("3", "Investor", "Global University Endowment", folder_id="f2")
    clusters = _cluster([d1_full, d1_leak, d2_full])
    assert len(clusters) == 1
    assert {m.entity_uid for m in clusters[0]} == {"1", "2", "3"}


# ---- guards -------------------------------------------------------------------------------------
@pytest.mark.ac("KG-AC-96")
def test_alias_bridge_never_crosses_entity_type():
    full = _m("1", "Investor", "Global University Endowment", declared_aliases=["the Party"])
    other = _m("2", "InvestmentManager", "The Party")
    assert len(_cluster([full, other])) == 2


@pytest.mark.ac("KG-AC-96")
def test_no_declared_aliases_reproduces_pre_p26_behaviour_exactly():
    a = _m("1", "Investor", "Global University Endowment")
    b = _m("2", "Investor", "Investor")
    # unreachable without a declared binding -- different blocks, 0.23 fuzzy vs a 0.80 floor
    assert len(_cluster([a, b])) == 2


@pytest.mark.ac("KG-AC-96")
def test_an_alias_equal_to_its_own_surface_is_harmless():
    a = _m("1", "Investor", "Investor", declared_aliases=["Investor"])
    assert len(_cluster([a])) == 1


@pytest.mark.ac("KG-AC-96")
def test_blank_and_non_string_aliases_are_ignored():
    full = _m("1", "Investor", "Global University Endowment", declared_aliases=["", "   "])
    leaked = _m("2", "Investor", "Investor")
    assert len(_cluster([full, leaked])) == 2  # nothing bindable -> no bridge, no crash


@pytest.mark.ac("KG-AC-96")
def test_alias_bridge_is_deterministic_under_input_reordering():
    full = _m("1", "Investor", "Global University Endowment", declared_aliases=["the Investor"])
    leaked = _m("2", "Investor", "Investor")
    assert len(_cluster([full, leaked])) == len(_cluster([leaked, full])) == 1


@pytest.mark.ac("KG-AC-96")
def test_identifier_conflict_still_outranks_a_declared_alias():
    # Tier 1's conflict rule is the stronger claim and must not be overridden by an alias binding:
    # two agreements with DIFFERENT ids stay apart even if one declares the other's surface.
    a = Mention(entity_uid="1", entity_type="Agreement", surface_form="Master Agreement", folder_id="f1")
    a.normalized_form = normalize_surface(a.surface_form)
    a.identifiers = {"agreementId": "a-1"}
    a.declared_aliases = ["The Agreement"]
    b = Mention(entity_uid="2", entity_type="Agreement", surface_form="The Agreement", folder_id="f1")
    b.normalized_form = normalize_surface(b.surface_form)
    b.identifiers = {"agreementId": "a-2"}
    assert len(_cluster([a, b])) == 2
