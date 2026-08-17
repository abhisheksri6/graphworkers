"""Pure canonicalization core — NO Celery/DB/network/LLM imports.

Deterministic given fixed input + a fixed adjudication decision: normalization, the ``canonical_key``,
the three-band match (KG-AC-24), union-find clustering (KG-AC-22/25), and type reconciliation
(KG-AC-23) all live here so they are unit-testable without a broker or DB. The LLM adjudication of the
AMBIGUOUS band is injected as a callback (D5); the DB index resolve/mint is store.py (KG-AC-38/40).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Legal/organizational suffixes stripped during normalization so "Acme Corp" == "Acme Corporation".
_LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "company",
    "plc", "na", "sa", "ag", "gmbh", "lp", "llp", "bv", "nv", "spa", "srl", "pty", "group", "holdings",
}

ACCEPT = "accept"
REJECT = "reject"
AMBIGUOUS = "ambiguous"


@dataclass
class Mention:
    """One staged entity mention being canonicalized. The provenance fields (KG-AC-76) are optional
    because clustering itself (block_key/match_band) never needs them — only choose_canonical_name
    does; callers that don't compute a display name may omit them.

    ``identifiers`` (KG-AC-24 amended, P25) maps a pack-declared identifier property to its
    normalized value (e.g. ``{"agreementId": "ima-gue-2026-101"}``) — populated by
    ``identifier_values`` from the mention's own extracted facts. Empty for identifier-less
    entities, which then cluster purely by the pre-P25 rules."""
    entity_uid: str
    entity_type: str
    surface_form: str
    normalized_form: str = ""
    source_doc_id: Optional[str] = None
    source_chunk_id: Optional[str] = None
    span_start: Optional[int] = None
    is_abstract: bool = False
    identifiers: Dict[str, str] = field(default_factory=dict)
    folder_id: Optional[str] = None  # KG-AC-96 (P26) — the document scope a declared alias binds
    # within. Always present in practice (it is the batch query's own filter); one folder == one
    # document in this pipeline, which makes it a stricter scope than source_doc_id (nullable when
    # chunk_metadata is incomplete, KG-AC-73).
    declared_aliases: List[str] = field(default_factory=list)  # KG-AC-96 (P26) — terms the DOCUMENT
    # explicitly binds to this entity, carried through from extraction. Never inferred.
    reference_only: bool = False  # KG-AC-94 — the document NAMED this entity (with an identifier)
    # but never described it. Read only by the canonical recompute (S6): a canonical entity is
    # reference_only iff EVERY contributing mention is. Never part of clustering.


def normalize_surface(surface: str) -> str:
    """Lowercase, strip accents + punctuation, drop legal suffixes, strip LEADING determiners,
    collapse whitespace.

    *(v16, KG-AC-101 — determiner stripping moved here from ``normalize_alias``.)* Contractual
    English cites the same entity as "The Acme Fund" and "Acme Fund" interchangeably, but with the
    determiner kept those two normalize differently, land in DIFFERENT blocks (`tok:the` vs
    `tok:acme`) and are therefore never even compared — no floor/ceiling tuning can reach a pair
    that is never a candidate. P26 already learned this for declared aliases and fixed it in
    ``normalize_alias`` alone; the asymmetry between the two was itself the bug."""
    s = unicodedata.normalize("NFKD", surface or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t and t not in _LEGAL_SUFFIXES]
    while tokens and tokens[0] in _LEADING_DETERMINERS:
        tokens.pop(0)
    return " ".join(tokens)


# Leading DETERMINERS, not just articles: contractual English cites a defined term as "the
# Investor", "this Agreement", "such Party" interchangeably with the bare term. Stripping only
# articles leaves "this agreement" unmatched against the declared term "Agreement" — found on the
# real document during P26's own end-to-end replay, after the article-only version had already
# fixed the party case. *(v16: consumed by `normalize_surface` itself — see KG-AC-101.)*
_LEADING_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "such", "said"}


def normalize_alias(value: str) -> str:
    """``normalize_surface`` plus leading-determiner removal (KG-AC-96, P26).

    A defined term is cited with and without its determiner interchangeably — the document writes
    *(the "Investor")* and then refers to *"Investor"*, or defines *"Agreement"* and then writes
    *"this Agreement"* — but ``normalize_surface`` keeps determiners (``the`` is not a legal
    suffix), so ``"the investor"`` and ``"investor"`` would not match. Found twice by this rule's
    own tests and the end-to-end replay: the bridge silently failed on the exact production cases
    it was written for. Applied to BOTH sides of the lookup so the comparison is symmetric.

    Scope keeps this safe: the bridge only fires for mentions of the SAME entity_type in the SAME
    document where one side explicitly DECLARED the term — a determiner-stripped collision cannot
    merge anything the document did not itself bind.

    *(v16, KG-AC-101: `normalize_surface` now strips leading determiners itself, so this is a thin
    alias kept for call-site clarity — the alias bridge and the surface tiers deliberately share
    ONE normalization, which is exactly what the pre-v16 asymmetry got wrong.)*"""
    return normalize_surface(value)


def _identifiers_conflict(a: Mention, b: Mention) -> bool:
    """True iff both mentions declare the SAME identifier property with DIFFERENT values — the
    decisive negative-evidence rule (KG-AC-24 amended). Shared by ``match_band`` and the Tier-2
    alias bridge so a declared alias can never override a hard identifier mismatch."""
    shared = set(a.identifiers or {}) & set(b.identifiers or {})
    return any(a.identifiers[p] != b.identifiers[p] for p in shared)


def normalize_identifier(value: str) -> str:
    """Casefold + collapse whitespace, and NOTHING else (KG-AC-24 amended, P25).

    Deliberately NOT ``normalize_surface``: an identifier is an **opaque token**, not a name. Its
    punctuation is significant, and its tokens are not company-name parts — running it through the
    legal-suffix stripper mangles real identifiers (``LP-2026-001`` → ``2026 001``, because ``lp``
    is in ``_LEGAL_SUFFIXES``). Applying name-normalization to an identifier is the same category
    error as treating a declared alias as coreference."""
    return " ".join((value or "").split()).casefold()


def identifier_values(attributes: Sequence[Dict[str, Any]], entity_type: str, pack) -> Dict[str, str]:
    """The pack-declared IDENTIFIER facts carried by one mention (KG-AC-24 amended, P25):
    ``{property: normalized_value}`` for every fact whose property the pack declares with
    ``range="identifier"`` AND ``domain=entity_type``.

    Domain-scoping is load-bearing, not decoration: it stops an ``Agreement`` and a
    ``Subscription`` that happen to carry the same raw string from being treated as one identity.
    No pack, no attributes, or a pack declaring no ``datatype_properties`` (``generic``,
    ``fibo_core``, ``fibo_custom``) all yield ``{}`` — never an error, so identifier-less packs
    keep exactly their pre-P25 behaviour."""
    if pack is None or not attributes:
        return {}
    declared = getattr(pack, "datatype_properties", None) or {}
    out: Dict[str, str] = {}
    # v16 (KG-AC-102): the domain gate accepts an ANCESTOR domain, matching extraction's own gate
    # (KG-AC-70). Pre-v16 this demanded `dp.domain == entity_type` exactly, so a `lei` declared on
    # `Organization` was silently invisible for a mention typed `Bank` — the identifier tier simply
    # never saw the strongest evidence it exists to use.
    ancestors = set(pack.ancestors(entity_type)) if hasattr(pack, "ancestors") else set()
    for fact in attributes:
        prop = (fact or {}).get("property")
        dp = declared.get(prop)
        if dp is None or dp.range != "identifier":
            continue
        if dp.domain != entity_type and dp.domain not in ancestors:
            continue
        raw = fact.get("normalized_value") or fact.get("value")
        norm = normalize_identifier(raw)
        if norm:
            out[prop] = norm
    return out


def cluster_identifier(cluster: Sequence[Mention], pack=None) -> Optional[str]:
    """The identity basis for a cluster that carries a pack-declared identifier (KG-AC-79 amended,
    P25) — used by store.py as the ``normalized_form``/``canonical_key`` basis so cross-run reuse
    keys on the IDENTIFIER, not on whichever surface happened to sort first.

    Without this, Tier-1 merging would fix within-batch fragmentation but BREAK cross-run identity
    (KG-AC-38): two runs whose clusters contain different surface subsets would pick different
    ``cluster[0].normalized_form`` values and therefore mint different canonical_keys for the same
    real entity. Deterministic under input reordering (``min`` over all values, not positional) —
    a well-formed cluster carries exactly one value; ``min`` only matters defensively, if the fuzzy
    or LLM tier merged two identifier-bearing mentions.

    ***(v16, KG-AC-103.)*** With a ``pack``, the basis is chosen by the pack's DECLARATION ORDER of
    the identifier properties, not by `min` over raw values. Two runs that extract different
    identifier SUBSETS of one entity (run 1 sees `agreementId`, run 2 sees `agreementId` + `lei`)
    would otherwise pick different minima and mint two canonical keys for one real entity — the
    same cross-run split KG-AC-38 exists to prevent, reintroduced through the tie-break."""
    if pack is not None and getattr(pack, "datatype_properties", None):
        for prop in pack.datatype_properties:  # declaration order
            values = sorted({(m.identifiers or {}).get(prop) for m in cluster
                             if (m.identifiers or {}).get(prop)})
            if values:
                return values[0]
    values = sorted({v for m in cluster for v in (m.identifiers or {}).values() if v})
    return values[0] if values else None


def slugify(s: str) -> str:
    """Lowercase; collapse any run of non-alphanumeric characters to a single ``-``; strip leading/
    trailing ``-``. Pure string transform, no normalization semantics of its own (that is
    normalize_surface's job) — used to make an already-normalized string human-readable/URL-safe."""
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower())
    return s.strip("-")


def canonical_key(entity_type: str, normalized_form: str, suffix: int = 0, *,
                  pack=None, fallback_surface: str = "") -> str:
    """KG-AC-79 (v13 — amends the v11 pipe-delimited format): the UNIQUE key on
    kg_canonical_entities, now a deterministic, human-readable ``<entity-type-slug>:
    <canonical-name-slug>`` identifier — "the addressable, exportable identity" (the AC's own
    words); ``canonical_id`` (a UUID) remains the actual storage key.

    Built from ``normalized_form``, NOT the `canonical_name` column P10 writes (full reasoning in
    `test_canonical_key.py`'s module docstring): `canonical_name` is deliberately mutable across
    cross-run merges (P10), which would contradict this AC's own "stable across runs" requirement
    in the same sentence, and creates a chicken-and-egg problem (`_resolve_or_mint` needs this key
    BEFORE `canonical_name` exists). `normalized_form` has neither problem and is the SAME basis
    `canonical_key` already used before this task — only the output format changes.

    Uses the RECONCILED cluster type so chunk-boundary type inconsistency (F-CHUNK-2) doesn't split
    one real entity (unchanged from the prior format).

    ``suffix`` (KG-AC-79's collision requirement): 0 omits it; >0 appends ``-<suffix>``,
    deterministically — store.py's `_resolve_or_mint` increments it only when a computed key
    already belongs to a DIFFERENT normalized_form (a genuine slug collision), never for a
    legitimate cross-run match (KG-AC-38, same normalized_form reusing the same key).

    ***(v16, KG-AC-103.)*** The type half is the pack hierarchy's **ROOT** type when a ``pack`` is
    supplied (`organization:acme` for every `Organization` descendant): the reconciled type of one
    real entity legitimately sharpens across batches — `Organization` in a document that names it
    plainly, `InvestmentAdviser` in one that does not — and keying on the specific type mints a
    second canonical node each time it moves. The reconciled specific type stays on the row as
    data. No pack ⇒ the bare type, unchanged.

    ***(v16, KG-AC-101.)*** ``fallback_surface`` is used when ``normalized_form`` slugs EMPTY (a
    suffix-only or determiner-only surface): `organization:` would otherwise be one shared key for
    every degenerate surface in the corpus. The fallback is the plain lowercased surface, which is
    unclusterable-but-distinct rather than silently shared."""
    type_slug = slugify(pack.root_of(entity_type) if pack is not None else entity_type)
    name_slug = slugify(normalized_form) or slugify(fallback_surface)
    key = f"{type_slug}:{name_slug}"
    return f"{key}-{suffix}" if suffix else key


def fuzzy_score(a: str, b: str) -> float:
    """Similarity of two NORMALIZED forms. An empty side short-circuits to 0.0 (KG-AC-101): legal-
    suffix stripping legitimately empties surfaces like "Ltd" or "The Company", and
    ``SequenceMatcher("", "").ratio()`` is **1.0** — so without this guard every degenerate surface
    in the corpus auto-accepted against every other one and collapsed into a single junk canonical
    entity. An unclusterable surface must simply never match, not match everything."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def types_compatible(a_type: str, b_type: str, pack=None) -> bool:
    """KG-AC-102's single rule, usable without a pack. With a pack: identical or ancestor/descendant
    per the declared hierarchy. Without one: identical only — the SAFE degradation, since it can
    never over-merge (it only declines pairs the hierarchy would have allowed)."""
    if a_type == b_type:
        return True
    if pack is None or not hasattr(pack, "is_compatible"):
        return False
    return pack.is_compatible(a_type, b_type)


def _is_generic_surface(normalized: str, pack) -> bool:
    """True when a normalized surface is just a pack TYPE NAME ("agreement", "organization") —
    clarify F7. Such a surface identifies nothing: two distinct agreements each titled only
    "Investment Management Agreement" are routine in this domain, and auto-accepting their exact
    match silently collapses them into one canonical entity."""
    if pack is None or not normalized:
        return False
    return any(normalize_surface(t) == normalized for t in getattr(pack, "entity_types", {}))


def match_band(a: Mention, b: Mention, *, fuzzy_floor: float, fuzzy_ceiling: float, pack=None) -> str:
    """Three-band match (KG-AC-24, *amended v11 — the LEI-equal short-circuit is removed with the
    gazetteer/external-id plane*; ***re-introduced on a pack-declared basis at P25***): a shared
    identifier property decides the match OUTRIGHT, in both directions, before any surface
    comparison — equal values ⇒ accept, DIFFERENT values ⇒ reject.

    The reject direction is the load-bearing half and is not symmetric decoration: a mismatched
    strong identifier is decisive NEGATIVE evidence. Without it, fifty distinct agreements all
    titled "Investment Management Agreement" — routine in this domain — would collapse into ONE
    canonical entity on exact surface equality, each one's identifier silently overruled by its
    generic title. Found by this rule's own test at implementation (P25).

    Then the surface bands, unchanged: exact normalized ⇒ accept; fuzzy ≥ ceiling ⇒ accept;
    < floor ⇒ reject; between ⇒ ambiguous (→ LLM)."""
    # v16 (KG-AC-102): ONE type gate in front of every positive decision. Incompatible types are
    # not the same entity however identical their strings — pre-v16 this function applied NO type
    # check at all, so "Jordan" the person merged with "Jordan" the place on exact equality.
    compatible = types_compatible(a.entity_type, b.entity_type, pack)

    if set(a.identifiers or {}) & set(b.identifiers or {}):
        if _identifiers_conflict(a, b):
            return REJECT
        # A shared value across INCOMPATIBLE types is a string collision, not an identity claim.
        return ACCEPT if compatible else REJECT

    if a.normalized_form and a.normalized_form == b.normalized_form:
        if not compatible:
            return AMBIGUOUS  # adjudicator-gated, never an unconditional accept
        # clarify F7: an identifier-LESS pair whose shared surface is merely a pack type name
        # identifies nothing — the adjudicator decides, in-batch and at cross-run resolution alike.
        if _is_generic_surface(a.normalized_form, pack):
            return AMBIGUOUS
        return ACCEPT

    score = fuzzy_score(a.normalized_form, b.normalized_form)
    if score >= fuzzy_ceiling:
        return ACCEPT if compatible else AMBIGUOUS
    if score < fuzzy_floor:
        return REJECT
    return AMBIGUOUS


def block_key(m: Mention) -> str:
    """Coarse candidate-generation key (blocking): the first token of the normalized form (*amended
    v11 — the LEI branch is removed with the gazetteer/external-id plane*). Bounds pairwise
    comparison to likely matches (KG-AC-25 — batch-scoped)."""
    first = m.normalized_form.split(" ", 1)[0] if m.normalized_form else ""
    return f"tok:{first}"


class _UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_mentions(
    mentions: List[Mention], *, fuzzy_floor: float, fuzzy_ceiling: float,
    adjudicate: Optional[Callable[[Mention, Mention], bool]] = None, pack=None,
) -> List[List[Mention]]:
    """Cluster mentions to one identity per real-world entity (KG-AC-22), in two tiers:

    **Tier 1 — identifier equality (KG-AC-24 amended, P25).** Mentions of the SAME entity_type
    sharing a pack-declared identifier value are unioned unconditionally, **bypassing blocking,
    the fuzzy bands and the LLM entirely**. Deterministic and free (the identifiers were already
    extracted as facts). This tier exists because the other one structurally cannot reach these
    cases: the four surfaces of one real agreement (``"Agreement"`` / ``"This Agreement"`` /
    ``"Investment Advisory and Fiduciary Management Agreement"`` / ``"… IMA-GUE-2026-101"``) land
    in four different blocks, so they are never even COMPARED, and score 0.15–0.38 against a 0.80
    floor, so they would be rejected if they were. Found live 2026-08-12: one agreement exported to
    Neo4j as four nodes, with three of them already carrying the identical ``agreementId``.

    **Tier 2 — blocking + the three-band surface match (unchanged).** Blocks by block_key, then
    within each block unions any pair that matches (ACCEPT, or AMBIGUOUS resolved true by
    ``adjudicate`` — the LLM; None ⇒ ambiguous pairs are NOT merged).

    Deterministic for a fixed adjudicate. Returns clusters in first-appearance order."""
    uf = _UnionFind([m.entity_uid for m in mentions])

    # Tier 1 — keyed by (property, value), then unioned only across COMPATIBLE types (KG-AC-102).
    # Pre-v16 the key included the exact `entity_type`, so the same company typed `Bank` in one
    # chunk and `Organization` in another never unioned on its shared identifier — ordinary
    # chunk-boundary typing variance defeating the strongest identity evidence available. The
    # compatibility gate keeps the protection the exact key was really providing: a value shared by
    # an Agreement and a Person is still a collision, not an identity.
    by_identifier: Dict[Tuple[str, str], List[Mention]] = {}
    for m in mentions:
        for prop, value in (m.identifiers or {}).items():
            by_identifier.setdefault((prop, value), []).append(m)
    for group in by_identifier.values():
        for other in group[1:]:
            if types_compatible(group[0].entity_type, other.entity_type, pack):
                uf.union(group[0].entity_uid, other.entity_uid)

    # Tier 2 — document-declared aliases (KG-AC-96). A mention whose normalized surface matches a
    # term another mention's document explicitly BOUND to itself is that entity, deterministically.
    # Keyed by (folder_id, entity_type, normalized alias): the folder scope is semantic, not
    # defensive — a defined term binds only inside its own document, and two documents may each
    # define "the Investor" as a different party. Cross-document identity is Tier 1/3's job and
    # composes with this transitively through the shared union-find.
    by_declared_alias: Dict[Tuple[Optional[str], str, str], List[Mention]] = {}
    for m in mentions:
        for alias in (m.declared_aliases or []):
            norm = normalize_alias(alias)
            if norm:
                by_declared_alias.setdefault((m.folder_id, m.entity_type, norm), []).append(m)
    if by_declared_alias:
        for m in mentions:
            key = (m.folder_id, m.entity_type, normalize_alias(m.surface_form))
            if not key[2]:
                continue
            for owner in by_declared_alias.get(key, ()):
                # A declared alias is strong evidence, but NOT stronger than a hard identifier
                # mismatch: Tier 1's conflict rule outranks it (two agreements with different ids
                # stay apart even if one's declared term matches the other's surface).
                if owner.entity_uid != m.entity_uid and not _identifiers_conflict(owner, m):
                    uf.union(owner.entity_uid, m.entity_uid)

    by_block: Dict[str, List[Mention]] = {}
    for m in mentions:
        by_block.setdefault(block_key(m), []).append(m)

    for block in by_block.values():
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                verdict = match_band(block[i], block[j], fuzzy_floor=fuzzy_floor,
                                     fuzzy_ceiling=fuzzy_ceiling, pack=pack)
                if verdict == ACCEPT or (verdict == AMBIGUOUS and adjudicate and adjudicate(block[i], block[j])):
                    uf.union(block[i].entity_uid, block[j].entity_uid)

    clusters: Dict[str, List[Mention]] = {}
    for m in mentions:  # first-appearance order preserved
        clusters.setdefault(uf.find(m.entity_uid), []).append(m)
    return list(clusters.values())


def choose_canonical_name(mentions: List[Mention]) -> Tuple[str, List[str]]:
    """KG-AC-76/77: ``canonical_name`` = the cluster's longest complete surface form, tie-broken by
    mention frequency (how many rows share that exact surface), then by earliest
    ``(source_doc_id, source_chunk_id, span_start)``. ``aliases`` = every OTHER distinct surface,
    ordered by descending frequency then alphabetically — together, ``{canonical_name} | aliases``
    equals exactly the cluster's distinct surface set (KG-AC-77's auditability purpose: "a reader
    can see every string that resolved to this instance").

    Deterministic under any input ordering (aggregates by surface across the whole list, not
    position-dependent). Abstract instances (KG-AC-90) need no special case: their ``surface_form``
    is already the document-printed identity value, never a composed name, and an abstract
    cluster's mentions share that identical surface by construction — the general algorithm
    preserves it for free. ``normalized_form`` (canonical_key's match key) is never read here."""
    if not mentions:
        raise ValueError("choose_canonical_name requires at least one mention")

    by_surface: Dict[str, List[Mention]] = {}
    for m in mentions:
        by_surface.setdefault(m.surface_form, []).append(m)

    def earliest_key(ms: List[Mention]) -> Tuple:
        # None sorts LAST within each component — a located mention beats an unlocated one.
        return min(
            ((m.source_doc_id is None, m.source_doc_id or ""),
             (m.source_chunk_id is None, m.source_chunk_id or ""),
             (m.span_start is None, m.span_start or 0))
            for m in ms
        )

    def rank(surface: str) -> Tuple:
        ms = by_surface[surface]
        return (-len(surface), -len(ms), earliest_key(ms))

    ordered = sorted(by_surface.keys(), key=rank)
    canonical_name, remaining = ordered[0], ordered[1:]
    aliases = sorted(remaining, key=lambda s: (-len(by_surface[s]), s))
    return canonical_name, aliases


def merge_attributes(mentions_attrs: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """KG-AC-78/80: merge facts across a canonicalised cluster's mentions, NEVER last-write-wins.
    ``mentions_attrs`` is one ``kg_entities.attributes`` list per mention (the
    ``{property, value, normalized_value, evidence, source_doc_id, page}`` shape
    ``attach_facts_to_entity_records`` already writes — no new shape invented).

    Groups by ``property``, then by ``normalized_value`` (the equality key, not raw ``value`` —
    mirrors KG-AC-70's own normalize-before-compare distinction: "15 March 2025" and "2025-03-15"
    are the same fact). Each entry's ``provenance`` list carries every contributing source
    (deduplicated on exact (source_doc_id, page, evidence) repeats, so a re-extracted duplicate
    mention doesn't pad the list) and a **status** (KG-AC-80, the full 3-state vocabulary):
      - ``conflicting`` — TWO OR MORE distinct normalized values exist for this property, however
        lopsided the split (clarify 2026-08-11, baked into the AC text) — EVERY entry is retained,
        each scoped to only the mentions that asserted THAT value; nothing is dropped, nothing is
        chosen as "the" answer.
      - ``single_source`` — exactly one distinct value, asserted by only ONE distinct source
        document (0 known documents also defaults here — never invents a 4th status).
      - ``consistent`` — exactly one distinct value, asserted by TWO OR MORE distinct source
        documents, all agreeing. Deliberately NOT used for the single-document case, which would
        overstate the evidence of one unchallenged reading (KG-AC-80's own stated rationale)."""
    # property -> normalized_value -> [source facts]
    by_property: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for attrs in mentions_attrs:
        for fact in attrs:
            prop = fact["property"]
            nval = fact.get("normalized_value") or fact["value"]
            by_property.setdefault(prop, {}).setdefault(nval, []).append(fact)

    merged: Dict[str, List[Dict[str, Any]]] = {}
    for prop, by_value in by_property.items():
        conflicting = len(by_value) > 1
        entries = []
        for nval, facts in by_value.items():
            seen_prov: set = set()
            provenance = []
            for f in facts:
                key = (f.get("source_doc_id"), f.get("page"), f.get("evidence"))
                if key in seen_prov:
                    continue
                seen_prov.add(key)
                provenance.append({"source_doc_id": f.get("source_doc_id"), "page": f.get("page"),
                                   "evidence": f.get("evidence")})
            if conflicting:
                status = "conflicting"
            else:
                distinct_docs = {p["source_doc_id"] for p in provenance if p["source_doc_id"]}
                status = "consistent" if len(distinct_docs) >= 2 else "single_source"
            entries.append({
                "value": facts[0]["value"], "normalized_value": nval,
                "status": status, "provenance": provenance,
            })
        merged[prop] = entries
    return merged


def reconcile_type(types: Sequence[str], pack) -> Optional[str]:
    """KG-AC-23: the most-specific type across a cluster per the pack's declared parent hierarchy
    (a descendant beats its ancestor; cross-branch → declaration order). Falls back to the first
    type when the pack can't resolve (e.g. all unknown)."""
    resolved = pack.most_specific_type(types)
    if resolved is not None:
        return resolved
    return types[0] if types else None


def aggregate_edge_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """KG-AC-47: collapse a group of mention-edges that all share ONE canonical
    (src_canonical_id, relation_type, dst_canonical_id) triple into the aggregated canonical edge.
    ``rows`` are dicts with ``folder_id``/``confidence``/``evidence_text``/``source_doc_id`` — the
    caller (store.py) groups raw kg_edges rows by triple before calling this. Pure/deterministic
    given a fixed, stably-ordered input:
      - ``support_count`` = the number of DISTINCT source documents (folder_ids) asserting it —
        not the raw mention-edge count (a document repeating the same relation counts once).
      - ``confidence`` = the max across contributing edges (None if none carry one).
      - ``evidence_text`` = the top-3 sentences by confidence, descending; ties keep the input's
        relative order (stable sort). A row with no evidence_text is never selected (evidence is
        mandatory upstream, KG-AC-46 — defensive here, not expected to occur).
      - ``source_doc_ids`` (KG-AC-80, v13) = the sorted, deduplicated set of ``source_doc_id``
        values across contributing rows (``None`` never enters the set) — the canonical edge's own
        contributing-document set, DELIBERATELY on a different basis than ``support_count``
        (folder_id, unchanged from KG-AC-47) — a folder and a document are not the same concept."""
    support_count = len({r["folder_id"] for r in rows if r.get("folder_id")})
    confidences = [r["confidence"] for r in rows if r.get("confidence") is not None]
    confidence = max(confidences) if confidences else None
    with_evidence = [r for r in rows if r.get("evidence_text")]
    ranked = sorted(with_evidence, key=lambda r: -(r["confidence"] or 0.0))
    evidence_text = [r["evidence_text"] for r in ranked[:3]]
    source_doc_ids = sorted({r["source_doc_id"] for r in rows if r.get("source_doc_id")})
    return {"support_count": support_count, "confidence": confidence, "evidence_text": evidence_text,
           "source_doc_ids": source_doc_ids}
