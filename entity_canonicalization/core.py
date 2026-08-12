"""Pure canonicalization core — NO Celery/DB/network/LLM imports.

Deterministic given fixed input + a fixed adjudication decision: normalization, the ``canonical_key``,
the three-band match (KG-AC-24), union-find clustering (KG-AC-22/25), and type reconciliation
(KG-AC-23) all live here so they are unit-testable without a broker or DB. The LLM adjudication of the
AMBIGUOUS band is injected as a callback (D5); the DB index resolve/mint is store.py (KG-AC-38/40).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
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
    does; callers that don't compute a display name may omit them."""
    entity_uid: str
    entity_type: str
    surface_form: str
    normalized_form: str = ""
    source_doc_id: Optional[str] = None
    source_chunk_id: Optional[str] = None
    span_start: Optional[int] = None
    is_abstract: bool = False


def normalize_surface(surface: str) -> str:
    """Lowercase, strip accents + punctuation, drop legal suffixes, collapse whitespace."""
    s = unicodedata.normalize("NFKD", surface or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t and t not in _LEGAL_SUFFIXES]
    return " ".join(tokens)


def slugify(s: str) -> str:
    """Lowercase; collapse any run of non-alphanumeric characters to a single ``-``; strip leading/
    trailing ``-``. Pure string transform, no normalization semantics of its own (that is
    normalize_surface's job) — used to make an already-normalized string human-readable/URL-safe."""
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower())
    return s.strip("-")


def canonical_key(entity_type: str, normalized_form: str, suffix: int = 0) -> str:
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
    legitimate cross-run match (KG-AC-38, same normalized_form reusing the same key)."""
    key = f"{slugify(entity_type)}:{slugify(normalized_form)}"
    return f"{key}-{suffix}" if suffix else key


def fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_band(a: Mention, b: Mention, *, fuzzy_floor: float, fuzzy_ceiling: float) -> str:
    """Three-band match (KG-AC-24, *amended v11 — the LEI-equal short-circuit is removed with the
    gazetteer/external-id plane*): exact normalized ⇒ accept; fuzzy ≥ ceiling ⇒ accept; < floor ⇒
    reject; between ⇒ ambiguous (→ LLM)."""
    if a.normalized_form and a.normalized_form == b.normalized_form:
        return ACCEPT
    score = fuzzy_score(a.normalized_form, b.normalized_form)
    if score >= fuzzy_ceiling:
        return ACCEPT
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
    adjudicate: Optional[Callable[[Mention, Mention], bool]] = None,
) -> List[List[Mention]]:
    """Cluster mentions to one identity per real-world entity (KG-AC-22). Blocks by block_key, then
    within each block unions any pair that matches (ACCEPT, or AMBIGUOUS resolved true by
    ``adjudicate`` — the LLM; None ⇒ ambiguous pairs are NOT merged). Deterministic for a fixed
    adjudicate. Returns clusters in first-appearance order."""
    uf = _UnionFind([m.entity_uid for m in mentions])
    by_block: Dict[str, List[Mention]] = {}
    for m in mentions:
        by_block.setdefault(block_key(m), []).append(m)

    for block in by_block.values():
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                verdict = match_band(block[i], block[j], fuzzy_floor=fuzzy_floor, fuzzy_ceiling=fuzzy_ceiling)
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
    """KG-AC-78: merge facts across a canonicalised cluster's mentions, NEVER last-write-wins.
    ``mentions_attrs`` is one ``kg_entities.attributes`` list per mention (the
    ``{property, value, normalized_value, evidence, source_doc_id, page}`` shape
    ``attach_facts_to_entity_records`` already writes — no new shape invented).

    Groups by ``property``, then by ``normalized_value`` (the equality key, not raw ``value`` —
    mirrors KG-AC-70's own normalize-before-compare distinction: "15 March 2025" and "2025-03-15"
    are the same fact). One distinct value ⇒ ONE merged entry, ``conflicting=False``, its
    ``provenance`` list carrying every contributing source (deduplicated on exact
    (source_doc_id, page, evidence) repeats, so a re-extracted duplicate mention doesn't pad the
    list). Two or more distinct values ⇒ EVERY entry for that property is retained —
    ``conflicting=True`` on each — with provenance scoped to only the mentions that asserted THAT
    value; nothing is dropped, nothing is chosen as "the" answer.

    Deliberately builds no ``single_source``/``consistent`` distinction here — that needs counting
    DISTINCT SOURCE DOCUMENTS, which is KG-AC-80's own stated scope (P13), not this task's. Each
    entry's ``provenance`` list is already complete, so P13 extends this shape additively (reads
    the same list, adds the finer status) rather than reworking it."""
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
            entries.append({
                "value": facts[0]["value"], "normalized_value": nval,
                "conflicting": conflicting, "provenance": provenance,
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
    ``rows`` are dicts with ``folder_id``/``confidence``/``evidence_text`` — the caller (store.py)
    groups raw kg_edges rows by triple before calling this. Pure/deterministic given a fixed,
    stably-ordered input:
      - ``support_count`` = the number of DISTINCT source documents (folder_ids) asserting it —
        not the raw mention-edge count (a document repeating the same relation counts once).
      - ``confidence`` = the max across contributing edges (None if none carry one).
      - ``evidence_text`` = the top-3 sentences by confidence, descending; ties keep the input's
        relative order (stable sort). A row with no evidence_text is never selected (evidence is
        mandatory upstream, KG-AC-46 — defensive here, not expected to occur)."""
    support_count = len({r["folder_id"] for r in rows if r.get("folder_id")})
    confidences = [r["confidence"] for r in rows if r.get("confidence") is not None]
    confidence = max(confidences) if confidences else None
    with_evidence = [r for r in rows if r.get("evidence_text")]
    ranked = sorted(with_evidence, key=lambda r: -(r["confidence"] or 0.0))
    evidence_text = [r["evidence_text"] for r in ranked[:3]]
    return {"support_count": support_count, "confidence": confidence, "evidence_text": evidence_text}
