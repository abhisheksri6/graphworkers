"""Pure extraction core — NO Celery, DB, network, or spaCy imports.

Deterministic given fixed strategy output: the layer-precedence merge (KG-AC-12), the bounded
top-N promotion (KG-AC-37), the deterministic uids (KG-AC-10), and the state-scalar summary
(KG-AC-9) all live here so they are unit-testable without a broker, model, or DB. Strategy
execution (regex / spaCy / LLM) lives in ``strategies/`` and is injected — this module only
transforms the candidates strategies produce.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Sequence, Tuple

# Layer precedence (ADR-0009 / KG-AC-12; evolve v8 ADR-0012 §2 inserts 'regex'): higher wins on
# span overlap within a chunk. Order: regex > spaCy > LLM (*amended v11 — the gazetteer tier that
# used to sit above regex, clarify 2026-08-06, is withdrawn with that capability*).
LAYER_PRECEDENCE: Dict[str, int] = {"regex": 3, "spacy": 2, "llm": 1}

# KG-AC-48: bare pronouns/anaphora that must never survive as an entity when coreference resolution
# is enabled -- a deterministic safety net independent of how well the coref rewrite performed.
_PRONOUN_STOPLIST = {
    "it", "its", "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs",
    "this", "that", "these", "those", "the company", "the group", "the fund", "the adviser",
    "the entity", "the firm",
}


@dataclass
class Candidate:
    """One entity mention proposed by a strategy layer."""
    surface_form: str
    entity_type: str
    source_chunk_id: str
    layer: str  # 'regex' | 'spacy' | 'llm' (*'gazetteer' withdrawn at v11*)
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    confidence: float = 1.0
    occurrence_idx: int = 0
    is_abstract: bool = False  # KG-AC-72 (v13) — a pack-declared abstract type, never span-located;
    # surface_form holds the model's synthesised name. Only LlmGraphStrategy ever sets this True.


@dataclass
class Relation:
    """One relation proposed by the graph extractor (evolve v5 — the one-pass LLM call, KG-AC-43).
    ``extractor`` (evolve v8, KG-AC-56) names the producing layer ('llm' | 'rules'); defaults to
    'llm' since that was the sole relation source before v8 — existing hand-built Relations in
    tests/other callers need no change."""
    relation_type: str
    src_surface: str
    src_type: str
    dst_surface: str
    dst_type: str
    source_chunk_id: str
    confidence: float = 1.0
    evidence_text: Optional[str] = None  # KG-AC-46 — mandatory for LLM-sourced relations (dropped
                                          # at parse time if absent; None here only for hand-built
                                          # Relations in tests/other callers, never written that way)
    extractor: str = "llm"


@dataclass
class Fact:
    """One attribute value proposed by the graph extractor for an already-known entity (evolve v13,
    KG-AC-70). Only ``LlmGraphStrategy`` produces these — classify/entity_scoped are relation-only
    modes over entities that already came from this same call (structural finding 2, cb-plan).
    ``value`` is the model's verbatim string (consistent with the evidence-grounding guarantee — a
    stored value that doesn't appear in the source would contradict it); ``normalized_value`` is
    derived deterministically from the property's declared ``range`` kind, ``None`` when the
    normaliser cannot parse it (never a guess). ``subject_type``/``subject_surface`` bind to the
    entity that carries this fact — same resolve-by-id-then-stamp-from-Candidate shape KG-AC-71
    already uses for relations, so the binding can't mismatch a subject's later filtered-out row
    (a fact whose subject is itself dropped, e.g. unmapped or unlocatable, is dropped alongside it)."""
    property: str
    value: str
    normalized_value: Optional[str]
    evidence_text: str
    subject_type: str
    subject_surface: str
    source_chunk_id: str
    extractor: str = "llm"


# KG-AC-64 (evolve v12) grounding helpers — relocated here from strategies/base.py at v13 (KG-AC-70)
# so both relation AND fact validation share ONE implementation (reuse, not duplicate).
_WS_RE = re.compile(r"\s+")


def normalize_ws(s: Optional[str]) -> str:
    return _WS_RE.sub(" ", s or "").strip()


def evidence_grounded(evidence: Optional[str], chunk_text: str) -> bool:
    """The evidence must occur VERBATIM (whitespace-normalised comparison; case-sensitive
    otherwise) in the source chunk text. Missing evidence never grounds."""
    if not evidence:
        return False
    return normalize_ws(evidence) in normalize_ws(chunk_text)


# KG-AC-70 (v13): per-range-kind fact normalisation. Deliberately hand-rolled against a small,
# explicit format set rather than a general-purpose date library (no new dependency decision folded
# into this task) — an unparseable value is None, never a guess, and the caller counts it. Best-
# effort by design: the AC itself accepts "unparseable -> null, counted," not universal coverage.
_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%d %b %Y", "%b %d, %Y")
_CURRENCY_WORDS_RE = re.compile(r"\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|dollars?|pounds?|euros?)\b", re.IGNORECASE)
_NUMBER_STRIP_RE = re.compile(r"[,$€£¥\s]")


def _normalize_date(value: str) -> Optional[str]:
    v = normalize_ws(value)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _normalize_number(value: str) -> Optional[str]:
    v = _CURRENCY_WORDS_RE.sub("", value)
    v = _NUMBER_STRIP_RE.sub("", v)
    if not v:
        return None
    try:
        d = Decimal(v)
    except InvalidOperation:
        return None
    return format(d, "f")


_RANGE_NORMALIZERS = {
    "date": _normalize_date,
    "number": _normalize_number,
    # identifier/string: whitespace-trimmed only (KG-AC-70's clarify answer) -- always succeeds on
    # a non-empty value, so both kinds share the plain normalize_ws helper directly (no entry
    # needed here; see normalize_fact_value's fallback branch).
}


def normalize_fact_value(value: str, range_kind: str) -> Optional[str]:
    """KG-AC-70: deterministic per-``range``-kind normalisation. ``date``/``number`` may fail to
    parse (returns None, caller counts it); ``identifier``/``string`` (and any other declared kind,
    defensively) fall back to whitespace-trimming, which always succeeds on a non-empty value."""
    normalizer = _RANGE_NORMALIZERS.get(range_kind)
    if normalizer is not None:
        return normalizer(value)
    trimmed = normalize_ws(value)
    return trimmed or None


def _precedence(layer: str) -> int:
    return LAYER_PRECEDENCE.get(layer, 0)


def spans_overlap(a: Candidate, b: Candidate) -> bool:
    """Two candidates conflict iff both carry spans and their [start,end) intervals intersect.
    Spanless candidates never overlap (they are disambiguated by occurrence_idx instead)."""
    if None in (a.span_start, a.span_end, b.span_start, b.span_end):
        return False
    return a.span_start < b.span_end and b.span_start < a.span_end


def merge_candidates(candidates: List[Candidate]) -> List[Candidate]:
    """KG-AC-12: within a ``source_chunk_id`` two candidates conflict iff their spans overlap;
    the highest-precedence layer's candidate (regex > spaCy > LLM) is kept WHOLE and the
    overlapping others dropped; non-overlapping candidates from every layer are unioned.
    Deterministic for fixed input (stable original order preserved on output)."""
    by_chunk: Dict[str, List[Tuple[int, Candidate]]] = {}
    for i, c in enumerate(candidates):
        by_chunk.setdefault(c.source_chunk_id, []).append((i, c))

    kept: List[Tuple[int, Candidate]] = []
    for _chunk_id, items in by_chunk.items():
        # Consider highest precedence first; tie-break by span_start (None last) then original index
        # so the resolution is fully deterministic.
        order = sorted(
            items,
            key=lambda t: (
                -_precedence(t[1].layer),
                t[1].span_start if t[1].span_start is not None else (1 << 30),
                t[0],
            ),
        )
        chosen: List[Tuple[int, Candidate]] = []
        for idx, cand in order:
            if any(spans_overlap(cand, kc) for _, kc in chosen):
                continue  # a higher/equal-precedence candidate already occupies this span
            chosen.append((idx, cand))
        kept.extend(chosen)

    kept.sort(key=lambda t: t[0])  # restore stable input order
    return [c for _, c in kept]


def filter_bare_pronouns(candidates: List[Candidate]) -> List[Candidate]:
    """KG-AC-48: drop any candidate whose surface (normalized) is a bare pronoun/anaphor. Applied
    only when ``coreference_enabled`` — a deterministic guarantee that survives even if the coref
    rewrite step imperfectly missed a reference. A relation whose endpoint is dropped here becomes
    dangling and is dropped downstream by build_edge_records — no separate relation-side filter
    needed."""
    return [c for c in candidates if c.surface_form.strip().lower() not in _PRONOUN_STOPLIST]


def assign_occurrence_indices(candidates: List[Candidate]) -> List[Candidate]:
    """For candidates lacking spans, stamp occurrence_idx = k-th mention of
    (chunk, type, surface) in appearance order (design §Idempotency & identity). Candidates WITH
    spans keep occurrence_idx=0 (the span disambiguates). Mutates in place, returns the list."""
    seen: Dict[Tuple[str, str, str], int] = {}
    for c in candidates:
        if c.span_start is None:
            key = (c.source_chunk_id, c.entity_type, c.surface_form)
            c.occurrence_idx = seen.get(key, 0)
            seen[key] = c.occurrence_idx + 1
        else:
            c.occurrence_idx = 0
    return candidates


def compute_entity_uid(
    folder_id: str, source_chunk_id: str, entity_type: str, surface_form: str,
    span_start: Optional[int], occurrence_idx: int,
) -> str:
    """Deterministic per-mention identity (KG-AC-10). run_id is deliberately EXCLUDED so re-runs
    converge. ``entity_uid = sha256(folder_id|source_chunk_id|entity_type|surface_form|
    COALESCE(span_start,'')|occurrence_idx)``."""
    parts = [
        folder_id, source_chunk_id, entity_type, surface_form,
        "" if span_start is None else str(span_start), str(occurrence_idx),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def compute_derived_entity_uid(folder_id: str, entity_type: str, identity_value: str) -> str:
    """KG-AC-90 (v15): identity for a DERIVED instance is **structural**, not textual —
    ``sha256(folder_id|entity_type|identity_value)``. Deliberately excludes chunk, span and
    occurrence (all of which a derived instance has none of, being document-scoped) and above all
    excludes any composed NAME: `entity_canonicalization.canonical_key` is
    ``<entity_type>|<normalized_form>``, so a model-composed name would make the same real-world
    relationship resolve to different keys across documents and split one canonical hub into two."""
    parts = [folder_id, entity_type, identity_value]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def compute_edge_uid(folder_id: str, relation_type: str, src_entity_uid: str, dst_entity_uid: str) -> str:
    """Deterministic edge identity (KG-AC-10), stable across serial-id churn."""
    parts = [folder_id, relation_type, src_entity_uid, dst_entity_uid]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def build_entity_records(
    folder_id: str, merged: List[Candidate], ontology_pack: str, ontology_version: str,
    model_id: Optional[str] = None,
    chunk_provenance: Optional[Dict[str, Tuple[Optional[str], Optional[int]]]] = None,
) -> List[Dict]:
    """Turn merged candidates into staged kg_entities row dicts (extraction-owned fields).
    normalized_form / canonical_id are left for canonicalization. run_id/dag_id are added by the
    worker at store time. KG-AC-73 (v13): `chunk_provenance` (chunk_id -> (doc_id, page)) stamps
    each row's `source_doc_id`/`page`; a chunk absent from the map (or the map itself omitted)
    records both as null — never a fabricated value."""
    chunk_provenance = chunk_provenance or {}
    rows: List[Dict] = []
    for c in merged:
        uid = compute_entity_uid(
            folder_id, c.source_chunk_id, c.entity_type, c.surface_form, c.span_start, c.occurrence_idx
        )
        doc_id, page = chunk_provenance.get(c.source_chunk_id, (None, None))
        rows.append({
            "entity_uid": uid,
            "entity_type": c.entity_type,
            "surface_form": c.surface_form,
            "source_chunk_id": c.source_chunk_id,
            "source_doc_id": doc_id,
            "page": page,
            "is_abstract": c.is_abstract,
            "span_start": c.span_start,
            "span_end": c.span_end,
            "occurrence_idx": c.occurrence_idx,
            "confidence": c.confidence,
            "extractor": c.layer,
            "ontology_pack": ontology_pack,
            "ontology_version": ontology_version,
            "model_id": model_id if c.layer == "llm" else None,
            "stage": "staged",
        })
    return rows


def attach_facts_to_entity_records(
    entity_rows: List[Dict], facts: List[Fact],
    chunk_provenance: Optional[Dict[str, Tuple[Optional[str], Optional[int]]]] = None,
) -> List[Dict]:
    """KG-AC-70 (v13): nest surviving facts onto their subject's kg_entities row — `attributes`
    (the existing jsonb column) becomes a list of `{property, value, normalized_value, evidence,
    source_doc_id, page}` objects, one row per entity, facts nested (no new table). Matched by the
    same `(chunk, type, surface)` key `entity_uid_key_map` uses. A fact whose subject was itself
    dropped (unmapped type, unlocatable span, ...) has no matching row and is silently excluded —
    its subject never got written, so the fact has nowhere to attach; this is not a separate drop
    reason and is not double-counted. Every row gets an `attributes` list, `[]` when it has no
    facts (never omitted, never null) — mirrors `kg_canonical_entities.attributes`'s own
    `DEFAULT '[]'::jsonb`. Mutates in place, returns the list (assign_occurrence_indices' pattern)."""
    chunk_provenance = chunk_provenance or {}
    by_subject_key: Dict[Tuple[str, str, str], List[Fact]] = {}
    for f in facts:
        key = (f.source_chunk_id, f.subject_type, f.subject_surface)
        by_subject_key.setdefault(key, []).append(f)
    for row in entity_rows:
        key = (row.get("source_chunk_id"), row.get("entity_type"), row.get("surface_form"))
        attrs = []
        for f in by_subject_key.get(key, []):
            doc_id, page = chunk_provenance.get(f.source_chunk_id, (None, None))
            attrs.append({
                "property": f.property, "value": f.value, "normalized_value": f.normalized_value,
                "evidence": f.evidence_text, "source_doc_id": doc_id, "page": page,
            })
        row["attributes"] = attrs
    return entity_rows


def mark_reference_only(entity_rows: List[Dict], pack) -> List[Dict]:
    """KG-AC-94 (v15): flag entities the document NAMED (with an identifier) but never described —
    ``Fee Schedule FS-2025-031`` cited in a definitions clause whose actual terms live in another
    document. They are written, not dropped: this ontology is multi-document, so such a node is the
    join point canonicalization merges the real document into when it arrives, and dropping it also
    discards a source-stated relation. The MARKER is the load-bearing half — an unmarked stub is
    worse than none, because nothing then distinguishes "a Fee Schedule we know exists" from "one we
    have read" (KG-AC-92's honesty posture, applied to a different distinction).

    Deterministic, from the pack's OWN ``range`` vocabulary (KG-AC-69) — no new declaration and no
    heuristic: all facts ``identifier``-ranged (or none at all) => reference-only; a single
    ``string``/``number``/``date`` fact means the document actually described it.

    DERIVED entities are excluded entirely (clarify F2): a ``reified_relation`` hub declares zero
    own attributes BY DEFINITION, so the rule would mark every one of them an unread stub. They are
    already marked by ``extractor="derived"``/``is_abstract`` — a different fact about a different
    thing. Must run AFTER attribute re-parenting (KG-AC-93), or an anchor whose bundle attributes
    have just moved off it could be mis-flagged. Mutates in place, returns the list."""
    identifier_props = {
        p.property for p in pack.datatype_properties.values() if p.range == "identifier"
    }
    for row in entity_rows:
        if row.get("extractor") == "derived":
            row["reference_only"] = False
            continue
        props = [a.get("property") for a in (row.get("attributes") or [])]
        row["reference_only"] = all(p in identifier_props for p in props)
    return entity_rows


def _representative_constituents(entity_rows: List[Dict], wanted_types: Sequence[str]) -> List[Dict]:
    """KG-AC-91(a): collapse mentions to ONE representative per ``(entity_type, normalized_form)``,
    chosen deterministically as the earliest ``(source_doc_id, source_chunk_id, span_start)``.
    Stage 1 emits one row per occurrence (KG-AC-63) and ``merge_candidates`` only collapses
    span-overlaps within a chunk, so without this a hub would fan out to every mention — five
    mentions, five duplicate edges, a cardinality nothing else in the system produces."""
    wanted = set(wanted_types)
    best: Dict[Tuple[str, str], Tuple[Tuple, Dict]] = {}
    for row in entity_rows:
        if row.get("entity_type") not in wanted:
            continue
        key = (row["entity_type"], normalize_ws(row.get("surface_form")))
        # None sorts last within each component — a located mention always beats a span-less one.
        order = (
            (row.get("source_doc_id") is None, row.get("source_doc_id") or ""),
            (row.get("source_chunk_id") is None, row.get("source_chunk_id") or ""),
            (row.get("span_start") is None, row.get("span_start") or 0),
        )
        if key not in best or order < best[key][0]:
            best[key] = (order, row)
    return [row for _order, row in best.values()]


def derive_abstract_entities(
    folder_id: str, entity_rows: List[Dict], pack, ontology_pack: str, ontology_version: str,
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """KG-AC-90/91 (v15): mint pack-declared abstract types DETERMINISTICALLY after extraction and
    attach their definitional edges. Runs AFTER ``attach_facts_to_entity_records`` (the identity
    value IS a fact) and BEFORE ``build_edge_records``.

    Mint trigger is **pattern-specific** — v14's presence-of-anchor test minted contentless hubs
    (an agreement merely REFERENCING a fee schedule minted `CommercialTerms` with zero attributes,
    disproved against a real golden):
      * ``reified_relation``  — identity resolves AND ≥1 auto_relation participant is present.
      * ``attribute_bundle``  — identity resolves AND ≥1 of its OWN datatype_properties was
        extracted (those arrive via KG-AC-93's anchoring, R2; before that they never mint, which
        is the correct outcome for a document that only references them).

    Returns ``(derived_rows, derived_edges, counters)``. Edges are returned as row dicts already
    keyed by ``entity_uid`` — a derived instance is document-scoped with ``source_chunk_id=None``,
    so it cannot be resolved through ``entity_uid_key_map``'s ``(chunk, type, surface)`` lookup."""
    derived_rows: List[Dict] = []
    derived_edges: List[Dict] = []
    counters = {"underivable_entity_count": 0, "ambiguous_attachment_count": 0,
                "unanchored_fact_count": 0}

    for etype, et in pack.entity_types.items():
        spec = et.derived
        if spec is None:
            continue
        anchor_type, _, id_prop = spec.identity_from.partition(".")
        own_props = {p.property for p in pack.datatype_properties.values() if p.domain == etype}

        # identity: every distinct value of the anchor's identity property present in this folder
        identity_values: List[str] = []
        for row in entity_rows:
            if row.get("entity_type") != anchor_type:
                continue
            for attr in row.get("attributes") or []:
                if attr.get("property") == id_prop and attr.get("value"):
                    value = attr["value"]
                    if value not in identity_values:
                        identity_values.append(value)

        # content test (the v15 mint trigger)
        if spec.pattern == "attribute_bundle":
            has_content = any(
                (attr.get("property") in own_props)
                for row in entity_rows for attr in (row.get("attributes") or [])
            )
        else:  # reified_relation
            # A participant only counts if it is NEITHER the hub itself NOR the identity anchor.
            # The anchor's presence is already required for identity to resolve, so counting it
            # again would make this test vacuous — `governedBy` puts `Agreement` in
            # InvestmentRelationship's participant set, and an agreement alone would then mint a
            # relationship with no parties, which is exactly v14's contentless-hub defect wearing
            # a different hat. Found by this task's own test, not by review.
            participant_types = set()
            for rel_name in spec.auto_relations:
                rel = pack.relations.get(rel_name)
                if rel is None:
                    continue
                participant_types |= {
                    t for t in list(rel.domain) + list(rel.range)
                    if t != etype and t != anchor_type
                }
            has_content = any(r.get("entity_type") in participant_types for r in entity_rows)

        if not has_content:
            continue  # nothing to represent — never mint a contentless hub (the v14 defect)
        if not identity_values:
            counters["underivable_entity_count"] += 1  # content but no identity: skip, never guess
            continue

        # KG-AC-93 (v15): facts the model attached to the ANCHOR that actually belong to this
        # abstract type. Collected across every anchor-type row (the abstract type is
        # document-scoped, so which mention the fact happened to land on is immaterial), then
        # re-parented below. With MORE than one identity value the fact→hub pairing is not
        # determinable — same reasoning as KG-AC-91(b)'s edge rule — so they are dropped and
        # counted rather than guessed at or left on the anchor, which does not declare them.
        anchored: List[Dict] = []
        for row in entity_rows:
            if row.get("entity_type") != anchor_type:
                continue
            keep = []
            for attr in row.get("attributes") or []:
                (anchored if attr.get("property") in own_props else keep).append(attr)
            if len(keep) != len(row.get("attributes") or []):
                row["attributes"] = keep  # strip from the anchor: it does not declare these
        if anchored and len(identity_values) != 1:
            counters["unanchored_fact_count"] += len(anchored)
            anchored = []

        for value in identity_values:
            derived_rows.append({
                "entity_uid": compute_derived_entity_uid(folder_id, etype, value),
                "entity_type": etype,
                "surface_form": value,           # the identity value itself, NEVER a composed name
                "source_chunk_id": None,          # document-scoped, not chunk-scoped
                "source_doc_id": None, "page": None,
                "is_abstract": True,
                "span_start": None, "span_end": None,
                "occurrence_idx": 0,
                "confidence": 1.0,                # deterministic entailment, not a model guess
                "extractor": "derived",           # KG-AC-92: distinguishable from source-grounded
                "ontology_pack": ontology_pack,
                "ontology_version": ontology_version,
                "model_id": None,
                "stage": "staged",
                # KG-AC-93: re-parented facts land here. Each keeps its OWN source_doc_id/page,
                # so the hub carries no evidence itself (KG-AC-92) while every attribute on it
                # stays individually evidence-grounded and traceable to real text.
                "attributes": anchored if len(identity_values) == 1 else [],
            })

        # KG-AC-91(b): >1 hub of this type ⇒ the hub↔constituent pairing is not determinable from
        # the ontology alone, so attach NOTHING and count — the hubs still exist, a pairing is
        # never invented.
        if len(identity_values) > 1:
            counters["ambiguous_attachment_count"] += len(spec.auto_relations)
            continue

        hub_uid = derived_rows[-1]["entity_uid"]
        for rel_name in spec.auto_relations:
            rel = pack.relations.get(rel_name)
            if rel is None:
                continue
            hub_is_domain = etype in rel.domain
            other_types = list(rel.range) if hub_is_domain else list(rel.domain)
            for other in _representative_constituents(entity_rows, other_types):
                src_uid, dst_uid = ((hub_uid, other["entity_uid"]) if hub_is_domain
                                    else (other["entity_uid"], hub_uid))
                derived_edges.append({
                    "edge_uid": compute_edge_uid(folder_id, rel_name, src_uid, dst_uid),
                    "relation_type": rel_name,
                    "src_entity_uid": src_uid, "dst_entity_uid": dst_uid,
                    # KG-AC-92: no sentence asserts a derived edge DIRECTLY, so no quote is
                    # fabricated; the constituent's own provenance keeps it traceable.
                    "evidence_text": None,
                    "source_doc_id": other.get("source_doc_id"), "page": other.get("page"),
                    "confidence": 1.0,
                    "extractor": "derived",
                })
    return derived_rows, derived_edges, counters


def build_edge_records(
    folder_id: str, relations: List[Relation], entity_uid_by_key: Dict[Tuple[str, str, str], str],
    chunk_provenance: Optional[Dict[str, Tuple[Optional[str], Optional[int]]]] = None,
) -> List[Dict]:
    """Turn relations into kg_edges row dicts. An edge whose endpoints are not both in the merged
    entity set (keyed by (chunk, type, surface)) is dropped — an edge can only connect written
    entities. Deterministic. KG-AC-73 (v13): `chunk_provenance` stamps each edge's `source_doc_id`/
    `page` from its own evidence chunk, same lookup + null-on-absence rule as build_entity_records."""
    chunk_provenance = chunk_provenance or {}
    rows: List[Dict] = []
    for r in relations:
        src_key = (r.source_chunk_id, r.src_type, r.src_surface)
        dst_key = (r.source_chunk_id, r.dst_type, r.dst_surface)
        src_uid = entity_uid_by_key.get(src_key)
        dst_uid = entity_uid_by_key.get(dst_key)
        if not src_uid or not dst_uid:
            continue
        doc_id, page = chunk_provenance.get(r.source_chunk_id, (None, None))
        rows.append({
            "edge_uid": compute_edge_uid(folder_id, r.relation_type, src_uid, dst_uid),
            "relation_type": r.relation_type,
            "src_entity_uid": src_uid,
            "dst_entity_uid": dst_uid,
            "source_doc_id": doc_id,
            "page": page,
            "confidence": r.confidence,
            "evidence_text": r.evidence_text,
            "extractor": r.extractor,
        })
    return rows


# Canonical label order for a merged multi-source edge (KG-AC-56: 'rules+llm', that literal order —
# not alphabetical). Any extractor name absent from this list is appended after, sorted, so an
# unexpected future source name still produces a deterministic (not crashing) label.
_EXTRACTOR_LABEL_ORDER = ["rules", "llm"]


def merge_edge_records(edge_rows: List[Dict]) -> List[Dict]:
    """KG-AC-56 (evolve v8): union+dedup relations from multiple sources on
    (src_entity_uid, relation_type, dst_entity_uid) — order-preserving (a plain dict keyed by the
    triple, never set-iteration), so output order is deterministic for fixed input. A triple with a
    single contributor passes through unchanged. A triple asserted by more than one extractor
    collapses to ONE row: confidence = max of the contributors; evidence_text = the distinct
    contributing evidence strings joined by '; ' in first-encounter order (nothing dropped — the
    richer per-source-pair top-N evidence selection is canonicalization's job, KG-AC-47); extractor =
    the contributing layer names joined per _EXTRACTOR_LABEL_ORDER (e.g. 'rules+llm'); edge_uid is
    shared by construction (it depends only on folder_id/relation_type/src/dst, identical for every
    contributor in the group)."""
    groups: Dict[Tuple[str, str, str], List[Dict]] = {}
    for row in edge_rows:
        key = (row["src_entity_uid"], row["relation_type"], row["dst_entity_uid"])
        groups.setdefault(key, []).append(row)

    out: List[Dict] = []
    for rows in groups.values():
        if len(rows) == 1:
            out.append(rows[0])
            continue
        confidence = max(r["confidence"] for r in rows)
        evidence_seen: List[str] = []
        for r in rows:
            ev = r.get("evidence_text")
            if ev and ev not in evidence_seen:
                evidence_seen.append(ev)
        present = {r.get("extractor") for r in rows}
        ordered = [name for name in _EXTRACTOR_LABEL_ORDER if name in present]
        ordered += sorted(present - set(_EXTRACTOR_LABEL_ORDER))
        merged = dict(rows[0])
        merged["confidence"] = confidence
        merged["evidence_text"] = "; ".join(evidence_seen) if evidence_seen else rows[0].get("evidence_text")
        merged["extractor"] = "+".join(ordered)
        out.append(merged)
    return out


def vote_relations(runs: List[List[Relation]], k: int) -> List[Relation]:
    """KG-AC-67 (evolve v12 — self-consistency voting): given ``k`` independent relation-extraction
    runs over the SAME input, keep only relations that appeared in **>= ceil(k/2)** of the runs;
    each surviving relation's ``confidence`` is overwritten with its vote fraction (``votes / k``).
    Identity for voting is ``(relation_type, src_surface, src_type, dst_surface, dst_type,
    source_chunk_id)`` — NOT ``evidence_text`` (different runs may quote different supporting
    sentences for the same claim). A run asserting the same triple twice counts as ONE vote, never
    two. The kept row's other fields (``evidence_text``, ``extractor``, ...) come from its FIRST
    occurrence across runs, in run order. Deterministic given fixed run order. ``k`` is accepted
    explicitly rather than derived from ``len(runs)`` so the threshold is correct even when a run
    legitimately returns zero relations."""
    threshold = -(-k // 2)  # ceil(k/2), no math import needed
    votes: Dict[Tuple[str, str, str, str, str, str], int] = {}
    first_seen: Dict[Tuple[str, str, str, str, str, str], Relation] = {}
    for run in runs:
        seen_this_run = set()
        for r in run:
            key = (r.relation_type, r.src_surface, r.src_type, r.dst_surface, r.dst_type, r.source_chunk_id)
            if key in seen_this_run:
                continue  # a run asserting the same triple twice is ONE vote, not two
            seen_this_run.add(key)
            votes[key] = votes.get(key, 0) + 1
            first_seen.setdefault(key, r)
    kept: List[Relation] = []
    for key, count in votes.items():
        if count >= threshold:
            kept.append(replace(first_seen[key], confidence=count / k))
    return kept


def promote_top_entities(entity_rows: List[Dict], promote_top_n: int = 10, hard_max: int = 20) -> List[Dict]:
    """KG-AC-37: deterministic doc-level top_entities — ranked by mention count (descending),
    ties broken by first-appearance order — never exceeding ``promote_top_n`` (clamped to
    [0, hard_max]). Bulk entities never ride the state plane; only this bounded summary does."""
    n = max(0, min(promote_top_n, hard_max))
    counts: Dict[Tuple[str, str], int] = {}
    first_seen: Dict[Tuple[str, str], int] = {}
    for i, row in enumerate(entity_rows):
        key = (row["entity_type"], row["surface_form"])
        counts[key] = counts.get(key, 0) + 1
        if key not in first_seen:
            first_seen[key] = i
    ranked = sorted(counts.keys(), key=lambda k: (-counts[k], first_seen[k]))
    return [
        {"entity_type": k[0], "surface_form": k[1], "mention_count": counts[k]}
        for k in ranked[:n]
    ]


def build_summary(
    entity_rows: List[Dict], edge_rows: List[Dict], ontology_pack: str, ontology_version: str,
    unmapped_type_count: int, promote_top_n: int = 10, ungrounded_relation_count: int = 0,
    self_consistency_votes: int = 1, chunk_metadata_missing_count: int = 0,
    unresolved_reference_count: int = 0, unlocatable_entity_count: int = 0,
    unmapped_property_count: int = 0, ungrounded_fact_count: int = 0,
    guardrails_blocked_facts: int = 0, underivable_entity_count: int = 0,
    ambiguous_attachment_count: int = 0, unanchored_fact_count: int = 0,
) -> Dict:
    """The KG-AC-9 state-plane scalar summary the callback promotes (no bulk rows on state).
    *Amended v11 — `linked_count` (gazetteer-link count) is dropped with that capability.*
    *Amended v12 — `ungrounded_relation_count` (KG-AC-64) added: LLM relations dropped because
    their evidence was not found verbatim in the source chunk. `self_consistency_votes` (KG-AC-67)
    added: the number of independent LLM relation-extraction runs made for the batch (1 when
    self-consistency voting is off, the default).*
    *Amended v13 — `chunk_metadata_missing_count` (KG-AC-73) added: chunks whose `chunk_metadata`
    lacked a doc_id or page, so document/page provenance recorded null rather than a fabricated
    value. `unresolved_reference_count` (KG-AC-71) added: relations whose src_id/dst_id did not
    resolve to an entity in the same LLM call's own response, across every LLM relation call the
    run made (run 1 and every self-consistency repeat). `unlocatable_entity_count` (KG-AC-72)
    added: non-abstract entities the model returned whose surface could not be located in the
    source chunk, dropped rather than written with a null span — this is a NEW v13 behaviour, not
    a preservation of an existing one (see this task's Done writeup for the correction).
    `unmapped_property_count`/`ungrounded_fact_count` (KG-AC-70) added: facts dropped either for an
    unknown property / invalid domain (the one vocabulary-mapping counter — KG-AC-70's row counts
    both gates together, unlike relations' uncounted illegal-domain posture) or for ungrounded
    evidence, respectively. `guardrails_blocked_facts` (KG-AC-84) added: facts dropped by the SAME
    once-per-batch guardrails screen entity candidates already pass through — mirrors the entity
    posture, but unlike the pre-existing entity `guardrails_blocked` count (a bare `run_pipeline`
    return value, never on the state plane), this one IS wired to the state plane from this task.*
    *Amended v15 — the derivation pass's three discard paths join KG-AC-74's enumerated family
    (KG-AC-92/93): `underivable_entity_count` (an abstract type had content but its identity value
    was absent — skipped, never minted under a fabricated key), `ambiguous_attachment_count`
    (more than one hub of a type, so a constituent pairing would have to be invented — withheld),
    and `unanchored_fact_count` (an anchored fact whose hub pairing is likewise undeterminable —
    dropped rather than left on an anchor that does not declare the property). Each names exactly
    one path, per the P8 audit rule.*"""
    distinct_types = len({r["entity_type"] for r in entity_rows})
    return {
        "entity_count": len(entity_rows),
        "edge_count": len(edge_rows),
        "distinct_types": distinct_types,
        "top_entities": promote_top_entities(entity_rows, promote_top_n),
        "ontology_pack": ontology_pack,
        "ontology_version": ontology_version,
        "unmapped_type_count": unmapped_type_count,
        "ungrounded_relation_count": ungrounded_relation_count,
        "self_consistency_votes": self_consistency_votes,
        "chunk_metadata_missing_count": chunk_metadata_missing_count,
        "unresolved_reference_count": unresolved_reference_count,
        "unlocatable_entity_count": unlocatable_entity_count,
        "unmapped_property_count": unmapped_property_count,
        "ungrounded_fact_count": ungrounded_fact_count,
        "guardrails_blocked_facts": guardrails_blocked_facts,
        "underivable_entity_count": underivable_entity_count,
        "ambiguous_attachment_count": ambiguous_attachment_count,
        "unanchored_fact_count": unanchored_fact_count,
    }


def entity_uid_key_map(entity_rows: List[Dict]) -> Dict[Tuple[str, str, str], str]:
    """(source_chunk_id, entity_type, surface_form) -> entity_uid, for wiring edges to endpoints.
    On duplicate keys (spanless repeats) the FIRST wins — deterministic."""
    out: Dict[Tuple[str, str, str], str] = {}
    for r in entity_rows:
        key = (r["source_chunk_id"], r["entity_type"], r["surface_form"])
        out.setdefault(key, r["entity_uid"])
    return out
