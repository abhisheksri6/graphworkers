"""Pure extraction core — NO Celery, DB, network, or spaCy imports.

Deterministic given fixed strategy output: the layer-precedence merge (KG-AC-12), the bounded
top-N promotion (KG-AC-37), the deterministic uids (KG-AC-10), and the state-scalar summary
(KG-AC-9) all live here so they are unit-testable without a broker, model, or DB. Strategy
execution (regex / spaCy / LLM) lives in ``strategies/`` and is injected — this module only
transforms the candidates strategies produce.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

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
) -> Dict:
    """The KG-AC-9 state-plane scalar summary the callback promotes (no bulk rows on state).
    *Amended v11 — `linked_count` (gazetteer-link count) is dropped with that capability.*
    *Amended v12 — `ungrounded_relation_count` (KG-AC-64) added: LLM relations dropped because
    their evidence was not found verbatim in the source chunk. `self_consistency_votes` (KG-AC-67)
    added: the number of independent LLM relation-extraction runs made for the batch (1 when
    self-consistency voting is off, the default).*
    *Amended v13 — `chunk_metadata_missing_count` (KG-AC-73) added: chunks whose `chunk_metadata`
    lacked a doc_id or page, so document/page provenance recorded null rather than a fabricated
    value.*"""
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
    }


def entity_uid_key_map(entity_rows: List[Dict]) -> Dict[Tuple[str, str, str], str]:
    """(source_chunk_id, entity_type, surface_form) -> entity_uid, for wiring edges to endpoints.
    On duplicate keys (spanless repeats) the FIRST wins — deterministic."""
    out: Dict[Tuple[str, str, str], str] = {}
    for r in entity_rows:
        key = (r["source_chunk_id"], r["entity_type"], r["surface_form"])
        out.setdefault(key, r["entity_uid"])
    return out
