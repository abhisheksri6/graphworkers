"""Candidate-pair enumeration for relation-classification mode (KG-AC-60, evolve v10, ADR-0014).

Pure — no spaCy/DB/network (same purity rule as ``core.py``). Sentence boundaries are handed in as
plain ``(start, end)`` character-offset tuples per chunk; this module never derives them itself —
that is L3's job (``strategies/base.py`` wires a spaCy parse into these spans before calling here),
so the enumeration stays testable with hand-built input and reusable regardless of which entity
layers produced the typed entities (KG-AC-59 — classify is decoupled from ``engine``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core import Candidate


@dataclass
class CandidatePair:
    """One directed, type-compatible, same-sentence entity pair — offered to the classifier
    (KG-AC-61) among ``allowed_relation_types`` (+ the implicit ``no_relation`` option)."""
    chunk_id: str
    src_surface: str
    src_type: str
    src_span: Tuple[Optional[int], Optional[int]]
    dst_surface: str
    dst_type: str
    dst_span: Tuple[Optional[int], Optional[int]]
    sentence_span: Tuple[int, int]
    allowed_relation_types: List[str]


def _sentence_index(pos: Optional[int], spans: List[Tuple[int, int]]) -> Optional[int]:
    if pos is None:
        return None
    for idx, (start, end) in enumerate(spans):
        if start <= pos < end:
            return idx
    return None


def enumerate_candidate_pairs(
    entities: List[Candidate], sentence_spans: Dict[str, List[Tuple[int, int]]], pack,
) -> List[CandidatePair]:
    """KG-AC-60: emit a ``CandidatePair`` for every ordered (src, dst) entity pair, WITHIN one
    chunk, that (a) co-occurs in the same sentence and (b) has >=1 pack relation whose domain/range
    is satisfied by (src_type, dst_type) — direction comes from the relation's own domain/range, so
    an unordered pair may yield zero, one, or two ``CandidatePair``s (one per legal direction). A
    chunk absent from ``sentence_spans`` yields no pairs (fail-closed, never a whole-chunk
    fallback). Order-preserving over ``entities`` input order — deterministic for fixed input."""
    by_chunk: Dict[str, List[Candidate]] = {}
    for e in entities:
        by_chunk.setdefault(e.source_chunk_id, []).append(e)

    pairs: List[CandidatePair] = []
    for chunk_id, ents in by_chunk.items():
        spans = sentence_spans.get(chunk_id)
        if not spans:
            continue
        sent_idx = [_sentence_index(e.span_start, spans) for e in ents]
        for i, src in enumerate(ents):
            if sent_idx[i] is None:
                continue
            for j, dst in enumerate(ents):
                if i == j or sent_idx[j] != sent_idx[i]:
                    continue
                allowed = sorted(
                    r.type for r in pack.relations.values()
                    if pack.relation_allowed(r.type, src.entity_type, dst.entity_type)
                )
                if not allowed:
                    continue
                pairs.append(CandidatePair(
                    chunk_id=chunk_id,
                    src_surface=src.surface_form, src_type=src.entity_type,
                    src_span=(src.span_start, src.span_end),
                    dst_surface=dst.surface_form, dst_type=dst.entity_type,
                    dst_span=(dst.span_start, dst.span_end),
                    sentence_span=spans[sent_idx[i]],
                    allowed_relation_types=allowed,
                ))
    return pairs
