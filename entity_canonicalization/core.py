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
    """One staged entity mention being canonicalized."""
    entity_uid: str
    entity_type: str
    surface_form: str
    normalized_form: str = ""


def normalize_surface(surface: str) -> str:
    """Lowercase, strip accents + punctuation, drop legal suffixes, collapse whitespace."""
    s = unicodedata.normalize("NFKD", surface or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t and t not in _LEGAL_SUFFIXES]
    return " ".join(tokens)


def canonical_key(entity_type: str, normalized_form: str) -> str:
    """The UNIQUE key on kg_canonical_entities: ``<reconciled_type>|<normalized_form>`` (*amended
    v11 — the LEI-authoritative-key short-circuit is removed with the gazetteer/external-id
    plane*). Uses the RECONCILED cluster type so chunk-boundary type inconsistency (F-CHUNK-2)
    doesn't split one real entity."""
    return f"{entity_type}|{normalized_form}"


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
