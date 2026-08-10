"""Pure Cypher builders for the Neo4j projection — NO neo4j/DB imports.

Deterministic: one idempotent ``MERGE`` per canonical node (keyed on canonical_id) and per canonical
relationship (keyed on the (src, type, dst) pattern) so re-export creates no duplicates (KG-AC-29)
and a drop+re-export reproduces the graph exactly (KG-AC-28). Labels/relationship types can't be
Cypher parameters, so they are SANITIZED and string-formatted; ids + properties are bound params.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class CanonicalNode:
    canonical_id: str
    entity_type: str
    normalized_form: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalEdge:
    src_canonical_id: str
    relation_type: str
    dst_canonical_id: str
    support_count: Optional[int] = None         # KG-AC-47 — distinct source documents asserting it
    confidence: Optional[float] = None          # KG-AC-47 — max across contributing mention-edges
    evidence_text: List[str] = field(default_factory=list)  # KG-AC-47 — top-3 sentences by confidence


def sanitize_ident(value: str, *, fallback: str = "Unknown") -> str:
    """A safe Neo4j label / relationship type: keep [A-Za-z0-9_], never start with a digit."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", value or "")
    if not s:
        return fallback
    if s[0].isdigit():
        s = "_" + s
    return s


def build_node_statement(node: CanonicalNode) -> Tuple[str, Dict[str, Any]]:
    label = sanitize_ident(node.entity_type, fallback="Entity")
    cypher = (
        f"MERGE (n:`{label}` {{canonical_id: $canonical_id}}) "
        "SET n.entity_type = $entity_type, n.normalized_form = $normalized_form, "
        "n.provenance = $provenance"
    )
    params = {
        "canonical_id": node.canonical_id,
        "entity_type": node.entity_type,
        "normalized_form": node.normalized_form,
        "provenance": _json_provenance(node.provenance),
    }
    return cypher, params


def build_rel_statement(edge: CanonicalEdge) -> Tuple[str, Dict[str, Any]]:
    rtype = sanitize_ident(edge.relation_type, fallback="RELATED_TO")
    cypher = (
        "MATCH (a {canonical_id: $src}), (b {canonical_id: $dst}) "
        f"MERGE (a)-[r:`{rtype}`]->(b) "
        "SET r.support_count = $support_count, r.confidence = $confidence, "
        "r.evidence_text = $evidence_text"
    )
    return cypher, {
        "src": edge.src_canonical_id, "dst": edge.dst_canonical_id,
        "support_count": edge.support_count, "confidence": edge.confidence,
        "evidence_text": edge.evidence_text,
    }


def _json_provenance(prov: Dict[str, Any]) -> str:
    import json
    return json.dumps(prov or {}, sort_keys=True)


def build_export_statements(
    nodes: List[CanonicalNode], edges: List[CanonicalEdge],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Nodes first (so relationship MATCHes resolve), then relationships. All idempotent MERGE."""
    stmts: List[Tuple[str, Dict[str, Any]]] = [build_node_statement(n) for n in nodes]
    stmts += [build_rel_statement(e) for e in edges]
    return stmts


def run_export(
    nodes: List[CanonicalNode], edges: List[CanonicalEdge], execute: Callable[[str, Dict[str, Any]], Any],
) -> Dict[str, int]:
    """Execute every MERGE via the injected ``execute(cypher, params)`` (driver-agnostic — the worker
    wires it to a Neo4j session; tests inject a recorder). Returns {node_count, relationship_count}."""
    for cypher, params in (build_node_statement(n) for n in nodes):
        execute(cypher, params)
    for cypher, params in (build_rel_statement(e) for e in edges):
        execute(cypher, params)
    return {"node_count": len(nodes), "relationship_count": len(edges)}
