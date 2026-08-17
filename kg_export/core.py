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
    attributes: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # KG-AC-83 — P11/P13's
    # merged-facts shape: {property: [{value, normalized_value, status, provenance}]}
    entity_iri: Optional[str] = None  # KG-AC-86 — the pack's declared iri for this entity_type, if any
    # v16 (KG-AC-104) — the display + honesty properties. Built by v13–v15 so a consumer could tell
    # how a node came to be, but never exported: an Agreement, its derived hub and a reference stub
    # all rendered as near-identical nodes carrying the same identifier string.
    canonical_name: Optional[str] = None   # KG-AC-76
    aliases: List[str] = field(default_factory=list)  # KG-AC-77
    reference_only: bool = False           # KG-AC-94 — named with an identifier, never described
    is_abstract: bool = False              # KG-AC-72
    is_derived: bool = False               # KG-AC-92 — an entailment, not a source assertion


@dataclass
class CanonicalEdge:
    src_canonical_id: str
    relation_type: str
    dst_canonical_id: str
    support_count: Optional[int] = None         # KG-AC-47 — distinct source documents asserting it
    confidence: Optional[float] = None          # KG-AC-47 — max across contributing mention-edges
    evidence_text: List[str] = field(default_factory=list)  # KG-AC-47 — top-3 sentences by confidence
    relation_iri: Optional[str] = None  # KG-AC-86 — the pack's declared iri for this relation_type


def sanitize_ident(value: str, *, fallback: str = "Unknown") -> str:
    """A safe Neo4j label / relationship type: keep [A-Za-z0-9_], never start with a digit."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", value or "")
    if not s:
        return fallback
    if s[0].isdigit():
        s = "_" + s
    return s


def qualified_name(iri: Optional[str], bare_name: str) -> str:
    """KG-AC-86: the pack's ``iri`` verbatim — whatever form its author declared (a full IRI, or an
    already-prefixed form like ``iiro:Investor`` if a future pack writes it that way; kg_export
    applies NO transformation of its own, since no pack today declares a SEPARATE prefix-mapping
    construct — the `iri` string IS the qualified name). Absent ``iri`` → the bare name, never an
    error (most packs today declare no `iri` at all)."""
    return iri if iri else bare_name


def _facts_to_properties(attributes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """KG-AC-83: each merged fact (P11/P13's ``{property: [{value, normalized_value, status,
    provenance}]}`` shape) becomes a node property. One entry (`single_source`/`consistent`) → the
    scalar value (`normalized_value` when present, else `value`) + a `<property>_status` sibling.
    Two-or-more entries (`conflicting`) → the LIST of every competing value, never a chosen winner
    — `<property>_status` is `"conflicting"`. Returned as ONE plain dict, delivered to Cypher as a
    single bound parameter (`SET n += $facts`) — property NAMES need no interpolation this way,
    only the (already trusted, pack-vocabulary-closed) dict KEYS, and VALUES stay fully bound."""
    props: Dict[str, Any] = {}
    for prop, entries in attributes.items():
        if not entries:
            continue
        if len(entries) == 1:
            e = entries[0]
            props[prop] = e.get("normalized_value") or e.get("value")
            props[f"{prop}_status"] = e["status"]
        else:
            props[prop] = [e.get("normalized_value") or e.get("value") for e in entries]
            props[f"{prop}_status"] = "conflicting"
    return props


# v16 (KG-AC-104): the STABLE primary label every exported node carries. The MERGE key must not be
# the entity type — a canonical type legitimately sharpens across batches (KG-AC-103), and a
# re-labelled entity MERGEing on its new label would create a SECOND node for one canonical_id,
# i.e. the export manufacturing exactly the duplicates this spec exists to remove. The type is data
# on the node; a secondary type label is deliberately NOT set, because removing a stale one on
# every re-export is bookkeeping with no consumer today.
NODE_LABEL = "KgEntity"


def build_node_statement(node: CanonicalNode) -> Tuple[str, Dict[str, Any]]:
    cypher = (
        f"MERGE (n:`{NODE_LABEL}` {{canonical_id: $canonical_id}}) "
        "SET n.entity_type = $entity_type, n.normalized_form = $normalized_form, "
        "n.provenance = $provenance, n.ontology_class = $ontology_class, "
        "n.canonical_name = $canonical_name, n.aliases = $aliases, "
        "n.reference_only = $reference_only, n.is_abstract = $is_abstract, "
        "n.is_derived = $is_derived "
        "SET n += $facts"
    )
    params = {
        "canonical_id": node.canonical_id,
        "entity_type": node.entity_type,
        "normalized_form": node.normalized_form,
        "provenance": _json_provenance(node.provenance),
        "ontology_class": qualified_name(node.entity_iri, node.entity_type),  # KG-AC-86
        "canonical_name": node.canonical_name,       # KG-AC-76
        "aliases": list(node.aliases or []),          # KG-AC-77
        "reference_only": bool(node.reference_only),  # KG-AC-94
        "is_abstract": bool(node.is_abstract),        # KG-AC-72
        "is_derived": bool(node.is_derived),          # KG-AC-92
        "facts": _facts_to_properties(node.attributes),  # KG-AC-83
    }
    return cypher, params


def reconcile_statement(keep_canonical_ids: List[str]) -> Tuple[str, Dict[str, Any]]:
    """KG-AC-105 (export half): remove nodes this scope's plane of record no longer contains.

    Sound ONLY because one database holds exactly one scope (KG-AC-97/98) — there the set-diff of
    `canonical_id`s is authoritative. Bounded to `:KgEntity` so it can never touch anything this
    exporter did not write, and `DETACH` so a removed node's relationships go with it (a canonical
    edge whose endpoint is gone has nothing left to mean)."""
    cypher = (
        f"MATCH (n:`{NODE_LABEL}`) "
        "WHERE NOT n.canonical_id IN $keep "
        "DETACH DELETE n"
    )
    return cypher, {"keep": list(keep_canonical_ids)}


def build_rel_statement(edge: CanonicalEdge) -> Tuple[str, Dict[str, Any]]:
    rtype = sanitize_ident(edge.relation_type, fallback="RELATED_TO")
    cypher = (
        "MATCH (a {canonical_id: $src}), (b {canonical_id: $dst}) "
        f"MERGE (a)-[r:`{rtype}`]->(b) "
        "SET r.support_count = $support_count, r.confidence = $confidence, "
        "r.evidence_text = $evidence_text, r.ontology_relation = $ontology_relation"
    )
    return cypher, {
        "src": edge.src_canonical_id, "dst": edge.dst_canonical_id,
        "support_count": edge.support_count, "confidence": edge.confidence,
        "evidence_text": edge.evidence_text,
        "ontology_relation": qualified_name(edge.relation_iri, edge.relation_type),  # KG-AC-86
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
