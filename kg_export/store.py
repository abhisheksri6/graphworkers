"""Read the canonical graph from Postgres (the plane of record) for projection into Neo4j.

Nodes = one per `canonical_id` (from kg_canonical_entities, joined to its canonicalized mentions for
provenance). Edges = one per (src_canonical, relation_type, dst_canonical), read directly from
`kg_canonical_edges` — the pre-aggregated index entity_canonicalization writes (support_count =
distinct source documents, confidence = max, evidence_text = top-3 sentences — KG-AC-47). This
worker never aggregates mention-edges itself; kg_canonical_edges is already collapsed to one row per
canonical relationship. ``folder_ids=None`` reads the WHOLE canonicalized graph (a rebuild,
KG-AC-28); a folder set scopes to canonical edges touching that batch's exported nodes (an edge whose
src or dst canonical entity is in the node set).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from core import CanonicalEdge, CanonicalNode

_NODE_SQL = """
    SELECT ce.canonical_id, ce.entity_type, ce.normalized_form, ce.attributes,
           array_agg(DISTINCT e.folder_id) AS folders, count(*) AS mention_count,
           ce.canonical_name, ce.aliases, ce.reference_only,
           bool_or(e.is_abstract) AS is_abstract,
           bool_or(e.extractor = 'derived') AS is_derived
      FROM public.kg_entities e
      JOIN public.kg_canonical_entities ce ON ce.canonical_id = e.canonical_id
     WHERE e.stage = 'canonicalized' {node_filter}
     GROUP BY ce.canonical_id, ce.entity_type, ce.normalized_form, ce.attributes,
              ce.canonical_name, ce.aliases, ce.reference_only
     ORDER BY ce.canonical_id
"""

# KG-AC-105 (export half): every canonical id the SCOPE's plane of record still contains. The
# reconcile deletes whatever the target database holds beyond this set — sound only because one
# database holds exactly one scope (KG-AC-97/98).
_SCOPE_CANONICAL_IDS_SQL = """
    SELECT DISTINCT ce.canonical_id
      FROM public.kg_canonical_entities ce
      JOIN public.kg_entities e ON e.canonical_id = ce.canonical_id
     WHERE ce.graph_scope = %s AND e.stage = 'canonicalized'
"""

_EDGE_SQL = """
    SELECT src_canonical_id, relation_type, dst_canonical_id, support_count, confidence, evidence_text
      FROM public.kg_canonical_edges
     {edge_filter}
     ORDER BY src_canonical_id, relation_type, dst_canonical_id
"""


def batch_pack_name(cur, folder_ids: Optional[Sequence[str]] = None) -> Optional[str]:
    """KG-AC-83/86: the ontology_pack to load so `iri` resolution has a pack to read (mirrors
    `entity_canonicalization.store.batch_pack_name` exactly, scoped to `stage = 'canonicalized'`
    since kg_export reads past that point). ``folder_ids=None`` (a full rebuild, KG-AC-28) has no
    single batch to scope to — picks ANY one canonicalized row's pack (the common, single-pack-in-
    practice case); a genuinely mixed-pack rebuild degrades some nodes to bare names, which is
    explicitly not an error (KG-AC-86)."""
    if folder_ids:
        cur.execute(
            """SELECT ontology_pack FROM public.kg_entities
                WHERE folder_id = ANY(%s) AND stage = 'canonicalized' AND ontology_pack IS NOT NULL
                LIMIT 1""",
            (list(folder_ids),),
        )
    else:
        cur.execute(
            """SELECT ontology_pack FROM public.kg_entities
                WHERE stage = 'canonicalized' AND ontology_pack IS NOT NULL
                LIMIT 1""",
        )
    row = cur.fetchone()
    return row[0] if row else None


def read_canonical_graph(
    cur, folder_ids: Optional[Sequence[str]] = None, pack=None, graph_scope: Optional[str] = None,
) -> Tuple[List[CanonicalNode], List[CanonicalEdge]]:
    """KG-AC-83/86 (v13): ``pack`` (optional — the CALLER's already-loaded ontology pack via
    `ontologies.load_pack(batch_pack_name(...))`, matching the entity_canonicalization worker's own
    load-in-the-worker-file precedent — store.py stays pure DB access, never imports `ontologies`
    itself) resolves each node's/edge's `iri` for the ontology-qualified export. No pack (or an
    unknown entity_type/relation_type within it) simply exports bare names (KG-AC-86's own "not an
    error" clause) — never fails the export."""
    # v16 (KG-AC-97): EVERY export read is scope-filtered. The rebuild path (`folder_ids=None`,
    # KG-AC-28) is the one that makes this load-bearing rather than redundant: without the filter it
    # reads every scope's canonical rows and MERGEs another department's graph into this database.
    # The folder-scoped path is implicitly single-scope (the worker validates that), but is filtered
    # too so there is ONE rule rather than a path that happens to be safe.
    filters, params = [], []
    if folder_ids:
        filters.append("AND e.folder_id = ANY(%s)")
        params.append(list(folder_ids))
    if graph_scope:
        filters.append("AND ce.graph_scope = %s")
        params.append(graph_scope)
    cur.execute(_NODE_SQL.format(node_filter=" ".join(filters)), tuple(params))
    nodes = [
        CanonicalNode(
            canonical_id=str(r[0]), entity_type=r[1], normalized_form=r[2], attributes=r[3] or {},
            provenance={"folders": [f for f in (r[4] or []) if f], "mentions": r[5]},
            entity_iri=_entity_iri(pack, r[1]),
            # v16 (KG-AC-104): display + honesty properties. `is_abstract`/`is_derived` are read
            # from the contributing MENTIONS (the canonical row carries neither), so a cluster with
            # any derived contributor exports as derived — the marker exists to stop a consumer
            # mistaking an entailment for something the document asserted.
            canonical_name=r[6], aliases=list(r[7] or []), reference_only=bool(r[8]),
            is_abstract=bool(r[9]), is_derived=bool(r[10]),
        )
        for r in cur.fetchall()
    ]

    edge_filters, edge_params = [], []
    if folder_ids:
        node_ids = [n.canonical_id for n in nodes]
        edge_filters.append("(src_canonical_id::text = ANY(%s) OR dst_canonical_id::text = ANY(%s))")
        edge_params.extend([node_ids, node_ids])
    if graph_scope:  # KG-AC-97 — same rule on the edge read
        edge_filters.append("graph_scope = %s")
        edge_params.append(graph_scope)
    where = ("WHERE " + " AND ".join(edge_filters)) if edge_filters else ""
    cur.execute(_EDGE_SQL.format(edge_filter=where), tuple(edge_params))
    edges = [
        CanonicalEdge(
            src_canonical_id=str(r[0]), relation_type=r[1], dst_canonical_id=str(r[2]),
            support_count=r[3], confidence=float(r[4]) if r[4] is not None else None,
            evidence_text=list(r[5]) if r[5] is not None else [],
            relation_iri=_relation_iri(pack, r[1]),
        )
        for r in cur.fetchall()
    ]
    return nodes, edges


def scope_canonical_ids(cur, graph_scope: str) -> List[str]:
    """Every canonical id this scope's plane of record still contains (KG-AC-105 export half)."""
    cur.execute(_SCOPE_CANONICAL_IDS_SQL, (graph_scope,))
    return [str(r[0]) for r in cur.fetchall()]


def batch_graph_scope(cur, folder_ids: Optional[Sequence[str]] = None) -> Optional[str]:
    """The scope this export serves, derived from the canonicalized rows it is about to read
    (KG-AC-97/98). Mixed scopes fail loud: one export writes one database, and a database holds
    exactly one scope — exporting a mixed batch would put one tenant's graph into another's store.
    A full rebuild (``folder_ids=None``) reads whatever single scope the canonicalized graph has."""
    if folder_ids:
        cur.execute(
            """SELECT DISTINCT graph_scope FROM public.kg_entities
                WHERE folder_id = ANY(%s) AND stage = 'canonicalized'""",
            (list(folder_ids),),
        )
    else:
        cur.execute(
            "SELECT DISTINCT graph_scope FROM public.kg_entities WHERE stage = 'canonicalized'")
    scopes = [r[0] for r in cur.fetchall()]
    if not scopes:
        return None
    if len(scopes) > 1:
        raise ValueError(
            f"kg_export: mixed graph_scope in one export ({sorted(scopes)}) — one export writes one "
            "database and a database holds exactly one scope"
        )
    return scopes[0]


def _entity_iri(pack, entity_type: str) -> Optional[str]:
    if pack is None:
        return None
    et = pack.entity_types.get(entity_type)
    return et.iri if et else None


def _relation_iri(pack, relation_type: str) -> Optional[str]:
    if pack is None:
        return None
    rel = pack.relations.get(relation_type)
    return rel.iri if rel else None
