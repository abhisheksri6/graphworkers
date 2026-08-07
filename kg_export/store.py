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
    SELECT ce.canonical_id, ce.entity_type, ce.normalized_form, ce.external_id,
           array_agg(DISTINCT e.folder_id) AS folders, count(*) AS mention_count
      FROM public.kg_entities e
      JOIN public.kg_canonical_entities ce ON ce.canonical_id = e.canonical_id
     WHERE e.stage = 'canonicalized' {node_filter}
     GROUP BY ce.canonical_id, ce.entity_type, ce.normalized_form, ce.external_id
     ORDER BY ce.canonical_id
"""

_EDGE_SQL = """
    SELECT src_canonical_id, relation_type, dst_canonical_id, support_count, confidence, evidence_text
      FROM public.kg_canonical_edges
     {edge_filter}
     ORDER BY src_canonical_id, relation_type, dst_canonical_id
"""


def read_canonical_graph(
    cur, folder_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[CanonicalNode], List[CanonicalEdge]]:
    if folder_ids:
        cur.execute(_NODE_SQL.format(node_filter="AND e.folder_id = ANY(%s)"), (list(folder_ids),))
    else:
        cur.execute(_NODE_SQL.format(node_filter=""))
    nodes = [
        CanonicalNode(
            canonical_id=str(r[0]), entity_type=r[1], normalized_form=r[2], external_id=r[3],
            provenance={"folders": [f for f in (r[4] or []) if f], "mentions": r[5]},
        )
        for r in cur.fetchall()
    ]

    if folder_ids:
        node_ids = [n.canonical_id for n in nodes]
        cur.execute(
            _EDGE_SQL.format(
                edge_filter="WHERE src_canonical_id::text = ANY(%s) OR dst_canonical_id::text = ANY(%s)"
            ),
            (node_ids, node_ids),
        )
    else:
        cur.execute(_EDGE_SQL.format(edge_filter=""))
    edges = [
        CanonicalEdge(
            src_canonical_id=str(r[0]), relation_type=r[1], dst_canonical_id=str(r[2]),
            support_count=r[3], confidence=float(r[4]) if r[4] is not None else None,
            evidence_text=list(r[5]) if r[5] is not None else [],
        )
        for r in cur.fetchall()
    ]
    return nodes, edges
