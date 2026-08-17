"""DB engine for entity_canonicalization — the one-transaction batch canonicalization (KG-AC-40).

Runs on the raw psycopg2 connection behind the injected session (the worker wrapper commits on
success / rolls back on any failure, so a mid-batch error leaves the whole batch all-`staged`). Reads
the hold-released batch's `staged` kg_entities, clusters them (core), reconciles each cluster's type
(D3), resolves-or-mints a `canonical_id` against the `kg_canonical_entities` index by `canonical_key`
(race-safe `ON CONFLICT DO NOTHING RETURNING`, KG-AC-38 — cross-run reuse), and assigns canonical_id
+ transitions `staged → canonicalized`. Edges collapse in the canonical view via the entities'
canonical_id (kg_export projects one node per canonical_id).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from psycopg2.extras import Json

from core import (
    Mention, aggregate_edge_group, canonical_key, choose_canonical_name, cluster_identifier,
    cluster_mentions, identifier_values, merge_attributes, normalize_surface, reconcile_type,
)

_DEFAULT_FLOOR = 0.80
_DEFAULT_CEILING = 0.95


def batch_pack_name(cur, folder_ids: Sequence[str]) -> Optional[str]:
    """The ontology_pack stamped on the batch's staged entities (they carry it from extraction) —
    canonicalization loads the SAME pack so type reconciliation uses the right hierarchy."""
    cur.execute(
        """SELECT ontology_pack FROM public.kg_entities
            WHERE folder_id = ANY(%s) AND stage = 'staged' AND ontology_pack IS NOT NULL
            LIMIT 1""",
        (list(folder_ids),),
    )
    row = cur.fetchone()
    return row[0] if row else None


def read_staged_mentions(cur, folder_ids: Sequence[str], pack=None) -> List[Mention]:
    """``pack`` (KG-AC-24 amended, P25) resolves each mention's pack-declared IDENTIFIER facts from
    its own ``attributes`` — the identity evidence Tier-1 clustering runs on. Omitted/None keeps
    exactly the pre-P25 behaviour (no identifiers, surface-only clustering)."""
    cur.execute(
        """SELECT entity_uid, entity_type, surface_form, attributes, folder_id, declared_aliases
             FROM public.kg_entities
            WHERE folder_id = ANY(%s) AND stage = 'staged'
            ORDER BY id""",
        (list(folder_ids),),
    )
    out: List[Mention] = []
    for uid, etype, surface, attributes, folder_id, declared_aliases in cur.fetchall():
        m = Mention(entity_uid=uid, entity_type=etype, surface_form=surface, folder_id=folder_id)
        m.normalized_form = normalize_surface(surface)
        m.identifiers = identifier_values(attributes or [], etype, pack)
        # KG-AC-96 (P26): the document's own declared bindings, scoped by folder_id (one folder ==
        # one document). Pre-P26 rows default to '[]' from the migration, so the Tier-2 pass is a
        # no-op over historical data rather than an error.
        m.declared_aliases = [a for a in (declared_aliases or []) if isinstance(a, str) and a.strip()]
        out.append(m)
    return out


def _resolve_or_mint(cur, entity_type: str, normalized_form: str, *,
                     pack=None, fallback_surface: str = "") -> Tuple[str, bool]:
    """Race-safe resolve-or-mint (KG-AC-38, key format amended KG-AC-79): INSERT ON CONFLICT DO
    NOTHING RETURNING mints a novel canonical; on conflict, distinguish a LEGITIMATE cross-run
    match (existing row has the SAME normalized_form — reuse it, KG-AC-38) from a genuine SLUG
    COLLISION (existing row has a DIFFERENT normalized_form — two distinct real clusters whose
    human-readable keys happened to collide) — the latter retries with KG-AC-79's deterministic
    suffix (-2, -3, ...) rather than incorrectly reusing an unrelated entity's canonical_id.
    Returns (canonical_id, was_minted)."""
    suffix = 0
    while True:
        key = canonical_key(entity_type, normalized_form, suffix, pack=pack,
                            fallback_surface=fallback_surface)
        cur.execute(
            """INSERT INTO public.kg_canonical_entities
                   (canonical_key, entity_type, normalized_form)
               VALUES (%s, %s, %s)
               ON CONFLICT (canonical_key) DO NOTHING
               RETURNING canonical_id""",
            (key, entity_type, normalized_form),
        )
        row = cur.fetchone()
        if row is not None:
            return str(row[0]), True
        cur.execute(
            "SELECT canonical_id, normalized_form FROM public.kg_canonical_entities WHERE canonical_key = %s",
            (key,),
        )
        existing_id, existing_norm = cur.fetchone()
        if existing_norm == normalized_form:
            return str(existing_id), False  # legitimate cross-run reuse (KG-AC-38)
        suffix += 1  # genuine collision — try the next deterministic suffix


def _assign(cur, entity_uids: Sequence[str], canonical_id: str, normalized_form: str) -> None:
    cur.execute(
        """UPDATE public.kg_entities
              SET canonical_id = %s, normalized_form = %s, stage = 'canonicalized'
            WHERE entity_uid = ANY(%s)""",
        (canonical_id, normalized_form, list(entity_uids)),
    )


def _mentions_for_canonical(cur, canonical_id: str) -> List[Mention]:
    """KG-AC-76/77: EVERY kg_entities row currently sharing this canonical_id — not just this
    batch's cluster — so the display name / alias set stays true to "every string that resolved to
    this instance" across cross-run merges (KG-AC-38), not only the founding batch. Re-derived from
    the durable kg_entities rows each time, rather than accumulated in application state, so it is
    idempotent and correct after a re-run."""
    cur.execute(
        """SELECT entity_uid, entity_type, surface_form, source_doc_id, source_chunk_id,
                  span_start, is_abstract
             FROM public.kg_entities
            WHERE canonical_id = %s
            ORDER BY id""",
        (canonical_id,),
    )
    return [
        Mention(entity_uid=r[0], entity_type=r[1], surface_form=r[2],
               source_doc_id=r[3], source_chunk_id=r[4], span_start=r[5], is_abstract=bool(r[6]))
        for r in cur.fetchall()
    ]


def _update_display_name(cur, canonical_id: str, canonical_name: str, aliases: List[str]) -> None:
    cur.execute(
        """UPDATE public.kg_canonical_entities
              SET canonical_name = %s, aliases = %s
            WHERE canonical_id = %s""",
        (canonical_name, Json(aliases), canonical_id),
    )


def _attributes_for_canonical(cur, canonical_id: str) -> List[List[Dict[str, Any]]]:
    """KG-AC-78: every mention's raw ``attributes`` list currently sharing this canonical_id —
    same full-recompute posture as `_mentions_for_canonical` (P10), so a later document's fact
    (agreeing OR conflicting) is reflected, not frozen at the founding batch."""
    cur.execute(
        "SELECT attributes FROM public.kg_entities WHERE canonical_id = %s ORDER BY id",
        (canonical_id,),
    )
    return [r[0] or [] for r in cur.fetchall()]


def _update_attributes(cur, canonical_id: str, merged: Dict[str, List[Dict[str, Any]]]) -> None:
    cur.execute(
        "UPDATE public.kg_canonical_entities SET attributes = %s WHERE canonical_id = %s",
        (Json(merged), canonical_id),
    )


def _affected_canonical_triples(cur, entity_uids: Sequence[str]) -> List[Tuple[str, str, str]]:
    """KG-AC-47: the distinct (src_canonical_id, relation_type, dst_canonical_id) triples touched by
    this batch's just-canonicalized entities (either endpoint) — scopes the aggregation re-derive to
    only what this batch could have changed. A mention-edge's src/dst are always from the same
    folder (extraction is per-folder), so every edge this batch could affect has at least one
    endpoint in ``entity_uids``; the OTHER endpoint may belong to a different (already-canonicalized)
    folder only if it's in the same batch, since edges never span folders."""
    if not entity_uids:
        return []
    cur.execute(
        """SELECT DISTINCT se.canonical_id, ed.relation_type, de.canonical_id
             FROM public.kg_edges ed
             JOIN public.kg_entities se ON se.entity_uid = ed.src_entity_uid
             JOIN public.kg_entities de ON de.entity_uid = ed.dst_entity_uid
            WHERE se.canonical_id IS NOT NULL AND de.canonical_id IS NOT NULL
              AND (ed.src_entity_uid = ANY(%s) OR ed.dst_entity_uid = ANY(%s))""",
        (list(entity_uids), list(entity_uids)),
    )
    return [(str(r[0]), r[1], str(r[2])) for r in cur.fetchall()]


def _rows_for_canonical_triple(
    cur, src_canonical_id: str, relation_type: str, dst_canonical_id: str,
) -> List[Dict[str, Any]]:
    """Every contributing mention-edge (any batch, any time) for one canonical triple — the
    aggregation re-derives from this full set so a LATER batch reinforcing an EXISTING canonical
    relationship (KG-AC-38-style cross-run reuse) is reflected correctly, not just this batch's rows.
    Stable order (by id) so aggregate_edge_group's evidence tie-break is deterministic."""
    cur.execute(
        """SELECT ed.folder_id, ed.confidence, ed.evidence_text, ed.source_doc_id
             FROM public.kg_edges ed
             JOIN public.kg_entities se ON se.entity_uid = ed.src_entity_uid
             JOIN public.kg_entities de ON de.entity_uid = ed.dst_entity_uid
            WHERE se.canonical_id = %s AND ed.relation_type = %s AND de.canonical_id = %s
            ORDER BY ed.id""",
        (src_canonical_id, relation_type, dst_canonical_id),
    )
    return [
        {"folder_id": r[0], "confidence": float(r[1]) if r[1] is not None else None,
         "evidence_text": r[2], "source_doc_id": r[3]}
        for r in cur.fetchall()
    ]


def _upsert_canonical_edge(
    cur, src_canonical_id: str, relation_type: str, dst_canonical_id: str, agg: Dict[str, Any],
) -> None:
    cur.execute(
        """INSERT INTO public.kg_canonical_edges
               (src_canonical_id, relation_type, dst_canonical_id, support_count, confidence,
                evidence_text, source_doc_ids)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (src_canonical_id, relation_type, dst_canonical_id) DO UPDATE SET
               support_count = EXCLUDED.support_count, confidence = EXCLUDED.confidence,
               evidence_text = EXCLUDED.evidence_text, source_doc_ids = EXCLUDED.source_doc_ids,
               updated_at = now()""",
        (src_canonical_id, relation_type, dst_canonical_id,
         agg["support_count"], agg["confidence"], agg["evidence_text"], Json(agg["source_doc_ids"])),
    )


def aggregate_canonical_edges(cur, entity_uids: Sequence[str]) -> int:
    """KG-AC-47: after this batch's entities are assigned canonical_id, collapse duplicate canonical
    edges for every triple the batch touches into ``kg_canonical_edges``. Runs in the SAME
    transaction as the rest of canonicalize_batch (KG-AC-40 atomicity — a failure here rolls back
    the whole batch, never leaving entities canonicalized with stale/missing edge aggregation).
    Returns the number of canonical edge triples written."""
    triples = _affected_canonical_triples(cur, entity_uids)
    for src_cid, rtype, dst_cid in triples:
        rows = _rows_for_canonical_triple(cur, src_cid, rtype, dst_cid)
        agg = aggregate_edge_group(rows)
        _upsert_canonical_edge(cur, src_cid, rtype, dst_cid, agg)
    return len(triples)


def canonicalize_batch(db, folder_ids: Sequence[str], *, fuzzy_floor: float = _DEFAULT_FLOOR,
                       fuzzy_ceiling: float = _DEFAULT_CEILING, pack=None,
                       adjudicate: Optional[Callable[[Mention, Mention], bool]] = None) -> Dict[str, int]:
    """Canonicalize the batch in ONE transaction. Returns {canonical_count, merged_count,
    minted_count}. A re-run after completion is a no-op (no rows left in `staged`) — idempotent
    (KG-AC-22)."""
    conn = db.connection().connection
    minted = 0
    merged = 0
    canonical_ids: set = set()
    with conn.cursor() as cur:
        mentions = read_staged_mentions(cur, folder_ids, pack)
        if not mentions:
            return {"canonical_count": 0, "merged_count": 0, "minted_count": 0}

        # v16 (KG-AC-102): the pack MUST reach clustering — the type-compatibility rule is what
        # lets `Organization` and its subtype `InvestmentAdviser` be recognised as one entity
        # (KG-AC-23's own premise). Without it `types_compatible` degrades to exact equality, which
        # is safe but would silently stop the cross-type merges this capability depends on.
        clusters = cluster_mentions(mentions, fuzzy_floor=fuzzy_floor, fuzzy_ceiling=fuzzy_ceiling,
                                    adjudicate=adjudicate, pack=pack)
        for cluster in clusters:
            rtype = reconcile_type([m.entity_type for m in cluster], pack) if pack else cluster[0].entity_type
            # KG-AC-79 amended (P25): an identifier-bearing cluster keys on its IDENTIFIER, not on
            # whichever surface sorted first — otherwise Tier-1 merging would fix within-batch
            # fragmentation while BREAKING cross-run reuse (KG-AC-38), since two runs seeing
            # different surface subsets of the same entity would mint different canonical_keys.
            # `normalized_form` is the match key by definition, so it carries the identity basis;
            # the human-readable surfaces live in canonical_name/aliases (KG-AC-76/77).
            norm = cluster_identifier(cluster, pack) or cluster[0].normalized_form
            cid, was_minted = _resolve_or_mint(cur, rtype, norm, pack=pack,
                                               fallback_surface=cluster[0].surface_form)
            minted += 1 if was_minted else 0
            merged += 0 if was_minted else 1
            canonical_ids.add(cid)
            _assign(cur, [m.entity_uid for m in cluster], cid, norm)

            # KG-AC-76/77: recompute the display name/alias set from the FULL set of entities now
            # sharing this canonical_id (this batch's cluster + any prior batch's, via cross-run
            # reuse) — not just this batch's cluster — so aliases stay true across merges.
            full_set = _mentions_for_canonical(cur, cid)
            canonical_name, aliases = choose_canonical_name(full_set)
            _update_display_name(cur, cid, canonical_name, aliases)

            # KG-AC-78: same full-recompute posture — merge facts from every mention now sharing
            # this canonical_id, never last-write-wins.
            merged_attrs = merge_attributes(_attributes_for_canonical(cur, cid))
            _update_attributes(cur, cid, merged_attrs)

        # KG-AC-47: aggregate duplicate canonical edges for every triple this batch touches, in the
        # SAME transaction (KG-AC-40 — a failure anywhere above rolls this back too).
        aggregate_canonical_edges(cur, [m.entity_uid for m in mentions])

    return {"canonical_count": len(canonical_ids), "merged_count": merged, "minted_count": minted}
