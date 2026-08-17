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

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from psycopg2.extras import Json

from core import (
    ACCEPT, AMBIGUOUS, Mention, aggregate_edge_group, canonical_key, choose_canonical_name,
    cluster_identifier, cluster_mentions, identifier_values, match_band, merge_attributes,
    normalize_surface, reconcile_type, types_compatible,
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


def batch_graph_scope(cur, folder_ids: Sequence[str]) -> Optional[str]:
    """The ONE graph scope this batch belongs to, derived from its own staged rows (KG-AC-97/98).

    Fails loud on a mixed-scope batch: such a batch has no single tenancy, and canonicalizing it
    would write one department's identities into another's partition — the exact leak v16 exists to
    close. Returns None when the batch has no staged rows at all (a clean no-op for the caller)."""
    cur.execute(
        """SELECT DISTINCT graph_scope FROM public.kg_entities
            WHERE folder_id = ANY(%s) AND stage = 'staged'""",
        (list(folder_ids),),
    )
    scopes = [r[0] for r in cur.fetchall()]
    if not scopes:
        return None
    if len(scopes) > 1:
        raise ValueError(
            f"entity_canonicalization: mixed graph_scope in one batch ({sorted(scopes)}) — a batch "
            "spans exactly one scope; refusing to write one scope's identities into another's"
        )
    return scopes[0]


def _lock_scope(cur, graph_scope: str) -> None:
    """KG-AC-100: serialize same-scope canonicalization. Transaction-scoped, so it composes with
    the KG-AC-40 all-or-nothing wrapper and releases on commit OR rollback with no cleanup path to
    forget. This is what removes the last-write-wins on the canonical name/alias/attribute
    recompute and the edge aggregation, the deadlock exposure between overlapping batches (any
    exception fails the whole batch, with no retry), and the in-flight blindness that let two
    concurrent batches mint near-duplicates. Different scopes never contend."""
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"kg_canon:{graph_scope}",))


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


def _candidate_canonicals(cur, graph_scope: str, normalized_form: str) -> List[Tuple[str, Mention]]:
    """Already-canonicalized entities IN THIS SCOPE sharing the batch cluster's block key —
    KG-AC-99's candidate set. Uses `idx_kg_canonical_scope_norm`; the block key is the first token
    of the normalized form, the same coarse key in-batch blocking uses, so cross-run candidates are
    generated exactly like in-batch ones rather than by a second, divergent rule."""
    first = (normalized_form or "").split(" ", 1)[0]
    if not first:
        return []
    cur.execute(
        """SELECT canonical_id, entity_type, COALESCE(canonical_name, normalized_form), normalized_form
             FROM public.kg_canonical_entities
            WHERE graph_scope = %s AND (normalized_form = %s OR normalized_form LIKE %s)
            ORDER BY canonical_id""",
        (graph_scope, normalized_form, f"{first} %"),
    )
    out: List[Tuple[str, Mention]] = []
    for cid, etype, name, norm in cur.fetchall():
        m = Mention(entity_uid=f"canon:{cid}", entity_type=etype, surface_form=name or "")
        m.normalized_form = norm or ""
        out.append((str(cid), m))
    return out


def _sharpen_type(cur, canonical_id: str, existing_type: str, cluster_type: str, pack) -> None:
    """KG-AC-103: the canonical row keeps the MOST SPECIFIC type seen so far. With the root-type
    key, `Organization` and `InvestmentAdviser` resolve to one key — this is what keeps the useful
    specific type on the row instead of whichever batch happened to arrive first."""
    if pack is None or existing_type == cluster_type:
        return
    best = pack.most_specific_type([existing_type, cluster_type])
    if best and best != existing_type:
        cur.execute(
            "UPDATE public.kg_canonical_entities SET entity_type = %s WHERE canonical_id = %s",
            (best, canonical_id),
        )


def _resolve_or_mint(cur, entity_type: str, normalized_form: str, *,
                     graph_scope: str, pack=None, fallback_surface: str = "",
                     rep: Optional[Mention] = None,
                     adjudicate: Optional[Callable[[Mention, Mention], bool]] = None,
                     fuzzy_floor: float = _DEFAULT_FLOOR,
                     fuzzy_ceiling: float = _DEFAULT_CEILING) -> Tuple[str, bool]:
    """Race-safe resolve-or-mint (KG-AC-38, key format amended KG-AC-79): INSERT ON CONFLICT DO
    NOTHING RETURNING mints a novel canonical; on conflict, distinguish a LEGITIMATE cross-run
    match (existing row has the SAME normalized_form — reuse it, KG-AC-38) from a genuine SLUG
    COLLISION (existing row has a DIFFERENT normalized_form — two distinct real clusters whose
    human-readable keys happened to collide) — the latter retries with KG-AC-79's deterministic
    suffix (-2, -3, ...) rather than incorrectly reusing an unrelated entity's canonical_id.
    Returns (canonical_id, was_minted)."""
    # --- 1. the exact-key FAST PATH (KG-AC-99 keeps it as the fast path, unchanged in spirit) ---
    suffix = 0
    while True:
        key = canonical_key(entity_type, normalized_form, suffix, pack=pack,
                            fallback_surface=fallback_surface)
        cur.execute(
            """SELECT canonical_id, normalized_form, entity_type
                 FROM public.kg_canonical_entities
                WHERE graph_scope = %s AND canonical_key = %s""",
            (graph_scope, key),
        )
        hit = cur.fetchone()
        if hit is None:
            break
        existing_id, existing_norm, existing_type = str(hit[0]), hit[1], hit[2]
        if existing_norm == normalized_form:
            # clarify F3: a key hit whose stored type is INCOMPATIBLE is a sibling collision under
            # the root-type key (Bank vs InvestmentAdviser), not the same entity. Blind reuse here
            # would merge exactly what KG-AC-102 forbids, so it goes through the same adjudicated
            # path any other ambiguous pair uses; a rejection mints a suffixed sibling instead.
            if types_compatible(entity_type, existing_type, pack):
                _sharpen_type(cur, existing_id, existing_type, entity_type, pack)
                return existing_id, False  # legitimate cross-run reuse (KG-AC-38)
            if rep is not None and adjudicate is not None:
                other = Mention(entity_uid=f"canon:{existing_id}", entity_type=existing_type,
                                surface_form=existing_norm)
                other.normalized_form = existing_norm
                if adjudicate(rep, other):
                    _sharpen_type(cur, existing_id, existing_type, entity_type, pack)
                    return existing_id, False
        suffix += 1  # slug collision OR a rejected sibling — try the next deterministic suffix

    # --- 2. cross-run MATCH path (KG-AC-99) — the half the design promised and never had ---
    if rep is not None:
        for cid, other in _candidate_canonicals(cur, graph_scope, normalized_form):
            verdict = match_band(rep, other, fuzzy_floor=fuzzy_floor,
                                 fuzzy_ceiling=fuzzy_ceiling, pack=pack)
            if verdict == ACCEPT or (verdict == AMBIGUOUS and adjudicate and adjudicate(rep, other)):
                _sharpen_type(cur, cid, other.entity_type, entity_type, pack)
                return cid, False

    # --- 3. novel entity: race-safe mint (KG-AC-38), now scoped ---
    cur.execute(
        """INSERT INTO public.kg_canonical_entities
               (graph_scope, canonical_key, entity_type, normalized_form)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (graph_scope, canonical_key) DO NOTHING
           RETURNING canonical_id""",
        (graph_scope, key, entity_type, normalized_form),
    )
    row = cur.fetchone()
    if row is not None:
        return str(row[0]), True
    # a concurrent batch minted this exact key first — reuse it, never split (KG-AC-38)
    cur.execute(
        """SELECT canonical_id FROM public.kg_canonical_entities
            WHERE graph_scope = %s AND canonical_key = %s""",
        (graph_scope, key),
    )
    return str(cur.fetchone()[0]), False


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
                  span_start, is_abstract, reference_only
             FROM public.kg_entities
            WHERE canonical_id = %s
            ORDER BY id""",
        (canonical_id,),
    )
    return [
        Mention(entity_uid=r[0], entity_type=r[1], surface_form=r[2],
               source_doc_id=r[3], source_chunk_id=r[4], span_start=r[5], is_abstract=bool(r[6]),
               reference_only=bool(r[7]))
        for r in cur.fetchall()
    ]


def _update_display_name(cur, canonical_id: str, canonical_name: str, aliases: List[str],
                         reference_only: bool) -> None:
    """KG-AC-76/77 display fields + KG-AC-94's cross-document `reference_only`. All three are
    recomputed from the FULL mention set on every touch (the P10 posture), which is what makes the
    stub → described transition work: the stub is the join point a later document merges into, so
    freezing any of them at the founding batch would leave a fully-read entity still labelled a
    stub."""
    cur.execute(
        """UPDATE public.kg_canonical_entities
              SET canonical_name = %s, aliases = %s, reference_only = %s
            WHERE canonical_id = %s""",
        (canonical_name, Json(aliases), reference_only, canonical_id),
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
    graph_scope: str,
) -> None:
    # KG-AC-97: the canonical edge is scoped like every other kg row. Its endpoints already belong
    # to exactly one scope, so this is the row's own tenancy stamp rather than a second identity —
    # which is why the (src, type, dst) primary key needs no scope component.
    cur.execute(
        """INSERT INTO public.kg_canonical_edges
               (src_canonical_id, relation_type, dst_canonical_id, graph_scope, support_count,
                confidence, evidence_text, source_doc_ids)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (src_canonical_id, relation_type, dst_canonical_id) DO UPDATE SET
               support_count = EXCLUDED.support_count, confidence = EXCLUDED.confidence,
               evidence_text = EXCLUDED.evidence_text, source_doc_ids = EXCLUDED.source_doc_ids,
               updated_at = now()""",
        (src_canonical_id, relation_type, dst_canonical_id, graph_scope,
         agg["support_count"], agg["confidence"], agg["evidence_text"], Json(agg["source_doc_ids"])),
    )


def aggregate_canonical_edges(cur, entity_uids: Sequence[str], graph_scope: str = "legacy") -> int:
    """KG-AC-47: after this batch's entities are assigned canonical_id, collapse duplicate canonical
    edges for every triple the batch touches into ``kg_canonical_edges``. Runs in the SAME
    transaction as the rest of canonicalize_batch (KG-AC-40 atomicity — a failure here rolls back
    the whole batch, never leaving entities canonicalized with stale/missing edge aggregation).
    Returns the number of canonical edge triples written."""
    triples = _affected_canonical_triples(cur, entity_uids)
    for src_cid, rtype, dst_cid in triples:
        rows = _rows_for_canonical_triple(cur, src_cid, rtype, dst_cid)
        agg = aggregate_edge_group(rows)
        _upsert_canonical_edge(cur, src_cid, rtype, dst_cid, agg, graph_scope)
    return len(triples)


def retract_orphans(cur, graph_scope: str) -> Tuple[int, int]:
    """KG-AC-105 (Postgres half): remove canonical rows this scope no longer has evidence for.
    Returns (edges_removed, entities_removed).

    Why it cannot be folded into `aggregate_canonical_edges`: that function derives its work list
    from the mention-edges that CURRENTLY exist, so a canonical edge whose contributors have all
    disappeared is never visited by it — the vanished row is precisely the one no longer reachable
    from the data. Retraction therefore has to look from the canonical side inwards.

    Runs inside the batch transaction (KG-AC-40), so an abort restores everything it removed; and
    it is scope-local, so a batch can never reach into another tenant's partition. Edges first —
    they reference the canonical entities."""
    cur.execute(
        """DELETE FROM public.kg_canonical_edges ce
            WHERE ce.graph_scope = %s
              AND NOT EXISTS (
                  SELECT 1
                    FROM public.kg_edges ed
                    JOIN public.kg_entities se ON se.entity_uid = ed.src_entity_uid
                    JOIN public.kg_entities de ON de.entity_uid = ed.dst_entity_uid
                   WHERE se.canonical_id = ce.src_canonical_id
                     AND de.canonical_id = ce.dst_canonical_id
                     AND ed.relation_type = ce.relation_type
              )""",
        (graph_scope,),
    )
    edges_removed = cur.rowcount

    cur.execute(
        """DELETE FROM public.kg_canonical_entities c
            WHERE c.graph_scope = %s
              AND NOT EXISTS (
                  SELECT 1 FROM public.kg_entities e WHERE e.canonical_id = c.canonical_id
              )""",
        (graph_scope,),
    )
    return edges_removed, cur.rowcount


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
        # KG-AC-97/98: one batch belongs to exactly one scope, derived from its own staged rows;
        # KG-AC-100: hold that scope's lock for the rest of THIS transaction before reading
        # anything, so a concurrent same-scope batch cannot interleave with the resolve/recompute.
        graph_scope = batch_graph_scope(cur, folder_ids)
        if graph_scope is None:
            return {"canonical_count": 0, "merged_count": 0, "minted_count": 0,
                    "retracted_entity_count": 0, "retracted_edge_count": 0}
        _lock_scope(cur, graph_scope)

        mentions = read_staged_mentions(cur, folder_ids, pack)
        if not mentions:
            return {"canonical_count": 0, "merged_count": 0, "minted_count": 0,
                    "retracted_entity_count": 0, "retracted_edge_count": 0}

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
            # `rep` carries the cluster's identity into the cross-run match (KG-AC-99): its
            # normalized form is the resolved basis, so an identifier-keyed cluster matches on the
            # identifier and a surface-keyed one on the surface — the same basis the key uses.
            rep = replace(cluster[0], normalized_form=norm)
            cid, was_minted = _resolve_or_mint(
                cur, rtype, norm, graph_scope=graph_scope, pack=pack,
                fallback_surface=cluster[0].surface_form, rep=rep, adjudicate=adjudicate,
                fuzzy_floor=fuzzy_floor, fuzzy_ceiling=fuzzy_ceiling)
            minted += 1 if was_minted else 0
            merged += 0 if was_minted else 1
            canonical_ids.add(cid)
            _assign(cur, [m.entity_uid for m in cluster], cid, norm)

            # KG-AC-76/77: recompute the display name/alias set from the FULL set of entities now
            # sharing this canonical_id (this batch's cluster + any prior batch's, via cross-run
            # reuse) — not just this batch's cluster — so aliases stay true across merges.
            full_set = _mentions_for_canonical(cur, cid)
            canonical_name, aliases = choose_canonical_name(full_set)
            # KG-AC-94: reference_only iff EVERY contributing mention is — "described beats
            # referenced", so one fully-read mention clears the stub marker for the whole instance.
            # Deliberately not a majority vote and not first-writer-wins: the AC fixes the rule
            # here precisely so the outcome cannot depend on which mention the merge happens to
            # pick as canonical.
            reference_only = all(m.reference_only for m in full_set)
            _update_display_name(cur, cid, canonical_name, aliases, reference_only)

            # KG-AC-78: same full-recompute posture — merge facts from every mention now sharing
            # this canonical_id, never last-write-wins.
            merged_attrs = merge_attributes(_attributes_for_canonical(cur, cid))
            _update_attributes(cur, cid, merged_attrs)

        # KG-AC-47: aggregate duplicate canonical edges for every triple this batch touches, in the
        # SAME transaction (KG-AC-40 — a failure anywhere above rolls this back too).
        aggregate_canonical_edges(cur, [m.entity_uid for m in mentions], graph_scope)

        # KG-AC-105: the plane of record converges instead of accumulating. A reprocessed folder
        # leaves canonical rows backed by nothing (its mentions were replaced wholesale by
        # extraction's partition replace), and pre-v16 those stayed forever — the "nothing ever
        # retracts" finding. Same transaction as everything above, so an abort restores them.
        retracted_edges, retracted_entities = retract_orphans(cur, graph_scope)

    return {"canonical_count": len(canonical_ids), "merged_count": merged, "minted_count": minted,
            "retracted_entity_count": retracted_entities, "retracted_edge_count": retracted_edges}
