# entity_canonicalization worker

Knowledge-graph **write path — canonicalization** (specs/knowledge-graph v2). A **single-instance**
node placed downstream of a `hold` barrier: it resolves duplicate entity mentions across the whole
hold-released batch to one canonical identity, **incrementally across runs** via the
`kg_canonical_entities` index, then transitions the batch `staged → canonicalized` in **one atomic
transaction** (ADR-0006).

- **Capability:** `entity_canonicalization` (variant-`None`; queue == task-stem == capability_type).
- **Pipeline:** normalize (lowercase, strip legal suffixes/punct → `normalized_form`) → block
  (own block + already-`canonicalized` rows sharing the block key) → **three-band match**
  (auto-accept exact-normalized · auto-reject below the fuzzy floor · ambiguous band →
  **LLM adjudication** — *the LEI-equal short-circuit is withdrawn at v11 with the gazetteer/
  external-id plane*) → cluster (union-find) → assign `canonical_id` (reuse an existing cluster on a
  match to a canonicalized row; else **mint via `INSERT … ON CONFLICT (canonical_key) DO NOTHING
  RETURNING`** — race-safe) → reconcile type (most-specific per the pack's `parent` hierarchy) →
  re-point edges. `canonical_key` = `type|normalized_form`.

## Layout
```
capability_schema.py   # import-light CAPABILITY_SCHEMA (F0 seed-parity, KG-AC-3)
core.py                # pure: normalize, canonical_key, 3-band match, union-find cluster, type reconcile
store.py               # read staged batch; resolve/mint against kg_canonical_entities; re-point edges;
                       #   staged->canonicalized — ALL in one transaction (KG-AC-40 atomic)
clients.py             # connection decrypt -> Bedrock (copied from entity_extraction); LLM adjudication
ontologies/            # vendored by-copy from entity_extraction (loader + generic/fibo_core packs) —
                       #   the parent hierarchy drives type reconciliation (KG-AC-23)
entity_canonicalization_worker.py  # single-instance task entrypoint
callback.py            # thin callback builders (canonical/merged/minted counts + usage[])
```

## Governance
Torch-free, no spaCy. Naming: WORKER_NAMING.md (variant-`None`: queue == capability_type). Contract
parity: `capability_schema.py` == the seeded `system_capabilities` row (KG-AC-3). The `ontologies/`
pack is vendored by-copy (the two worker repos can't import each other) — kept in step with
entity_extraction's packs.
