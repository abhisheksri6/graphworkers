# kg_export worker

Knowledge-graph **write path — export** (specs/knowledge-graph v2). A **single-instance** node
downstream of a `hold` barrier: reads the **canonicalized** Postgres graph and projects it into
**Neo4j** via the `knowledge_graph` connection. Neo4j is a **rebuildable projection** (ADR-0006) —
Postgres stays the plane of record; dropping + re-exporting reproduces it exactly.

- One Neo4j **node per `canonical_id`** (labelled by the reconciled pack type; properties carry the
  normalized form, external_id/LEI, and provenance). One **relationship per canonical edge**
  (mention-edges collapse to the canonical endpoints). **Idempotent `MERGE`** (KG-AC-29) →
  rebuildable (KG-AC-28) → round-trip parity with the Postgres canonical graph (KG-AC-27).
- Callback → the generic `/kg-export/worker-results` (the B12 single-instance handler; FinOps +
  step-reconcile).

## Layout
```
capability_schema.py   # import-light CAPABILITY_SCHEMA (F0 seed-parity, KG-AC-4)
core.py                # pure: build idempotent MERGE Cypher (nodes + relationships) from the graph
store.py               # read the canonical graph (nodes per canonical_id + collapsed edges) from Postgres
clients.py             # Neo4j driver via the knowledge_graph connection decrypt (injected for tests)
kg_export_worker.py    # single-instance task entrypoint
callback.py            # thin callback (node_count / relationship_count + usage[])
```

## Governance
Torch-free, no spaCy/LLM. Naming: WORKER_NAMING.md (variant-`None`: queue == capability_type).
Contract parity: `capability_schema.py` == the seeded row (KG-AC-4). Tests mock the Neo4j driver
(round-trip / rebuildable / idempotent); the **live smoke** (`@pytest.mark.live`) needs a real Neo4j
connection profile and is skipped by default.
