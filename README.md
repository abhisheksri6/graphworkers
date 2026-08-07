# graphworkers

The three Knowledge Graph Celery worker repos from ContextBuilder's `knowledge-graph` capability,
previously on-disk-only with no git home:

- `entity_extraction/` — NER + relation extraction (deterministic layers + LLM generate/classify
  modes) into the Postgres staging tables.
- `entity_canonicalization/` — cross-document entity resolution / clustering into
  `kg_canonical_entities`.
- `kg_export/` — projects the canonicalized graph into Neo4j.

Spec-driven development; the full spec (`requirements.md`/`design.md`/`tasks.md`, acceptance
criteria, ADRs) lives in the `coding-governance` repo under `specs/knowledge-graph/`, not here.
Each worker's own `CLAUDE.md`/`WORKER_NAMING.md` is a generated copy from that governance repo's
`scripts/sync_governance.py` — edit the source there, not these copies directly.
