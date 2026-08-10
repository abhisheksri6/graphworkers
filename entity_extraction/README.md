# entity_extraction worker

Knowledge-graph **write path — extraction** (specs/knowledge-graph v11). Reads `text_chunks` for a
folder, runs a layered ensemble (regex › spaCy › LLM, ADR-0009 — the gazetteer tier is withdrawn at
v11) governed by a closed ontology pack (ADR-0008), and writes typed, provenance-stamped
`kg_entities`/`kg_edges` rows
(`stage='staged'`) to Postgres — then POSTs a thin scalar summary to the backend callback.

- **Capability:** `entity_extraction` (variant-`None`; `queue_name == capability_type == task-stem`
  per WORKER_NAMING.md). Task `entity_extraction_worker.entity_extraction_task` on queue
  `entity_extraction`; runtime preview on the aux queue `runtime_entity_extraction`.
- **Plane:** artifact-plane producer of the new `entity_records` type (ADR-0007). Per-folder fan-out
  (one task per `folder_id`); canonicalization + export run downstream of a `hold` barrier.

## Layout
```
capability_schema.py   # import-light CAPABILITY_SCHEMA (F0 seed-parity, KG-AC-21)
core.py                # pure: chunks + config + pack -> mentions + relations + bounded top-N (no Celery/DB/net)
strategies/            # base registry+precedence; rules_entities; spacy_ner; llm_ner; rules_relations
ontologies/            # pack schema + loader (DAG) + samples generic.json / fibo_core.json
clients.py             # connection-profile decrypt -> Bedrock (Nova + anthropic bodies); fail-loud
store.py               # read_chunks + one-transaction partition-replace (ON CONFLICT entity_uid/edge_uid)
callback.py            # pure callback builders + the shared canonical fixture (byte-identical to backend)
entity_extraction_worker.py  # with_capability task entrypoint
runtime.py             # runtime_entity_extraction aux task (store-only preview)
```

## Run (dev)
```bash
cd workers/entity_extraction
python -m venv venv && venv/Scripts/Activate.ps1
pip install -r requirements.txt          # TORCH-FREE (ADR-0009); en_core_web_lg is by-copy, not pip
celery -A entity_extraction_worker worker -Q entity_extraction,runtime_entity_extraction -l info
```

## By-copy assets (offline, no network — ADR-0008/0009)
- **spaCy model** `en_core_web_lg-3.7.1` → copy into `SPACY_MODEL_PATH` (default `./models/en_core_web_lg`).
  Loaded offline via `spacy.load(SPACY_MODEL_PATH)` — never downloaded at container start (KG-AC-15).

## Governance
Torch-free (`pip check` carries no `torch`; KG-AC-18). Naming: WORKER_NAMING.md (variant-`None`
invariant). Contract parity: `capability_schema.py` == the seeded `system_capabilities` row (KG-AC-21).
