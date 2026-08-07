<!-- GENERATED FILE — DO NOT EDIT BY HAND.
     Assembled by coding-governance/scripts/sync_governance.py from
     governance pieces: base.md, worker.md.
     Edit those source pieces in coding-governance/governance/, then re-run the script. -->

# ContextBuilder — Shared Governance

This is the shared baseline every ContextBuilder repo inherits. It is assembled into
each repo's `CLAUDE.md` by the governance sync script. Do not edit a generated
`CLAUDE.md` directly — edit the source piece `coding-governance/governance/base.md`
and re-run `scripts/sync_governance.py`.

## Project Overview

ContextBuilder is an unstructured data harmonization platform that governs *what data reaches AI models* — turning raw, scattered enterprise content into precise, structured, reusable context. The result: lower inference cost, higher answer accuracy.

Main codebases (each its own git repo):
- `backend/` — FastAPI orchestrator: routes, services, repositories, MCP server, search API generation
- `workers/*` — Celery task processors.
- `guardrails-service/` — FastAPI sidecar enforcing input/output/agentic guardrail checks
- `designer-demo/` — disposable unit-testing UI (React + Vite); NOT version-controlled. Do NOT explore, search, or modify it unless explicitly asked.

## Capability Specs (keep current)

`specs/<capability>/` (governance repo) is the per-capability home, spec-driven: `requirements.md`
(business / end-user behavior — the capability catalog), `design.md` (technical design), `tasks.md`
(status + target version). **Mandatory:** before changing a functional flow, first read that
capability's `specs/<capability>/requirements.md` to confirm intended behavior; and whenever you add
or change a functionality, update it in the **same change** (for a new capability, add a
`specs/<capability>/` folder + link it in `specs/README.md`). Cross-cutting standards live in
`governance/standards/` and system-wide architecture in `architecture/` (both **in the governance
repo**, read on demand — not copied into component repos); single-component technical docs stay in
`backend/docs/`.

**Spec-first for non-trivial changes:** start from `specs/SPEC_TEMPLATE.md` in the governance
repo (Full/Lite tier rules inside); acceptance criteria — deterministic + probabilistic, each with an AC ID — exist **before**
implementation, and changing accepted behavior means editing the spec first and re-freezing.
Cross-cutting locked decisions get an ADR in `adr/` in the same change.

## Architecture — Three Planes (how data flows)

Data moves across three independent planes — never conflate them (full reference:
`architecture/data-flow-planes.md` in the governance repo):

1. **Execution-order plane** — DAG edges; sequence only, carry no data.
2. **State plane** — small scalars keyed by `(run_id, folder_id, step_name)` in `run_state` (written only by the orchestrator from worker callbacks; audit copy in `task_details.reserved`); read via `state.<step>.<field>`. (`pipeline_entity_variables` is a different store — designer-declared per-entity variables.)
3. **Artifact plane** — bulk content keyed by `folder_id` (== `package_id`); bindings in `capability_schema.artifact_inputs` / `artifact_outputs`.

`artifact_type` is the closed shape vocabulary (`file_collection`, `text_elements`, `text_chunks`,
`vector_embeddings`, `structured_record`, `record_collection`, `raw_text`, `source_code`); shape
conversions are first-class `format_adapter` nodes — never intra-worker. Graphs are **acyclic**
(static Airflow DAGs) — loops are forbidden.

## Configuration

- `os.environ` first (Prod), then `.env` fallback (Dev). Backend reads `settings.<name>` from `backend/config.py`; workers read their own env.
- No scattered `os.getenv()` in app code; all secrets/URLs from env / `.env` (never hardcoded).
- Never log credentials, tokens, API keys, or encryption material.

## Python Conventions

- Python **3.10**. `snake_case` files/functions/vars, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- Absolute imports from project root; group imports stdlib → third-party → local.
- Type hints on new/changed signatures. Keep changes minimal — no unrelated refactors or formatting churn.

## Pydantic

Pinned to **2.9.2** (a `CoreSchemaOrFieldType` bug in 2.10+ breaks the project — do not upgrade).
Use `AliasChoices` for flexible field names, `field_validator(..., mode="before")` for coercion.

## Error Logging

- `logger.exception(...)` for unexpected failures; `logger.warning(...)` for expected/recoverable issues.
- `# noqa: BLE001` where broad exception handling is intentional at API boundaries.

## Engineering Principles

- **Think before coding** — don't assume or hide confusion; state assumptions, surface tradeoffs, ask when unclear; present multiple interpretations rather than silently picking one; push back when warranted.
- **Simplicity first** — minimum code that solves the problem; nothing speculative, no abstractions for single-use code, no unrequested flexibility/config, no error handling for impossible cases. If 200 lines could be 50, rewrite.
- **Fallbacks & hardcoded values — get approval first** — no fallback/default shims, mock or sample data, static catalogs mirroring a backend source, or magic constants standing in for real data, without explicit approval. They hide drift and mask failures. Default to failing loudly; if one seems warranted, STOP and ask (explain why + exactly what it contains). Applies when editing existing code too.
- **Surgical changes** — touch only what the task requires; don't refactor, reformat, or "improve" adjacent code; match existing style; remove only orphans your change created; mention (don't delete) pre-existing dead code. Every changed line traces to the request.
- **Goal-driven execution** — turn tasks into verifiable goals (e.g. write the failing test, then make it pass; ensure tests pass before and after a refactor). For multi-step work, state a brief plan with a verify step for each.

## Workers — Pattern, Naming & Schema (worker repos)

### Worker Pattern
Each worker: reads input (`folder_id` + config) from the Celery task payload → processes →
writes artifacts to DB (parser_element, chunk, embedding rows) via `StorageClient` → POSTs a
thin result summary to the backend callback endpoint. Workers declare their accepted input and
output fields in a `CAPABILITY_SCHEMA` dict at the top of the worker file; this schema is stored
in `system_capabilities.capability_schema` and used by the frontend inspector and backend
field-presence validator.

### Worker Naming Standard
The full governance reference is `WORKER_NAMING.md` **in this repo** (adding workers, renaming,
migrations, the seed-file update). The rules below are mandatory; deviations get rejected by DB
CHECK constraints.

**One identifier per thing — file, queue, and task name are all derived from it:**

- **Profile-based nodes** (parser, chunking, vectorize, ingest, export, textprocessing, classification):
  identifier is `system_capabilities.queue_name`. The frontend dropdown values come from this column
  unchanged — never alias them.
- **Inline nodes** (llm, integration, transform, script, agent, format_adapter): identifier is
  `node_type_registry.type_key`. Multiple type_keys may share one worker module when they share runtime dependencies.

**Derivation (no exceptions):**
```
queue_name / domain  →  {name}_worker.py        (file)
                     →  Celery("{name}_worker") (app name = file stem)
                     →  {name}_worker.{name}_task        (profile-based task)
                        {domain}_worker.{type_key}_task  (inline shared task)
                     →  queue = queue_name (exactly, no suffix)
```

**Format**: `^[a-z][a-z0-9_]*$` — lowercase ASCII snake_case. Enforced by CHECK constraints
`queue_name_canonical` and `capability_type_canonical`.

**Forbidden:**
- Queue/task string aliases. If two systems disagree on a name, fix the source — do not bridge.
- `runtime_{queue_name}` as a worker's **primary (pipeline-node) queue identity** — a worker's
  file/app/task/queue derive from the bare `queue_name`. (The **auxiliary** `runtime_*` queues used
  by Profile Runtime Testing below are the sanctioned exception — subscribed *in addition to* the
  primary queue, never replacing it.)
- Non-snake_case `capability_type` values (`Import`, `Embedding`, PascalCase) — use `ingest`, `vectorize`, etc.

### Profile Runtime Testing (auxiliary `runtime_*` queues)

Profile-based workers can offer **profile runtime testing**: run ONE profile against a sample input,
off to the side of any pipeline, and show the result directly in the UI. This is **distinct from
Quick Run** (which test-runs a whole pipeline graph through the normal node queues).

**Applies to (profile types):** `parser` (per-variant: `runtime_{parser_key}`, e.g.
`runtime_docling`), `textprocessing` (`runtime_textprocessing`), `chunking`
(`runtime_{base_queue}`), and `classification` (`runtime_classification` — spec'd v6, pending
freeze). Not offered for `ingest`, `vectorize`, `export` (no meaningful single-profile preview).

**The flow (one pattern, all workers):**
1. **Dispatch** — `POST /{domain}/runtime-*` on the backend: validate the profile/config, create a
   `task_details` row (+ `reserved` payload), then the **backend assigns** the Celery task to the
   worker's **auxiliary runtime queue** (`runtime_{queue_name}`). Tasks are always backend-assigned;
   there is no separate runtime dispatch path beyond the queue name.
2. **Worker** — the SAME worker process subscribes to the auxiliary queue in addition to its primary
   queue (its `*_QUEUES` env, e.g. `TEXTPROCESSING_QUEUES=textprocessing,runtime_textprocessing`)
   and runs the SAME engine as production — runtime tests must never grow a divergent code path.
3. **Result** — the worker reports back via callback; the result lands on the `task_details` row.
   Runtime callbacks are **store-only**: no `run_state` write, no M1–M4 state merge, no pipeline
   linkage (there is no run to merge into).
4. **Poll** — `GET /{domain}/runtime_status/{task_id}` returns the detailed result (pending /
   result / failed+error), which the UI shows directly.

**Endpoint naming standard (record 2026-07-09 — owner: development must be consistent):** per
domain, the runtime-testing surface is named as a family, parallel to the production callback:
- `POST /{domain}/runtime-{action}` — dispatch (e.g. `runtime-parse`, `runtime-classify`)
- `POST /{domain}/runtime-worker-results` — the **store-only** runtime callback (parallel to the
  production `/{domain}/worker-results`; never the production handler)
- `GET /{domain}/runtime_status/{task_id}` — poll (underscore matches the existing siblings)

*Legacy note:* parser/textprocessing/chunking predate this standard — they dispatch the
**production task** with sentinel run identities (`"RuntimeTesting"`) and their runtime results
ride the production `/{domain}/worker-results` handler (through the state-merge path). New
runtime-testing surfaces MUST use the store-only `runtime-worker-results` pattern
(classification is the reference); migrating the legacy three is tracked in the
architectural-observations log.

**Rules:** subscribing the worker to its `runtime_*` queue is a deployment requirement (an
unsubscribed queue makes runtime tests hang silently — check `.env`/`.env.example`); external
calls made during a runtime test (LLM/SaaS) are real and FinOps-captured like production; and the
runtime callback must be reachable by an unsigned worker — `/{domain}/runtime-worker-results` is
covered by the backend auth allowlist (`core/security.py` `_WORKER_CALLBACK_SUFFIXES`, same
transitional status as `/worker-results`; found live 2026-07-10 as a 401). A new worker-facing
endpoint that doesn't match those suffixes needs an allowlist entry (or go under `/internal/`
HMAC) **in the same change**.

### Dispatch source of truth
`template_service._resolve_worker` (backend) reads `system_capabilities` for profile-based nodes
via `repository.get_capability_dispatch(...)`. Inline nodes use `_INLINE_WORKER_MAP`. The only
other dispatch path is `chunk_service._resolve_chunking_task`, whose convention-based fallback
picks up new chunking variants automatically. Do not add other dispatch paths.
