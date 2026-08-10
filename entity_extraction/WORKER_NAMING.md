<!-- SYNCED COPY — DO NOT EDIT HERE.
     Source of truth: coding-governance/WORKER_NAMING.md
     Edit the source and re-run coding-governance/scripts/sync_governance.py. -->

# Worker Naming Standard

This document is the authoritative governance reference for worker naming in ContextBuilder.
All engineers adding workers, queues, or capabilities must follow these rules.
The rules are enforced by database constraints — violations fail at INSERT/UPDATE time,
not silently at dispatch time.

---

## The one rule everything derives from

> **One identifier per thing. One table owns it. Worker file name, Celery queue,
> and Celery task name are all derived from that identifier by a fixed convention.**

---

## Two node categories, two sources of truth

### Profile-based nodes

These are nodes where the user selects a specific implementation at design time
(e.g., which OCR engine, which chunking strategy, which vector store).

| Field | Value |
|---|---|
| **Canonical identifier** | `system_capabilities.queue_name` |
| **Celery queue dispatched to** | `queue_name` (same value) |
| **Worker module file** | `{queue_name}_worker.py` |
| **Celery task name** | `{queue_name}_worker.{queue_name}_task` |
| **Stored in DB** | `system_capabilities.task_name` |

Domains: `parser`, `chunking`, `vectorize`, `ingest`, `export`, `textprocessing`, `classification`, `extraction`, `package_validation`, `entity_extraction`, `entity_canonicalization`, `kg_export`

PR 1-5 additions to this list:
- `ingest` gained variants `source_csv`, `source_api`, `source_text`, `source_api_trigger` (the api_trigger node dispatches to `source_api_trigger`)
- `parser` gained `structured_reader` (reads `.json/.xml/.csv` files, emits `structured_record`)
- `extraction` (new capability type) ships variant `llm_extraction` in v1
- `package_validation` (new capability type) — single-implementation guard node

Knowledge-graph additions (spec v2, 2026-08-02) — three new **variant-`None`** capability types
(`_PROFILE_VARIANT_FIELD[type] = None`), so all five derivations collapse onto the `capability_type`
per the single-implementation invariant below (`entity_extraction` ≠ the older `extraction` capability):
- `entity_extraction` — per-folder; layered regex › spaCy › LLM ensemble (ADR-0009 — the gazetteer tier is withdrawn at knowledge-graph spec v11)
- `entity_canonicalization` — single-instance, placed downstream of the `hold` barrier; cross-run entity resolution
- `kg_export` — single-instance, downstream of `hold`; Neo4j projection (ADR-0006)

  For each `{cap}` above: worker file `{cap}_worker.py` · Celery app `Celery("{cap}_worker", …)` ·
  task `@task(name="{cap}_worker.{cap}_task")` · queue `{cap}` · `system_capabilities.task_name`
  = `{cap}_worker.{cap}_task`. (`entity_extraction` also ships an aux runtime queue
  `runtime_entity_extraction` for the store-only preview — the sole sanctioned `runtime_` use.)

> **Single-implementation invariant (variant-`None`): `queue_name` MUST equal `capability_type`.**
> A capability with no user-selectable variant (`_PROFILE_VARIANT_FIELD[type] = None` — e.g.
> `classification`, `textprocessing`, `assembler`, `entity_extraction`, `entity_canonicalization`,
> `kg_export`) is dispatched by
> `_resolve_worker` with `variant = capability_type`, and `get_capability_dispatch` matches
> `queue_name == variant`. So the row's `queue_name` (and thus worker file / Celery app / task stem)
> must be the `capability_type` itself. A divergent name (e.g. `entity_extractor` for
> `capability_type='entity_extraction'`) misses the lookup, falls to `_default_task_queue`, and the
> task is sent to a phantom queue no worker consumes — a **silent hang, not an error**. (Variant-*based*
> capabilities like `parser`/`chunking` are the opposite: `queue_name` = the variant value, e.g.
> `docling`.)

### Inline nodes

These are nodes with no user-selectable variant — the implementation is fixed.
Multiple node types may share one worker module when they share runtime dependencies.

| Field | Value |
|---|---|
| **Canonical identifier** | `node_type_registry.type_key` |
| **Celery queue** | worker module's domain name (e.g., `transform`, `llm`, `integration`) |
| **Worker module file** | `{domain}_worker.py` |
| **Celery task name** | `{domain}_worker.{type_key}_task` |
| **Stored in DB** | `node_type_registry.queue_resolver` (static JSON) |

Domains: `llm`, `integration`, `transform`, `script`, `agent`, `format_adapter`

PR 2-5 additions to this list:
- `format_adapter` (renamed from `contract_adapter` in PR 2) — single inline node, conversion picked from `format_adapter_conversions` matrix
- `package_validation` — inline guard node (also gets a `capability_type='package_validation'` row for dispatch symmetry)
- `api_trigger` — inline ingest-class entry node; dispatches to `source_worker.source_api_trigger_task`

---

## Naming rules

### `queue_name` / `type_key` (the identifier)

- Format: `^[a-z][a-z0-9_]*$` — lowercase, ASCII, snake_case, starts with a letter
- Enforced by: `CHECK` constraint `queue_name_canonical` on `system_capabilities`;
  `CHECK` constraint `capability_type_canonical` also applies to `capability_type`
- Examples: `tesseract`, `fixed_characters`, `aws_embedding`, `semantic_chunking`
- **Never**: `FixedCharacters`, `aws-embedding`, `fixedcharacters` (no trailing plurals or camelCase)

### Worker file name

```
{queue_name}_worker.py           # one variant per file (profile-based)
{domain}_worker.py               # one file for a multi-type domain (inline)
```

### Celery app name (inside the file)

```python
celery_app = Celery("{queue_name}_worker", ...)   # must match the file stem
```

### Celery task decorator

```python
@celery_app.task(name="{queue_name}_worker.{queue_name}_task")   # profile-based
@celery_app.task(name="{domain}_worker.{type_key}_task")         # inline shared
```

### Celery queue (the `-Q` flag / `task_default_queue`)

```
queue_name         # from system_capabilities — exactly this string, no suffix
```

The `runtime_{queue_name}` suffix seen in some legacy startup scripts is **not**
part of the standard. New workers must not add it unless the `runtime_dispatch`
feature is explicitly implemented and documented.

---

## `capability_type` in `system_capabilities`

Maps to `node_type_registry.type_key`:

| capability_type | Meaning |
|---|---|
| `parser` | Document parsing / OCR engines |
| `chunking` | Text chunking strategies |
| `vectorize` | Embedding / vectorization providers |
| `ingest` | File import sources |
| `export` | Vector store export targets |
| `textprocessing` | Text cleaning filters (single variant) |
| `classification` | Document classification (single variant) |
| `assembler` | Decomposed-parsing branch-content merge (single variant; specs/parsing v10, P4.2) |
| `entity_extraction` | Named-entity + relation extraction (single variant; layered rules/spaCy/LLM ensemble; specs/knowledge-graph v2) |
| `entity_canonicalization` | Cross-run entity resolution → canonical ids (single variant; downstream of `hold`; specs/knowledge-graph v2) |
| `kg_export` | Neo4j knowledge-graph projection (single variant; downstream of `hold`; specs/knowledge-graph v2) |

Rules:
- Always lowercase snake_case (constraint enforced)
- Must match the corresponding `type_key` in `node_type_registry`
- **Never** use `Import`, `Embedding`, or any PascalCase form

---

## `CAPABILITY_SCHEMA` — mandatory for every profile-based worker

Every profile-based worker **must** declare a `CAPABILITY_SCHEMA` dict at module level.
This dict is the canonical description of the worker's accepted input fields and produced
output fields. It is stored in `system_capabilities.capability_schema` and consumed by:

- The frontend inspector (GraphInspector / FieldMapper) to render per-node field configuration
- The backend `_validate_field_presence` check before dispatching a task
- The `validate_capability_input()` helper inside the worker for runtime contract checking

### Minimal shape

```python
CAPABILITY_SCHEMA: Dict[str, Any] = {
    "worker": "my_parser",          # matches queue_name
    "version": "2.0",
    "input_fields": {
        "folder_id":   {"type": "string",  "required": "always",   "implicit": True,
                        "description": "DataBackbone folder containing the source PDF"},
        "ParserName": {"type": "string",  "required": "always",   "queue_routing": True,
                        "description": "Must be my_parser — set from profile"},
    },
    "output_fields": {
        "folder_id":          {"type": "string",  "always_present": True},
        "parser_output_count":{"type": "integer", "always_present": True,
                               "description": "Number of parser_element rows written"},
    },
}
```

### Field flags

| Flag | Meaning |
|---|---|
| `"required": "always"` | Worker fails if field absent |
| `"required": "optional"` | Has a default or can be omitted |
| `"implicit": true` | Backend injects this; never shown to user |
| `"queue_routing": true` | Routing discriminator; never user-configurable |
| `"always_present": true` | Always in callback payload (output only) |

### Seed file

After writing the worker, add an `UPDATE system_capabilities SET capability_schema = '...'::jsonb WHERE queue_name = 'my_parser'` statement to `backend/db/seeds/reference_data.sql`.

---

## Adding a new profile-based capability

A profile-based capability is a new selectable variant within an existing domain
(e.g., a new OCR engine, a new chunking strategy, a new vector store).
It requires three things: a worker file, a DB row, and an update to the seed.
**No backend service code changes are required.**

### Step 1 — Choose the identifier

Pick a `queue_name` that:
- Describes the implementation, not the domain (`paddle`, not `ocr_paddle`)
- Is unique within its `capability_type`
- Passes `^[a-z][a-z0-9_]*$` — the constraint rejects it otherwise

Everything else is derived from this single string.

```
queue_name          = "my_parser"
worker file         = workers/parser/my_parser_worker.py
Celery app name     = "my_parser_worker"
Celery task name    = "my_parser_worker.my_parser_task"
Celery queue        = "my_parser"
task_name in DB     = "my_parser_worker.my_parser_task"
```

### Step 2 — Write the worker file

Place it in the correct domain directory. Use the existing workers as a template.
The mandatory elements are:

```python
# workers/parser/my_parser_worker.py

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from celery import Celery
from kombu import Queue

class MyParserSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    broker_url: str = Field(
        validation_alias=AliasChoices("PARSER_BROKER_URL", "CELERY_BROKER_URL")
    )
    result_backend: str = Field(
        validation_alias=AliasChoices("PARSER_RESULT_BACKEND", "CELERY_RESULT_BACKEND")
    )
    data_backbone_dir: str = Field(validation_alias=AliasChoices("DATA_BACKBONE_DIR"))
    worker_results_url: str = Field(validation_alias=AliasChoices("WORKER_RESULTS_URL"))
    # Default queue must equal queue_name chosen in Step 1
    parser_queues: str = Field(
        default="my_parser",
        validation_alias=AliasChoices("PARSER_QUEUES", "CELERY_QUEUE"),
    )

settings = MyParserSettings()

# Celery app name = worker file stem
celery_app = Celery("my_parser_worker", broker=settings.broker_url, backend=settings.result_backend)
queue_names = [q.strip() for q in settings.parser_queues.split(",") if q.strip()]
celery_app.conf.task_queues = tuple(Queue(name) for name in queue_names)
celery_app.conf.task_default_queue = queue_names[0]


# Task name = "{worker_file_stem}.{queue_name}_task"
@celery_app.task(name="my_parser_worker.my_parser_task")
def my_parser_task(task_id, folder_id, parser_config, dag_id, run_id):
    """Parse documents using My Parser and POST results to the callback URL."""
    # ... implementation ...
    pass
```

Key rules:
- The `Celery(...)` first argument, the `@task(name=...)`, and the function name
  must all use the same `queue_name` stem — no mixing
- The default in the `parser_queues` field must equal `queue_name`
- The standard callback args are `(task_id, folder_id, config, dag_id, run_id)` —
  match this signature so `_build_celery_args` works without modification

### Step 3 — Register in `system_capabilities`

Create a migration file in `backend/migrations/` and run it:

```python
# backend/migrations/add_my_parser_capability.py
import psycopg2, os

conn = psycopg2.connect(
    os.getenv("DATABASE_URL", "postgresql://postgres:postgres@10.73.88.101:5432/RND")
    .replace("postgresql+asyncpg://", "postgresql://")
)
with conn.cursor() as cur:
    cur.execute("""
        INSERT INTO public.system_capabilities
            (capability_type, queue_name, task_name, display_name,
             description, is_enabled, is_default, runtime_enabled, licensing)
        VALUES
            ('parser', 'my_parser', 'my_parser_worker.my_parser_task', 'My Parser',
             '{"Speed": "Fast", "Purpose": "..."}',
             true, false, true, 'Open Source')
        ON CONFLICT (capability_type, queue_name) DO NOTHING
    """)
conn.commit()
conn.close()
```

Run it:
```bash
cd backend && venv/Scripts/python.exe migrations/add_my_parser_capability.py
```

The `CHECK` constraints will reject any `queue_name` or `capability_type` that
violates the format — the INSERT fails immediately, not at dispatch time.

### Step 4 — Update the seed file

Add the same row to `backend/db/seeds/reference_data.sql` so fresh installs have
the correct data. Follow the existing INSERT format in that file exactly, including
the `task_name` column.

### Step 5 — Verify

Confirm the frontend palette API now includes the new variant:
```bash
curl http://localhost:8000/pipeline/node-types | python -m json.tool | grep my_parser
```

Dispatch a test pipeline step with `ParserName: "my_parser"`. `_resolve_worker`
will look up the row and return `("my_parser_worker.my_parser_task", "my_parser")`.

**That's all.** No changes to `template_service.py`, no new maps, no aliases.

---

## Adding a new inline node type

An inline node has a fixed implementation — the user does not select a variant.
The two paths differ only in whether the node fits an existing worker domain.

### Path A — Add a task to an existing worker (most common)

Use this when the new node type shares the same runtime dependencies as an
existing domain. For example, a new data transformation belongs in `transform_worker`.

> **Sanctioned case — an inline entry node hosted in a profile-based module.**
> `api_trigger` (Pipeline Triggers spec v1) is an inline node type whose hydrate
> task `folder_worker.api_trigger_task` lives in the **profile-based** ingest
> module `folder_worker.py`, subscribed on its own `api_trigger` queue
> (`INGEST_QUEUES=folder,api_trigger`). This is allowed because it shares the
> ingest runtime deps (StorageClient/fsspec + the metadata libraries) and MUST
> emit a callback shape-identical to `folder_task`. The task/queue name is the
> node `type_key` (`api_trigger`), not the module's `queue_name`.

**Step 1 — Add the task function to the existing worker file:**

```python
# workers/transform_worker/transform_worker.py  (existing file)

@celery_app.task(name="transform_worker.my_transform_task")
def my_transform_task(task_id, folder_id, cfg, dag_id, run_id):
    """Execute my transformation and POST results to the callback URL."""
    # ... implementation ...
    pass
```

Task name rule: `{domain_worker_stem}.{type_key}_task`

**Step 2 — Register in `node_type_registry` via a migration:**

```sql
-- backend/migrations/add_my_transform_node_type.sql
INSERT INTO public.node_type_registry (
    type_key, display_name, category, description,
    input_ports, output_ports, config_schema,
    queue_resolver,
    gateway_compat, is_active, updated_at
)
VALUES (
    'my_transform',
    'My Transform',
    'Transform',
    'Describe what this node does.',
    '[{"name":"in","contract_key":"any","contract_version":1,"label":"In","required":true}]'::jsonb,
    '[{"name":"out","contract_key":"any","contract_version":1,"label":"Out","required":false}]'::jsonb,
    '{}'::jsonb,
    '{"task": "transform_worker.my_transform_task", "queue": "transform"}'::jsonb,
    'worker', true, NOW()
)
ON CONFLICT (type_key) DO UPDATE
SET queue_resolver = EXCLUDED.queue_resolver,
    updated_at     = NOW();
```

The `queue_resolver` JSONB column is the declarative record of the task name and queue.
It drives the frontend palette and is the source of truth for documentation.
Dispatch itself is handled by `_INLINE_WORKER_MAP` in `template_service.py` (Step 3 below).

**Step 3 — Add the type_key to the inline map in `_resolve_worker`:**

```python
# backend/services/template_service.py  →  _resolve_worker  →  _INLINE_WORKER_MAP

_INLINE_WORKER_MAP: Dict[str, Tuple[str, str]] = {
    ...
    "my_transform": ("transform_worker.my_transform_task", "transform"),  # ← add this
}
```

**Step 4 — Update the seed file**

Add the same row to `backend/db/seeds/reference_data.sql` so fresh installs have
the correct data. Follow the existing `node_type_registry` INSERT format in that file.

This is the complete changeset for a new inline node on an existing worker.

---

### Path B — Create a new worker domain (new capability area)

Use this when the new node type has its own runtime dependencies and cannot share
a process with any existing worker.

**Step 1 — Decide the domain name.**

The domain name becomes the queue name and the worker file stem.

```
domain              = "ocr_validator"
worker file         = workers/ocr_validator/ocr_validator_worker.py
Celery app name     = "ocr_validator_worker"
Celery task name    = "ocr_validator_worker.ocr_validate_task"   ← type_key is "ocr_validate"
Celery queue        = "ocr_validator"
```

**Step 2 — Create the worker directory and file:**

```
workers/
└── ocr_validator/
    ├── ocr_validator_worker.py     ← worker implementation
    └── config.py                   ← Pydantic settings (optional; some workers inline settings)
```

Minimal worker structure:

```python
# workers/ocr_validator/ocr_validator_worker.py
import os, logging
from celery import Celery
from kombu import Queue
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class OcrValidatorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    broker_url: str     = Field(validation_alias=AliasChoices("OCR_VALIDATOR_BROKER_URL", "CELERY_BROKER_URL"))
    result_backend: str = Field(validation_alias=AliasChoices("OCR_VALIDATOR_RESULT_BACKEND", "CELERY_RESULT_BACKEND"))
    data_backbone_dir: str = Field(validation_alias=AliasChoices("DATA_BACKBONE_DIR"))
    worker_results_url: str = Field(validation_alias=AliasChoices("WORKER_RESULTS_URL"))
    queues: str = Field(default="ocr_validator", validation_alias=AliasChoices("OCR_VALIDATOR_QUEUES", "CELERY_QUEUE"))

settings = OcrValidatorSettings()
logger = logging.getLogger("ocr_validator_worker")

celery_app = Celery("ocr_validator_worker", broker=settings.broker_url, backend=settings.result_backend)
queue_names = [q.strip() for q in settings.queues.split(",") if q.strip()]
celery_app.conf.task_queues = tuple(Queue(name) for name in queue_names)
celery_app.conf.task_default_queue = queue_names[0]


@celery_app.task(name="ocr_validator_worker.ocr_validate_task")
def ocr_validate_task(task_id, folder_id, cfg, dag_id, run_id):
    """Validate OCR output quality and POST results to the callback URL."""
    # ... implementation ...
    pass
```

**Step 3 — Register in `node_type_registry`** (same as Path A Step 2, with
the new domain's task name and queue):

```sql
queue_resolver = '{"task": "ocr_validator_worker.ocr_validate_task", "queue": "ocr_validator"}'
```

**Step 4 — Add the type_key to `_INLINE_WORKER_MAP` in `_resolve_worker`:**

```python
"ocr_validate": ("ocr_validator_worker.ocr_validate_task", "ocr_validator"),
```

**Step 5 — Create a `.env` file** for local development in the new worker directory:

```bash
# workers/ocr_validator/.env
CELERY_BROKER_URL=pyamqp://guest:guest@10.73.88.101:5672//
CELERY_RESULT_BACKEND=rpc://
DATA_BACKBONE_DIR=\\10.73.88.101\DataBackBone
WORKER_RESULTS_URL=http://10.73.88.101:8000/pipeline/worker-results
```

**Step 6 — Update the seed file**

Add the `node_type_registry` row to `backend/db/seeds/reference_data.sql`, following
the existing INSERT format in that file.

---

## Adding a new profile-based capability type (new domain)

This is rare. Use it only when adding an entirely new stage to the pipeline
(e.g., a new `"ocr_post_processing"` domain), not when adding a new variant
within an existing domain like `parser` or `chunking`.

In addition to all five steps under "Adding a new profile-based capability", you
must also:

**Backend: register the new `capability_type` in `_PROFILE_VARIANT_FIELD`**

```python
# backend/services/template_service.py  →  _PROFILE_VARIANT_FIELD

_PROFILE_VARIANT_FIELD: Dict[str, Optional[str]] = {
    ...
    "ocr_post_processing": "OcrPostProcessor",   # ← add this; value is the profile_config field name
}
```

This is the only backend code change required. Without it, `_resolve_worker`
will not recognise the new capability type and will fall through to the generic
convention fallback.

Also add the new `capability_type` to the `capability_type` table in this document.

---

## Special note: chunking domain

The chunking stage has a second dispatch path in `backend/services/chunk_service.py`.
`chunk_service._resolve_chunking_task` uses its own small mapping dict with a
convention-based fallback:

```python
worker_name = f"{strategy}_worker.{strategy}_task"
```

A new chunking variant that follows the naming convention (Step 1 of the profile-based
guide above) will be picked up by this fallback automatically — no code change to
`chunk_service.py` is required unless the strategy name requires normalization that
the fallback cannot handle. In that case, add an entry to both the
`_normalize_strategy_name` method and the `mapping` dict inside `_resolve_chunking_task`.

---

## Renaming an existing capability (the right way)

A rename is a **migration**, not a dictionary entry.

1. `UPDATE system_capabilities SET queue_name = 'new_name' WHERE queue_name = 'old_name'`
2. Rename the worker file: `old_name_worker.py` → `new_name_worker.py`
3. Update the task decorator inside: `name="new_name_worker.new_name_task"`
4. Update the Celery app name inside: `Celery("new_name_worker", ...)`
5. Update `task_name` column in the same migration (it will be auto-derived if you
   follow the convention, or set explicitly)

**Never** handle a rename by adding a new alias key to a mapping dictionary.
The constraint on `queue_name` blocks you from inserting duplicates anyway.

---

## Invariants enforced by the database

| Constraint | Column | Pattern |
|---|---|---|
| `queue_name_canonical` | `system_capabilities.queue_name` | `^[a-z][a-z0-9_]*$` |
| `capability_type_canonical` | `system_capabilities.capability_type` | `^[a-z][a-z0-9_]*$` |
| `NOT NULL` | `system_capabilities.task_name` | (cannot be empty) |
| `UNIQUE` | `(capability_type, queue_name)` | (no duplicate variants) |

---

## Worker file organisation

```
workers/
├── {domain}/                        ← one directory per domain
│   ├── {variant}_worker.py          ← one file per variant (profile-based)
│   │   OR
│   ├── {domain}_worker.py           ← one file for shared-runtime domain (inline)
│   └── config.py                    ← Pydantic settings; reads queue from env
```

**Decision rule for file structure:**
> If variants have different system/Python dependencies (e.g., different OCR
> engines, different cloud SDKs), use one file per variant.  
> If variants share the same dependencies and can run in the same process
> (e.g., transform operations), use one domain file with multiple task functions.

A file = a Docker image you'd build and deploy. If you'd build them together, they
can be in the same file.

---

## What aliases are for (and are not for)

**Pydantic `AliasChoices` in config files** — these are valid and intentional.
They allow environment variables to use different names across deployment contexts
(e.g., `CHUNKING_BROKER_URL` or `CELERY_BROKER_URL`). These are env-var aliases,
not queue/task aliases, and are not covered by this standard.

**Queue and task string aliases** — these are forbidden. If you feel the urge to
add a mapping like `"fixedcharacters": "fixed_characters"`, it means two systems
disagree on the canonical name. Fix the disagreement at the source; do not add
a bridge.
