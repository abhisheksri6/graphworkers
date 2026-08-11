"""entity_extraction Celery worker — reads text_chunks, runs the layered ensemble, writes
kg_entities/kg_edges directly (one-transaction partition replace), POSTs the thin callback.

Standard worker pattern (read -> process via StorageClient -> callback), mirroring
classification_worker.py: the backend dispatches the WORKER_NAMING 5-arg signature
``(task_id, folder_id, entity_extraction_config, dag_id, run_id)`` post-commit (KG-AC-6). (The
design's "with_capability wrapper" phrasing is superseded — with_capability's (payload, storage, db)
signature can't provide the AC-mandated 5-arg dispatch; classification's direct get_worker_session +
StorageClient pattern is the working precedent. design.md updated to match.)

Runtime deps (ctx_worker_shared, celery, httpx) are imported here; the pure pipeline
(``core``/``strategies``/``callback``/``runtime``) is broker/DB-free and unit-tested separately.
The task BODY is ``process_folder`` (deps injected) so per-folder isolation, empty-folder fail-loud,
and the always-callback are testable without a broker.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import httpx
from celery import Celery
from celery.signals import celeryd_init
from kombu import Queue
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ctx_worker_shared import guardrails_client
from ctx_worker_shared.db_session import get_worker_session
from ctx_worker_shared.storage import StorageClient
from ctx_worker_shared.telemetry import log_event

import callback as cb
import runtime
import store
from capability_schema import CAPABILITY_SCHEMA  # re-exported per WORKER_NAMING
from clients import build_llm_client
from ontologies import load_pack
from strategies import ExtractionConfig, run_pipeline

__all__ = ["CAPABILITY_SCHEMA", "celery_app", "entity_extraction_task",
           "runtime_entity_extraction_task", "process_folder", "preflight_spacy_model"]

logger = logging.getLogger("entity_extraction_worker")


class EntityExtractionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    broker_url: str = Field(validation_alias=AliasChoices("ENTITY_EXTRACTION_BROKER_URL", "CELERY_BROKER_URL"))
    result_backend: str = Field(validation_alias=AliasChoices("ENTITY_EXTRACTION_RESULT_BACKEND", "CELERY_RESULT_BACKEND"))
    worker_results_url: str = Field(validation_alias=AliasChoices("WORKER_RESULTS_URL"))
    runtime_worker_results_url: str = Field(default="", validation_alias=AliasChoices("RUNTIME_WORKER_RESULTS_URL"))
    blob_storage_uri: str = Field(default="", validation_alias=AliasChoices("BLOB_STORAGE_URI"))
    spacy_model_path: str = Field(default="", validation_alias=AliasChoices("SPACY_MODEL_PATH"))
    guardrails_url: str = Field(default="", validation_alias=AliasChoices("GUARDRAILS_URL"))
    queues: str = Field(default="entity_extraction,runtime_entity_extraction",
                        validation_alias=AliasChoices("ENTITY_EXTRACTION_QUEUES", "CELERY_QUEUE"))


settings = EntityExtractionSettings()
celery_app = Celery("entity_extraction_worker", broker=settings.broker_url, backend=settings.result_backend)
_queue_names = [q.strip() for q in settings.queues.split(",") if q.strip()]
celery_app.conf.task_queues = tuple(Queue(name) for name in _queue_names)
celery_app.conf.task_default_queue = _queue_names[0]


def preflight_spacy_model(cfg: EntityExtractionSettings) -> None:
    """KG-AC-42: validate at boot that the configured by-copy spaCy model loads (a full offline
    ``spacy.load``, result discarded), failing loud BEFORE the worker accepts tasks. Gated on
    ``SPACY_MODEL_PATH`` being set — a worker not serving spaCy (unset → llm-only) skips it. The
    running model is still lazily instantiated by ``SpacyNerStrategy`` on first use; this only
    proves the asset is present so a missing by-copy model fails the deploy, not each folder."""
    path = cfg.spacy_model_path
    if not path:
        return  # llm-only worker — no spaCy asset required
    import spacy  # lazy — same offline load path as SpacyNerStrategy (KG-AC-15), never spacy.download
    try:
        spacy.load(path)
    except Exception as exc:  # noqa: BLE001 — boot gate: turn any load failure into a loud deploy abort
        raise RuntimeError(
            f"entity_extraction boot preflight failed: the by-copy spaCy model at "
            f"SPACY_MODEL_PATH={path!r} could not be loaded ({exc}). Provision the by-copy "
            f"en_core_web_lg model, or unset SPACY_MODEL_PATH for an llm-only worker (KG-AC-42)."
        ) from exc


@celeryd_init.connect
def _entity_extraction_boot_preflight(**_) -> None:
    """Run the spaCy boot preflight (KG-AC-42) in the main process, once, before forking — raising
    here aborts worker startup (``celeryd_init.send`` propagates receiver exceptions)."""
    preflight_spacy_model(settings)


def _screen_text(item) -> str:
    """The model-produced string guardrails must screen — `Candidate` carries `.surface_form`,
    `Fact` carries `.value` (KG-AC-84: facts are screened the same way entity candidates are, via
    this same function, in the SAME once-per-batch call)."""
    return item.surface_form if hasattr(item, "surface_form") else item.value


def _guardrails_screen(config: ExtractionConfig, run_id, folder_id) -> Optional[Callable]:
    """A once-per-batch guardrails screen (KG-AC-17/84) built from the sidecar, or None when unset.
    ``items`` is entity candidates AND facts concatenated (base._screen_guardrails's contract)."""
    policy = config.__dict__.get("guardrail_policy") if hasattr(config, "__dict__") else None
    if not settings.guardrails_url:
        return None

    def screen(items: List) -> List[bool]:
        texts = [_screen_text(item) for item in items]
        verdicts = guardrails_client.guardrails_check(
            texts=texts, policy=policy, direction="output",
            enforcement_point="entity_extraction", run_id=run_id, folder_id=folder_id,
            step_name="entity_extraction", logger=logger,
        )
        # guardrails_check returns per-text allow/deny; keep True = allowed. Fail-open on shape drift.
        allowed = getattr(verdicts, "allowed", None)
        if isinstance(allowed, list) and len(allowed) == len(items):
            return list(allowed)
        return [True] * len(items)

    return screen


def _post_callback(http_post: Callable, url: str, payload: Dict[str, Any]) -> None:
    try:
        resp = http_post(url, json=payload, timeout=30.0)
        if getattr(resp, "status_code", 200) >= 300:
            logger.error("entity_extraction callback HTTP %s: %s", resp.status_code, resp.text)
    except Exception as cb_err:  # noqa: BLE001
        logger.error("Failed to POST entity_extraction worker results: %s", cb_err)


def process_folder(task_id, folder_id, entity_extraction_config, dag_id, run_id, *,
                   storage, db, http_post, worker_results_url, spacy_model_path="",
                   guardrails_screen=None, llm_client=None) -> Dict[str, Any]:
    """Task body (deps injected). Reads chunks (empty -> loud fail, KG-AC-7), runs the pipeline,
    partition-replaces the folder's kg rows in the task transaction, and ALWAYS posts a callback.
    Per-folder isolation (KG-AC-19): any failure here fails only THIS folder's task."""
    status, error_message, summary, usage = "success", None, None, []
    try:
        chunks = store.read_chunks(storage, folder_id)
        if not chunks:
            raise RuntimeError(f"no chunks for folder_id={folder_id} — cannot extract an empty graph")
        config = ExtractionConfig.from_dict(entity_extraction_config)
        pack = load_pack(config.ontology_pack)
        if llm_client is None and config.engine == "llm":
            llm_client = build_llm_client(config.connection_id)
            if llm_client is None:
                raise RuntimeError("entity_extraction: engine=llm requires an LLM connection_id")
        if guardrails_screen is None:
            guardrails_screen = _guardrails_screen(config, run_id, folder_id)

        ent_rows, edge_rows, summary, usage, _blocked = run_pipeline(
            chunks, config, pack, folder_id=folder_id,
            spacy_model_path=spacy_model_path or settings.spacy_model_path,
            llm_client=llm_client, guardrails_screen=guardrails_screen,
        )
        store.partition_replace(db, folder_id, run_id, dag_id, task_id, ent_rows, edge_rows)
        cb.log_result_summary(logger, folder_id, len(chunks), summary)
    except Exception as exc:  # noqa: BLE001 — worker boundary: fail THIS folder loud, POST the error
        status, error_message, summary, usage = "failed", str(exc), None, []
        log_event(logger, logging.ERROR, "entity_extraction.failed", folder_id=folder_id, action="fail", exc=exc)

    payload = cb.build_callback(task_id, folder_id, dag_id, run_id, status=status,
                                summary=summary, usage=usage, error_message=error_message)
    _post_callback(http_post, worker_results_url, payload)
    return payload


@celery_app.task(name="entity_extraction_worker.entity_extraction_task")
def entity_extraction_task(task_id, folder_id, entity_extraction_config, dag_id, run_id):
    """Extract one folder's chunks -> kg_entities/kg_edges and POST the thin callback."""
    with get_worker_session() as db:
        storage = StorageClient(blob_uri=settings.blob_storage_uri, db=db)
        result = process_folder(
            task_id, folder_id, entity_extraction_config, dag_id, run_id,
            storage=storage, db=db, http_post=httpx.post,
            worker_results_url=settings.worker_results_url,
            spacy_model_path=settings.spacy_model_path,
        )
    return {"task_id": task_id, "status": result["status"], "folder_id": folder_id}


@celery_app.task(name="entity_extraction_worker.runtime_entity_extraction_task")
def runtime_entity_extraction_task(task_id, sample, entity_extraction_config):
    """Profile runtime preview (KG-AC-20): run the SAME engine on inline sample_text and POST the
    STORE-ONLY runtime callback (no run_state, no M1–M4, no kg rows written). ``sample`` = {"text": str}."""
    if not settings.runtime_worker_results_url:
        raise RuntimeError(
            "RUNTIME_WORKER_RESULTS_URL is not set — this worker cannot serve the "
            "runtime_entity_extraction queue. Set it (or stop subscribing to the queue)."
        )
    sample = sample or {}
    try:
        config = ExtractionConfig.from_dict(entity_extraction_config)
        llm_client = None
        if config.engine == "llm":
            llm_client = build_llm_client(config.connection_id)
        payload = runtime.run_preview(
            entity_extraction_config, sample.get("text"),
            spacy_model_path=settings.spacy_model_path, llm_client=llm_client,
        )
    except Exception as exc:  # noqa: BLE001 — worker boundary: fail loud, POST the error
        payload = {"status": "failed", "error_message": str(exc)}
        log_event(logger, logging.ERROR, "entity_extraction.runtime_failed", action="fail", exc=exc)

    payload["task_id"] = task_id
    _post_callback(httpx.post, settings.runtime_worker_results_url, payload)
    return {"task_id": task_id, "status": payload["status"]}
