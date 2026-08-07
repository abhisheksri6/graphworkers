"""entity_canonicalization Celery worker — single-instance node fed by a Hold barrier. Reads the
hold-released batch's staged entity_records, canonicalizes them in ONE transaction (KG-AC-40), and
POSTs a thin summary callback.

Standard worker pattern (get_worker_session + StorageClient; the shared wheel commits the session on
success / rolls back on failure — the atomicity guarantee for the batch). The task BODY is
``process_batch`` (deps injected) so the fail-loud + always-callback behavior is testable without a
broker. The AMBIGUOUS match band is adjudicated by an LLM via the connection profile (KG-AC-35) —
loud failure if an ambiguous pair arises with no connection configured.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx
from celery import Celery
from kombu import Queue
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ctx_worker_shared.db_session import get_worker_session
from ctx_worker_shared.storage import StorageClient
from ctx_worker_shared.telemetry import log_event

import callback as cb
import store
from capability_schema import CAPABILITY_SCHEMA  # re-exported per WORKER_NAMING
from clients import build_llm_client
from core import Mention
from ontologies import load_pack

__all__ = ["CAPABILITY_SCHEMA", "celery_app", "entity_canonicalization_task", "process_batch",
           "make_llm_adjudicator"]

logger = logging.getLogger("entity_canonicalization_worker")


class EntityCanonicalizationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    broker_url: str = Field(validation_alias=AliasChoices("ENTITY_CANONICALIZATION_BROKER_URL", "CELERY_BROKER_URL"))
    result_backend: str = Field(validation_alias=AliasChoices("ENTITY_CANONICALIZATION_RESULT_BACKEND", "CELERY_RESULT_BACKEND"))
    worker_results_url: str = Field(validation_alias=AliasChoices("WORKER_RESULTS_URL"))
    blob_storage_uri: str = Field(default="", validation_alias=AliasChoices("BLOB_STORAGE_URI"))
    queues: str = Field(default="entity_canonicalization",
                        validation_alias=AliasChoices("ENTITY_CANONICALIZATION_QUEUES", "CELERY_QUEUE"))


settings = EntityCanonicalizationSettings()
celery_app = Celery("entity_canonicalization_worker", broker=settings.broker_url, backend=settings.result_backend)
_queue_names = [q.strip() for q in settings.queues.split(",") if q.strip()]
celery_app.conf.task_queues = tuple(Queue(name) for name in _queue_names)
celery_app.conf.task_default_queue = _queue_names[0]


def make_llm_adjudicator(client) -> Callable[[Mention, Mention], bool]:
    """Return an adjudicate(a, b) -> bool for the AMBIGUOUS band (KG-AC-35). Each call goes through
    the LLM (temperature 0, deterministic gating) and is captured in client.usage[]."""
    def adjudicate(a: Mention, b: Mention) -> bool:
        prompt = (
            'Are these two mentions the SAME real-world entity? Answer with only "yes" or "no".\n'
            f'A: {a.entity_type} "{a.surface_form}"\n'
            f'B: {b.entity_type} "{b.surface_form}"'
        )
        return client.complete(prompt).strip().lower().startswith("y")
    return adjudicate


def _no_connection_adjudicator(a: Mention, b: Mention) -> bool:
    # An ambiguous pair was reached but no LLM connection is configured -> fail loud (KG-AC-35),
    # never a silent skip.
    raise RuntimeError(
        "entity_canonicalization: an ambiguous match pair requires an LLM connection for adjudication, "
        "but no connection_id is configured"
    )


def _post_callback(http_post: Callable, url: str, payload: Dict[str, Any]) -> None:
    try:
        resp = http_post(url, json=payload, timeout=30.0)
        if getattr(resp, "status_code", 200) >= 300:
            logger.error("entity_canonicalization callback HTTP %s: %s", resp.status_code, resp.text)
    except Exception as cb_err:  # noqa: BLE001
        logger.error("Failed to POST entity_canonicalization worker results: %s", cb_err)


def process_batch(task_id, folder_ids: Sequence[str], entity_canonicalization_config, dag_id, run_id,
                  *, db, http_post, worker_results_url, llm_client=None, adjudicate=None) -> Dict[str, Any]:
    """Task body (deps injected). Canonicalizes the hold-released batch atomically and ALWAYS posts a
    callback. Any failure fails the whole batch (the wrapper rolls back → all-staged, KG-AC-40)."""
    status, error_message, summary, usage = "success", None, None, []
    cfg = entity_canonicalization_config or {}
    try:
        fuzzy_floor = float(cfg.get("fuzzy_floor", 0.80))
        fuzzy_ceiling = float(cfg.get("fuzzy_ceiling", 0.95))
        connection_id = cfg.get("connection_id")
        if adjudicate is None:
            if llm_client is None and connection_id:
                llm_client = build_llm_client(connection_id)
            adjudicate = make_llm_adjudicator(llm_client) if llm_client is not None else _no_connection_adjudicator

        conn = db.connection().connection
        with conn.cursor() as cur:
            pack_name = store.batch_pack_name(cur, folder_ids) or "generic"
        pack = load_pack(pack_name)

        summary = store.canonicalize_batch(
            db, folder_ids, fuzzy_floor=fuzzy_floor, fuzzy_ceiling=fuzzy_ceiling,
            pack=pack, adjudicate=adjudicate,
        )
        usage = list(getattr(llm_client, "usage", []) or [])
        cb.log_result_summary(logger, len(folder_ids), summary)
    except Exception as exc:  # noqa: BLE001 — fail the whole batch loud, POST the error
        status, error_message, summary, usage = "failed", str(exc), None, []
        log_event(logger, logging.ERROR, "entity_canonicalization.failed", action="fail", exc=exc)

    payload = cb.build_callback(task_id, folder_ids, dag_id, run_id, status=status,
                                summary=summary, usage=usage, error_message=error_message)
    _post_callback(http_post, worker_results_url, payload)
    return payload


@celery_app.task(name="entity_canonicalization_worker.entity_canonicalization_task")
def entity_canonicalization_task(task_id, folder_ids, entity_canonicalization_config, dag_id, run_id):
    """Canonicalize the hold-released batch and POST the thin callback."""
    with get_worker_session() as db:
        _ = StorageClient(blob_uri=settings.blob_storage_uri, db=db)  # session-backed (shared bootstrap)
        result = process_batch(
            task_id, folder_ids or [], entity_canonicalization_config, dag_id, run_id,
            db=db, http_post=httpx.post, worker_results_url=settings.worker_results_url,
        )
    return {"task_id": task_id, "status": result["status"], "folder_count": len(folder_ids or [])}
