"""kg_export Celery worker — single-instance node fed by a Hold barrier. Reads the canonicalized
Postgres graph and projects it into Neo4j (idempotent MERGE; rebuildable), then POSTs a thin summary
callback to the generic single-instance handler (/kg-export/worker-results, B12).

Standard worker pattern (get_worker_session + StorageClient). The task BODY is ``process_export``
(deps injected — DB session, Neo4j exporter, http_post) so it is testable with a MOCKED driver; the
live smoke needs a real Neo4j (kept out of the default suite).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Sequence

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
from clients import Neo4jConnectionError, Neo4jExporter
from core import NODE_LABEL, reconcile_statement, run_export
from ontologies import load_pack

__all__ = ["CAPABILITY_SCHEMA", "celery_app", "kg_export_task", "process_export"]

logger = logging.getLogger("kg_export_worker")


class KgExportSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    broker_url: str = Field(validation_alias=AliasChoices("KG_EXPORT_BROKER_URL", "CELERY_BROKER_URL"))
    result_backend: str = Field(validation_alias=AliasChoices("KG_EXPORT_RESULT_BACKEND", "CELERY_RESULT_BACKEND"))
    worker_results_url: str = Field(validation_alias=AliasChoices("WORKER_RESULTS_URL"))
    blob_storage_uri: str = Field(default="", validation_alias=AliasChoices("BLOB_STORAGE_URI"))
    queues: str = Field(default="kg_export", validation_alias=AliasChoices("KG_EXPORT_QUEUES", "CELERY_QUEUE"))


settings = KgExportSettings()
celery_app = Celery("kg_export_worker", broker=settings.broker_url, backend=settings.result_backend)
_queue_names = [q.strip() for q in settings.queues.split(",") if q.strip()]
celery_app.conf.task_queues = tuple(Queue(name) for name in _queue_names)
celery_app.conf.task_default_queue = _queue_names[0]


def _post_callback(http_post: Callable, url: str, payload: Dict[str, Any]) -> None:
    try:
        resp = http_post(url, json=payload, timeout=30.0)
        if getattr(resp, "status_code", 200) >= 300:
            logger.error("kg_export callback HTTP %s: %s", resp.status_code, resp.text)
    except Exception as cb_err:  # noqa: BLE001
        logger.error("Failed to POST kg_export worker results: %s", cb_err)


def assert_uniqueness_constraint(exp) -> None:
    """KG-AC-104: the target database must carry a uniqueness constraint on
    `:KgEntity(canonical_id)` before anything is written.

    Read-only (`SHOW CONSTRAINTS`) — the worker still issues NO schema DDL, preserving the
    least-privilege posture KG-AC-52 fixed for `CREATE DATABASE`: an export connection needs write
    rights, never DBMS-admin. Without the constraint, concurrent MERGEs on one canonical_id can
    race into duplicate nodes, which is precisely the defect this export exists not to create — so
    a missing constraint fails loud, pointing at the provisioning runbook, rather than producing a
    graph that looks fine until it doesn't."""
    rows = exp.execute("SHOW CONSTRAINTS YIELD labelsOrTypes, properties RETURN labelsOrTypes, properties", {})
    for row in rows or []:
        labels = (row.get("labelsOrTypes") or []) if isinstance(row, dict) else []
        props = (row.get("properties") or []) if isinstance(row, dict) else []
        if NODE_LABEL in labels and "canonical_id" in props:
            return
    raise Neo4jConnectionError(
        f"target database has no uniqueness constraint on :{NODE_LABEL}(canonical_id). It is "
        "created once at provisioning, not by this worker (least-privilege, KG-AC-52) — see "
        "specs/knowledge-graph/provisioning-runbook.md"
    )


def process_export(task_id, folder_ids: Sequence[str], kg_export_config, dag_id, run_id, *,
                   db, http_post, worker_results_url, exporter=None) -> Dict[str, Any]:
    """Task body (deps injected). Reads the canonical graph for the batch and MERGE-projects it into
    Neo4j, then ALWAYS posts a callback. ``exporter`` is a context manager exposing ``execute`` (a
    Neo4jExporter in production; a fake in tests)."""
    status, error_message, node_count, rel_count = "success", None, 0, 0
    cfg = kg_export_config or {}
    try:
        connection_id = cfg.get("connection_id")
        if not connection_id:
            raise Neo4jConnectionError("kg_export requires a knowledge_graph connection_id")

        conn = db.connection().connection
        with conn.cursor() as cur:
            # KG-AC-97/98: one export serves one scope. Derived from the rows about to be read, and
            # re-validated against the profile below — a drifted pairing must never write a
            # tenant's graph into another tenant's database.
            graph_scope = store.batch_graph_scope(cur, folder_ids)
            declared_scope = cfg.get("graph_scope")
            if graph_scope and declared_scope and graph_scope != declared_scope:
                raise Neo4jConnectionError(
                    f"kg_export: this batch's canonical rows belong to scope {graph_scope!r} but "
                    f"the node's profile declares {declared_scope!r} — refusing to export one "
                    "scope's graph into another's database"
                )
            # KG-AC-100's sibling lock class: same-scope exports serialize, so the reconcile below
            # can never act on a snapshot a concurrent export is still adding to. Transaction-scoped
            # (released on commit OR rollback); a DIFFERENT class from canonicalization's, so canon
            # and export of one scope may still overlap — export reads only `canonicalized` rows.
            if graph_scope:
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"kg_export:{graph_scope}",))

            # KG-AC-83/86: resolve the pack for ontology-qualified export -- an unresolvable pack
            # name (or none at all) is NOT fatal, per KG-AC-86's own "not an error" clause; the
            # export simply proceeds with bare names (pack=None).
            pack_name = store.batch_pack_name(cur, folder_ids)
            try:
                pack = load_pack(pack_name) if pack_name else None
            except Exception:  # noqa: BLE001 — an unloadable pack degrades to bare names, never fails export
                pack = None
            nodes, edges = store.read_canonical_graph(cur, folder_ids, pack=pack)
            keep_ids = store.scope_canonical_ids(cur, graph_scope) if graph_scope else []

        exp_cm = exporter if exporter is not None else Neo4jExporter(connection_id, database=cfg.get("database"))
        with exp_cm as exp:
            assert_uniqueness_constraint(exp)  # KG-AC-104 — read-only preflight, before any write
            summary = run_export(nodes, edges, exp.execute)
            # KG-AC-105 (export half): reconcile the database to the plane of record. Runs AFTER
            # the MERGEs so a node written by this very export is in `keep_ids` by construction.
            if graph_scope:
                cypher, params = reconcile_statement(keep_ids)
                exp.execute(cypher, params)
        node_count, rel_count = summary["node_count"], summary["relationship_count"]
        cb.log_result_summary(logger, len(folder_ids or []), node_count, rel_count)
    except Exception as exc:  # noqa: BLE001 — fail loud, POST the error
        status, error_message = "failed", str(exc)
        log_event(logger, logging.ERROR, "kg_export.failed", action="fail", exc=exc)

    payload = cb.build_callback(task_id, folder_ids or [], dag_id, run_id, status=status,
                                node_count=node_count, relationship_count=rel_count,
                                error_message=error_message)
    _post_callback(http_post, worker_results_url, payload)
    return payload


@celery_app.task(name="kg_export_worker.kg_export_task")
def kg_export_task(task_id, folder_ids, kg_export_config, dag_id, run_id):
    """Project the canonicalized batch into Neo4j and POST the thin callback."""
    with get_worker_session() as db:
        _ = StorageClient(blob_uri=settings.blob_storage_uri, db=db)
        result = process_export(
            task_id, folder_ids or [], kg_export_config, dag_id, run_id,
            db=db, http_post=httpx.post, worker_results_url=settings.worker_results_url,
        )
    return {"task_id": task_id, "status": result["status"], "folder_count": len(folder_ids or [])}
