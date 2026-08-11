"""Pure callback builders for the entity_extraction worker — no ctx_worker_shared, Celery, or network,
so the callback contract tests without the wheel.

The callback is THIN (KG-AC-8): it carries only the scalar summary (counts + bounded top-N + ontology
stamps) and usage[] (llm calls only) — NEVER the bulk kg_entities/kg_edges rows, which the worker
writes directly to Postgres. On success it carries ``folders[]`` — the per-folder state the
orchestrator merges into run_state (M1, KG-AC-9). A canonical success payload is committed at
``tests/fixtures/callback_success.json`` and asserted byte-identical here AND by the backend M1–M4
handler test (B4) — the one shared fixture that keeps the two sides in lockstep.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# The KG-AC-9 scalar state summary promoted per folder (order-stable for the shared fixture).
# *Amended v11 — `linked_count` is dropped with the gazetteer capability.*
# *Amended v12 — `ungrounded_relation_count` (KG-AC-64) and `self_consistency_votes` (KG-AC-67) added.*
# *Amended v13 — `chunk_metadata_missing_count` (KG-AC-73) added: chunks whose chunk_metadata lacked
# a doc_id or page, so provenance recorded null rather than a fabricated value. `unresolved_
# reference_count` (KG-AC-71) added: relations whose src_id/dst_id did not resolve to an entity in
# the same LLM call's own response. `unlocatable_entity_count` (KG-AC-72) added: non-abstract
# entities dropped for having no locatable span. `unmapped_property_count`/`ungrounded_fact_count`
# (KG-AC-70) added: facts dropped for an unknown property/invalid domain, or for ungrounded
# evidence, respectively.*
_STATE_SCALARS = (
    "entity_count", "edge_count", "distinct_types", "top_entities",
    "ontology_pack", "ontology_version", "unmapped_type_count", "ungrounded_relation_count",
    "self_consistency_votes", "chunk_metadata_missing_count", "unresolved_reference_count",
    "unlocatable_entity_count", "unmapped_property_count", "ungrounded_fact_count",
)


def flatten_state(folder_id: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """The per-folder STATE dict the orchestrator merges into run_state (state.<step>.<field>)."""
    state: Dict[str, Any] = {"folder_id": folder_id}
    for field in _STATE_SCALARS:
        state[field] = summary.get(field)
    return state


def build_callback(task_id: str, folder_id: str, dag_id: Optional[str], run_id: Optional[str],
                   *, status: str, summary: Optional[Dict[str, Any]] = None,
                   usage: Optional[List[Dict[str, Any]]] = None,
                   error_message: Optional[str] = None) -> Dict[str, Any]:
    """Thin orchestration callback (same envelope as the classification/parser workers). Success
    carries the scalar summary + usage[] + folders[]; NO entity/edge lists (KG-AC-8)."""
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "status": status,
        "process_name": dag_id or "",
        "job_id": run_id,
        "folder_id": folder_id,
    }
    if error_message is not None:
        payload["error_message"] = error_message
    if summary is not None:
        for field in _STATE_SCALARS:
            payload[field] = summary.get(field)
        payload["usage"] = usage or []
        payload["folders"] = [flatten_state(folder_id, summary)]
    return payload


def log_result_summary(logger: logging.Logger, folder_id: str, chunk_count: int,
                       summary: Dict[str, Any]) -> None:
    """Log counts ONLY — never document text or entity surfaces (KG-AC-17: no content in logs)."""
    logger.info(
        "entity_extraction.done folder_id=%s chunks=%s entities=%s edges=%s distinct_types=%s "
        "unmapped_types=%s pack=%s/%s",
        folder_id, chunk_count,
        (summary or {}).get("entity_count"), (summary or {}).get("edge_count"),
        (summary or {}).get("distinct_types"), (summary or {}).get("unmapped_type_count"),
        (summary or {}).get("ontology_pack"), (summary or {}).get("ontology_version"),
    )
