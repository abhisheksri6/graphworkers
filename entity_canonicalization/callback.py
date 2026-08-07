"""Pure callback builders for the entity_canonicalization worker — thin job-scoped summary
(canonical/merged/minted counts + usage[]); no bulk rows. Single-instance node, so the callback is
keyed by the job's folder set, not per-folder."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

_SCALARS = ("canonical_count", "merged_count", "minted_count")


def build_callback(task_id: str, folder_ids: Sequence[str], dag_id: Optional[str], run_id: Optional[str],
                   *, status: str, summary: Optional[Dict[str, Any]] = None,
                   usage: Optional[List[Dict[str, Any]]] = None,
                   error_message: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task_id": task_id, "status": status, "process_name": dag_id or "", "job_id": run_id,
        "folder_ids": list(folder_ids),
    }
    if error_message is not None:
        payload["error_message"] = error_message
    if summary is not None:
        for k in _SCALARS:
            payload[k] = summary.get(k)
        payload["usage"] = usage or []
    return payload


def log_result_summary(logger: logging.Logger, folder_count: int, summary: Dict[str, Any]) -> None:
    """Counts only — never entity surfaces or document text."""
    logger.info(
        "entity_canonicalization.done folders=%s canonical=%s merged=%s minted=%s",
        folder_count, (summary or {}).get("canonical_count"),
        (summary or {}).get("merged_count"), (summary or {}).get("minted_count"),
    )
