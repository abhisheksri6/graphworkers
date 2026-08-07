"""Pure callback builder for kg_export — thin job-scoped summary (node_count / relationship_count +
usage[]); consumed by the generic B12 KG single-instance worker-results handler."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence


def build_callback(task_id: str, folder_ids: Sequence[str], dag_id: Optional[str], run_id: Optional[str],
                   *, status: str, node_count: Optional[int] = None,
                   relationship_count: Optional[int] = None, error_message: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task_id": task_id, "status": status, "process_name": dag_id or "", "job_id": run_id,
        "folder_ids": list(folder_ids),
    }
    if error_message is not None:
        payload["error_message"] = error_message
    if status == "success":
        payload["node_count"] = node_count
        payload["relationship_count"] = relationship_count
        payload["usage"] = []
    return payload


def log_result_summary(logger: logging.Logger, folder_count: int, node_count: int, rel_count: int) -> None:
    logger.info("kg_export.done folders=%s nodes=%s relationships=%s", folder_count, node_count, rel_count)
