"""Connection -> Neo4j driver for kg_export. Credentials come ONLY via the connection-profile decrypt
path (``expected_category='knowledge_graph'``), never ambient env. The driver + session are opened in
a context manager; ``execute(cypher, params)`` runs one statement. ``decrypt`` and ``driver_factory``
are injectable so the exporter is unit-testable without a live Neo4j (tests pass a fake driver).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import httpx


class Neo4jConnectionError(RuntimeError):
    """A hard Neo4j/connection failure — fail loud, never a silent no-op."""


def _config_value(*keys: str) -> str:
    for key in keys:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, val = line.partition("=")
            if name.strip() in keys and val.strip():
                return val.strip()
    return ""


def _decrypt_connection(profile_ref: str, expected_category: str) -> Tuple[str, Dict[str, Any]]:
    url = _config_value("PROFILE_DECRYPT_URL")
    if not url:
        raise Neo4jConnectionError(f"PROFILE_DECRYPT_URL is not set — cannot resolve connection '{profile_ref}'.")
    resp = httpx.post(url, json={"profile_name": profile_ref, "expected_category": expected_category}, timeout=30.0)
    if resp.status_code != 200:
        raise Neo4jConnectionError(
            f"decrypt connection '{profile_ref}' returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    data = resp.json() or {}
    return str(data.get("connection_type") or ""), (data.get("config") or {})


class Neo4jExporter:
    """Context-managed Neo4j writer. ``with Neo4jExporter(cid) as exp: run_export(..., exp.execute)``.

    ``database`` (KG-AC-52, evolve v7) is the kg_export profile's write-target Neo4j database —
    empty/None defaults to ``'neo4j'``. The database MUST already exist: this client never issues
    ``CREATE DATABASE`` (the export connection needs only write privilege, not DBMS-admin; an admin
    pre-creates databases, the profile maps one by name — owner decision 2026-08-05). A
    non-existent/unreachable database fails loud as ``Neo4jConnectionError``, never a silent no-op.
    """

    def __init__(self, connection_id: str, *, database: Optional[str] = None,
                 decrypt: Optional[Callable] = None, driver_factory: Optional[Callable] = None):
        if not connection_id:
            raise Neo4jConnectionError("kg_export requires a knowledge_graph connection_id")
        self.connection_id = connection_id
        self.database = database or "neo4j"
        self._decrypt = decrypt or _decrypt_connection
        self._driver_factory = driver_factory
        self._driver = None
        self._session = None

    def __enter__(self) -> "Neo4jExporter":
        ctype, cfg = self._decrypt(self.connection_id, expected_category="knowledge_graph")
        uri = cfg.get("uri") or cfg.get("url")
        user = cfg.get("username") or cfg.get("user")
        password = cfg.get("password")
        if not uri:
            raise Neo4jConnectionError(f"connection '{self.connection_id}' has no Neo4j uri")
        try:
            if self._driver_factory is not None:
                self._driver = self._driver_factory(uri, user, password)
            else:
                from neo4j import GraphDatabase  # lazy
                self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._session = self._driver.session(database=self.database)
        except Neo4jConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise Neo4jConnectionError(f"Neo4j connect failed for '{self.connection_id}': {exc}") from exc
        return self

    def execute(self, cypher: str, params: Dict[str, Any]) -> Any:
        """Run one statement and **fully consume** it, returning its rows as a list.

        Consumption is not a convenience — it is correctness. `session.run` opens an auto-commit
        transaction whose result streams lazily, so a statement whose result is never read can be
        discarded when the session closes: a `CREATE CONSTRAINT` issued this way was observed
        (2026-08-17) to report success and then not exist. The same hazard applies to the final
        MERGE of an export. Reading the result here also means a server-side failure is raised
        INSIDE this try, so it is wrapped as `Neo4jConnectionError` (KG-AC-52's fail-loud posture)
        rather than escaping later, outside the handler that exists to catch it."""
        try:
            return list(self._session.run(cypher, params))
        except Neo4jConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001 — includes "database does not exist" (KG-AC-52)
            raise Neo4jConnectionError(
                f"Neo4j write failed for '{self.connection_id}' (database={self.database!r}): {exc}"
            ) from exc

    def __exit__(self, *exc) -> None:
        try:
            if self._session is not None:
                self._session.close()
        finally:
            if self._driver is not None:
                self._driver.close()
