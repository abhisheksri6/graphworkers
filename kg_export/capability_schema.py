"""Canonical capability contract for the kg_export worker — schema v1.0.

Import-light. **Seed-parity (KG-AC-4):** this dict MUST equal the
``system_capabilities.capability_schema`` row seeded by ``20260802_02_kg_registries.sql``.

Three-plane note: kg_export consumes ``entity_records`` (stage ``canonicalized``) and writes to an
EXTERNAL store (Neo4j) — no CB artifact_output; the callback carries only scalar counts.
"""
from __future__ import annotations

from typing import Any, Dict

CAPABILITY_SCHEMA: Dict[str, Any] = {
    "worker": "kg_export",
    "version": "1.0",
    "input_fields": {
        "folder_ids": {"type": "array", "implicit": True, "required": "always",
                       "description": "The hold-released job folder set"},
        "kg_export_config.connection_id": {"type": "string", "required": "always",
                                           "description": "knowledge_graph (Neo4j) connection profile id"},
        "kg_export_config.database": {"type": "string", "required": "optional",
                                      "description": "Neo4j database to write into (must already "
                                      "exist -- the worker never creates one); defaults to "
                                      "'neo4j' when left blank"},
    },
    "output_fields": {
        "node_count": {"type": "integer", "always_present": True,
                       "description": "Neo4j nodes written (one per canonical_id)"},
        "relationship_count": {"type": "integer", "always_present": True,
                               "description": "Neo4j relationships written (one per canonical edge)"},
    },
    "artifact_inputs": {
        "canonicalized_entities": {"stage": "canonicalized", "required": "always",
                                   "description": "Canonicalized entity_records to project into Neo4j",
                                   "artifact_type": "entity_records"},
    },
}
