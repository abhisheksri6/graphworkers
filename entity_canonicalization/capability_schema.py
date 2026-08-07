"""Canonical capability contract for the entity_canonicalization worker — schema v1.0.

Import-light (no Celery/DB/LLM) so it imports for the contract + F0 seed-parity checks without a
broker. **Seed-parity (KG-AC-3):** this dict MUST equal the
``system_capabilities.capability_schema`` row seeded by ``20260802_02_kg_registries.sql``.

Three-plane note: canonicalization is a single-instance ARTIFACT-plane transition — it consumes the
hold-released batch's ``entity_records`` (stage ``staged``) and produces ``entity_records`` (stage
``canonicalized``) with ``canonical_id`` assigned; the callback carries only scalar counts.
"""
from __future__ import annotations

from typing import Any, Dict

CAPABILITY_SCHEMA: Dict[str, Any] = {
    "worker": "entity_canonicalization",
    "version": "1.0",
    "input_fields": {
        "folder_ids": {"type": "array", "implicit": True, "required": "always",
                       "description": "The hold-released job folder set (single-instance, whole batch)"},
        "entity_canonicalization_config.fuzzy_floor": {"type": "number", "required": "optional",
                                                       "description": "Below this fuzzy score = auto-reject"},
        "entity_canonicalization_config.fuzzy_ceiling": {"type": "number", "required": "optional",
                                                         "description": "Above this = auto-accept; between floor and ceiling = LLM adjudication"},
        "entity_canonicalization_config.connection_id": {"type": "string", "required": "optional",
                                                         "description": "LLM connection for ambiguous-band adjudication (required if adjudication reachable)"},
    },
    "output_fields": {
        "canonical_count": {"type": "integer", "always_present": True,
                            "description": "Distinct canonical entities after resolution"},
        "merged_count": {"type": "integer", "always_present": True,
                         "description": "Mentions collapsed into existing canonical ids"},
        "minted_count": {"type": "integer", "always_present": True,
                         "description": "Novel canonical entities minted"},
    },
    "artifact_inputs": {
        "staged_entities": {"stage": "staged", "required": "always",
                            "description": "Staged entity_records from the hold-released batch",
                            "artifact_type": "entity_records"},
    },
    "artifact_outputs": {
        "canonicalized_entities": {"stage": "canonicalized",
                                   "description": "entity_records transitioned staged->canonicalized with canonical_id",
                                   "artifact_type": "entity_records", "always_present": True},
    },
}
