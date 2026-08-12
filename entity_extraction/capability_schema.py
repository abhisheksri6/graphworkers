"""Canonical capability contract for the entity_extraction worker — schema v1.0.

Import-light (no Celery, settings, spaCy, or DB) so it can be imported by the worker, the
contract test, and the F0 seed-parity check without pulling a broker or the model. The worker
(``entity_extraction_worker.py``) re-exports it per the WORKER_NAMING requirement that every
profile-based worker declare a ``CAPABILITY_SCHEMA``.

**Seed-parity (KG-AC-21):** this dict MUST equal the ``system_capabilities.capability_schema``
row seeded by ``backend/migrations/20260802_02_kg_registries.sql`` (base contract) +
``20260806_01_kg_composable_layers.sql`` (spec v8 — the two ``rules_*_enabled`` fields) +
``20260807_01_kg_relation_strategy.sql`` (spec v10 L1 — ``relation_strategy``). The
migrations author the contract; this module is the worker-side mirror. If they diverge, the
parity test goes red.

Three-plane note (data-flow-planes.md): entity_extraction is an ARTIFACT-plane producer of the new
``entity_records`` type (ADR-0007) — it consumes ``text_chunks`` and writes ``kg_entities``/
``kg_edges`` (stage ``staged``); the callback carries only scalar state summaries (no bulk rows).
"""
from __future__ import annotations

from typing import Any, Dict

CAPABILITY_SCHEMA: Dict[str, Any] = {
    "worker": "entity_extraction",
    "version": "1.0",
    "input_fields": {
        "folder_id": {"type": "string", "implicit": True, "required": "always",
                      "description": "Package folder whose chunks are extracted"},
        "entity_extraction_config.engine": {"type": "string", "required": "always",
                                            "description": "Extraction engine: spacy | llm"},
        "entity_extraction_config.ontology_pack": {"type": "string", "required": "always",
                                                   "description": "Ontology pack name (e.g. generic, fibo_core)"},
        "entity_extraction_config.confidence_threshold": {"type": "number", "required": "optional",
                                                          "description": "Minimum confidence to keep a mention (0-1)"},
        "entity_extraction_config.promote_top_n": {"type": "integer", "required": "optional",
                                                   "description": "Top-N entities promoted to the state summary (default 10, max 20)"},
        "entity_extraction_config.entity_types": {"type": "array", "required": "optional",
                                                  "description": "Optional subset filter over the pack types — leave empty to extract every type in the selected ontology pack (KG-AC-57)"},
        "entity_extraction_config.connection_id": {"type": "string", "required": "optional",
                                                   "description": "LLM connection profile id; required when engine=llm"},
        "entity_extraction_config.coreference_enabled": {"type": "boolean", "required": "optional",
                                                         "description": "Resolve document-scoped anaphora (pronouns, definite noun phrases) before extraction (default false)"},
        "entity_extraction_config.rules_entities_enabled": {"type": "boolean", "required": "optional",
                                                            "description": "Opt-in deterministic EntityRuler regex/phrase entity layer, merged alongside engine (default false, KG-AC-54)"},
        "entity_extraction_config.rules_relations_enabled": {"type": "boolean", "required": "optional",
                                                             "description": "Opt-in deterministic DependencyMatcher relation layer, merged alongside engine (default false, KG-AC-55)"},
        "entity_extraction_config.relation_strategy": {"type": "string", "required": "optional",
                                                        "description": "Relation-extraction mode: generate (default, LLM open triple-generation) | classify (candidate-pair closed-set classification) | entity_scoped (one call per chunk over the merged entity list, no candidate-pair/same-sentence gate — KG-AC-65) — decoupled from engine (KG-AC-59)"},
        "entity_extraction_config.relation_self_consistency_k": {"type": "integer", "required": "optional",
                                                                  "description": "Self-consistency voting: run the active LLM relation mode k times and keep majority-vote relations (default 1 = off, bounded 1-5, KG-AC-67)"},
        "entity_extraction_config.llm_max_tokens": {"type": "integer", "required": "optional",
                                                    "description": "Bedrock Converse maxTokens override for the LLM tool-use call (default: the client's built-in 4096); raise for packs/documents whose extraction output legitimately needs more room (KG-AC-87)"},
    },
    "output_fields": {
        "folder_id": {"type": "string", "always_present": True},
        "entity_count": {"type": "integer", "always_present": True,
                         "description": "Entities written for the folder"},
        "edge_count": {"type": "integer", "always_present": True,
                       "description": "Edges written for the folder"},
        "distinct_types": {"type": "integer", "always_present": True,
                           "description": "Distinct entity types seen"},
        "top_entities": {"type": "array", "always_present": False,
                         "description": "Bounded top-N entities by mention count"},
        "ontology_pack": {"type": "string", "always_present": True},
        "ontology_version": {"type": "string", "always_present": True},
        "unmapped_type_count": {"type": "integer", "always_present": True,
                                "description": "LLM types dropped as out-of-vocab"},
        "ungrounded_relation_count": {"type": "integer", "always_present": True,
                                     "description": "LLM relations dropped because their evidence was not found verbatim in the source chunk (KG-AC-64)"},
        "self_consistency_votes": {"type": "integer", "always_present": True,
                                   "description": "Number of independent LLM relation-extraction runs made for the batch (1 when self-consistency voting is off, KG-AC-67)"},
        "chunk_metadata_missing_count": {"type": "integer", "always_present": True,
                                         "description": "Chunks whose chunk_metadata lacked a doc_id or page, so document/page provenance recorded null rather than a fabricated value (KG-AC-73)"},
        "unresolved_reference_count": {"type": "integer", "always_present": True,
                                       "description": "Relations whose src_id/dst_id did not resolve to an entity in the same LLM call's own response (KG-AC-71)"},
        "unlocatable_entity_count": {"type": "integer", "always_present": True,
                                     "description": "Non-abstract entities the model returned whose surface could not be located in the source chunk, dropped rather than written (KG-AC-72)"},
        "unmapped_property_count": {"type": "integer", "always_present": True,
                                    "description": "Facts dropped for an unknown property or a property whose subject type does not satisfy its declared domain (KG-AC-70)"},
        "ungrounded_fact_count": {"type": "integer", "always_present": True,
                                  "description": "Facts dropped because their evidence was not found verbatim in the source chunk (KG-AC-70)"},
        "guardrails_blocked_facts": {"type": "integer", "always_present": True,
                                    "description": "Facts dropped by the same once-per-batch guardrails screen entity candidates already pass through (KG-AC-84)"},
        "underivable_entity_count": {"type": "integer", "always_present": True,
                                     "description": "Abstract types with content but no resolvable identity value — skipped rather than minted under a fabricated key (KG-AC-92)"},
        "ambiguous_attachment_count": {"type": "integer", "always_present": True,
                                       "description": "Auto-relation attachments withheld because more than one hub of that type exists, so the pairing would have to be invented (KG-AC-91)"},
        "unanchored_fact_count": {"type": "integer", "always_present": True,
                                  "description": "Anchored facts dropped because their hub pairing is undeterminable — never left on an anchor that does not declare the property (KG-AC-93)"},
    },
    "artifact_inputs": {
        "chunks": {"required": "always",
                   "description": "Text chunks to extract entities and relations from",
                   "artifact_type": "text_chunks"},
    },
    "artifact_outputs": {
        "entities": {"stage": "staged",
                     "description": "Entity + edge rows written to kg_entities/kg_edges",
                     "artifact_type": "entity_records", "always_present": True},
    },
}
