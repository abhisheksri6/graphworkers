"""Profile runtime preview (KG-AC-20). Runs the SAME `core`/strategy engine as production against an
inline ``sample_text`` and returns the extracted entities + relations + usage[] — **store-only**: it
NEVER writes kg_entities/kg_edges (no partition_replace), no run_state, no M1–M4, no pipeline linkage.
Pure (strategies injected) — the Celery ``runtime_entity_extraction_task`` (in the worker module) wraps
this and POSTs to the store-only ``/entity-extraction/runtime-worker-results`` callback.

KG-AC-75 (v13): the preview is the operator's only single-document debugging surface, so it also
carries each relation's ``evidence_text``, the extracted ``facts[]``, per-item document/page
provenance (always null here — a synthetic preview chunk carries no real chunk_metadata, KG-AC-73's
own null-on-absence rule), and every KG-AC-74 drop counter (+ P7's ``guardrails_blocked_facts``) —
read straight off the SAME ``summary``/``ent_rows``/``edge_rows`` production already builds, never a
second, divergent computation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ontologies import load_pack
from strategies import Chunk, ExtractionConfig, run_pipeline

_RUNTIME_FOLDER = "__runtime__"


def run_preview(config_dict: Dict[str, Any], sample_text: Optional[str], *,
                spacy_model_path=None, llm_client=None) -> Dict[str, Any]:
    config = ExtractionConfig.from_dict(config_dict)
    pack = load_pack(config.ontology_pack)
    chunks = [Chunk(chunk_id="sample", text=sample_text or "")]
    ent_rows, edge_rows, summary, usage, _blocked = run_pipeline(
        chunks, config, pack, folder_id=_RUNTIME_FOLDER,
        spacy_model_path=spacy_model_path, llm_client=llm_client,
    )
    surface_by_uid = {e["entity_uid"]: e["surface_form"] for e in ent_rows}
    # KG-AC-75: flatten each entity's nested `attributes` (KG-AC-70) back into a top-level `facts[]`,
    # parallel to entities/relations, stamping the subject's type/surface since the nested attribute
    # dict itself carries only the property/value/evidence/provenance.
    facts = [
        {
            "subject_type": e["entity_type"], "subject_surface": e["surface_form"],
            "property": attr["property"], "value": attr["value"],
            "normalized_value": attr["normalized_value"], "evidence_text": attr["evidence"],
            "source_doc_id": attr["source_doc_id"], "page": attr["page"],
        }
        for e in ent_rows for attr in e.get("attributes", [])
    ]
    return {
        "status": "success",
        "entities": [
            {"surface_form": e["surface_form"], "entity_type": e["entity_type"],
             "span_start": e["span_start"], "span_end": e["span_end"],
             "confidence": e["confidence"],
             "source_doc_id": e["source_doc_id"], "page": e["page"]}  # KG-AC-75
            for e in ent_rows
        ],
        "relations": [
            {"relation_type": e["relation_type"],
             "src": surface_by_uid.get(e["src_entity_uid"]),
             "dst": surface_by_uid.get(e["dst_entity_uid"]),
             "evidence_text": e["evidence_text"],                     # KG-AC-75
             "source_doc_id": e["source_doc_id"], "page": e["page"]}  # KG-AC-75
            for e in edge_rows
        ],
        "facts": facts,  # KG-AC-75
        "usage": usage,
        "ontology_pack": summary["ontology_pack"],
        "ontology_version": summary["ontology_version"],
        "unmapped_type_count": summary["unmapped_type_count"],
        # KG-AC-75: every KG-AC-74 drop counter + P7's guardrails_blocked_facts
        "unmapped_property_count": summary["unmapped_property_count"],
        "unresolved_reference_count": summary["unresolved_reference_count"],
        "unlocatable_entity_count": summary["unlocatable_entity_count"],
        "ungrounded_fact_count": summary["ungrounded_fact_count"],
        "guardrails_blocked_facts": summary["guardrails_blocked_facts"],
    }
