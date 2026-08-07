"""Profile runtime preview (KG-AC-20). Runs the SAME `core`/strategy engine as production against an
inline ``sample_text`` and returns the extracted entities + relations + usage[] — **store-only**: it
NEVER writes kg_entities/kg_edges (no partition_replace), no run_state, no M1–M4, no pipeline linkage.
Pure (strategies injected) — the Celery ``runtime_entity_extraction_task`` (in the worker module) wraps
this and POSTs to the store-only ``/entity-extraction/runtime-worker-results`` callback.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ontologies import load_pack
from strategies import Chunk, ExtractionConfig, run_pipeline

_RUNTIME_FOLDER = "__runtime__"


def run_preview(config_dict: Dict[str, Any], sample_text: Optional[str], *,
                gazetteer=None, spacy_model_path=None, llm_client=None) -> Dict[str, Any]:
    config = ExtractionConfig.from_dict(config_dict)
    pack = load_pack(config.ontology_pack)
    chunks = [Chunk(chunk_id="sample", text=sample_text or "")]
    ent_rows, edge_rows, summary, usage, _blocked = run_pipeline(
        chunks, config, pack, folder_id=_RUNTIME_FOLDER,
        gazetteer=gazetteer, spacy_model_path=spacy_model_path, llm_client=llm_client,
    )
    surface_by_uid = {e["entity_uid"]: e["surface_form"] for e in ent_rows}
    return {
        "status": "success",
        "entities": [
            {"surface_form": e["surface_form"], "entity_type": e["entity_type"],
             "span_start": e["span_start"], "span_end": e["span_end"],
             "external_id": e.get("external_id"), "confidence": e["confidence"]}
            for e in ent_rows
        ],
        "relations": [
            {"relation_type": e["relation_type"],
             "src": surface_by_uid.get(e["src_entity_uid"]),
             "dst": surface_by_uid.get(e["dst_entity_uid"])}
            for e in edge_rows
        ],
        "usage": usage,
        "ontology_pack": summary["ontology_pack"],
        "ontology_version": summary["ontology_version"],
        "unmapped_type_count": summary["unmapped_type_count"],
    }
