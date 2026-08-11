"""DB I/O for the entity_extraction worker.

Reads chunks via the shared ``StorageClient.read_chunks`` (KG-AC-7). Writes the folder's entities +
edges DIRECTLY to Postgres (KG-AC-8) as a **one-transaction partition replace** (KG-AC-10): on the
injected ``db`` session (committed atomically by the with_capability wrapper), DELETE this folder's
kg rows then bulk INSERT ``ON CONFLICT (entity_uid/edge_uid) DO UPDATE`` — so a re-run for the same
folder converges to an identical row set instead of accumulating. Every row carries full provenance
(KG-AC-11). Edges reference entities by both the stable ``*_entity_uid`` and the freshly-minted
serial ``*_entity_id`` (resolved from the RETURNING map).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from psycopg2.extras import execute_values

from strategies.base import Chunk


def read_chunks(storage, folder_id: str) -> List[Chunk]:
    """Load the folder's text_chunks via the shared StorageClient (KG-AC-7). Empty list means the
    folder has no chunks — the worker turns that into a loud failure (never an empty graph).
    KG-AC-73 (v13): `chunk_metadata` (already written by chunking, `{"page": N, "source":
    {"filename": ...}}`) is no longer discarded — `doc_id`/`page` are read from it, defaulting to
    None (never a fabricated value) when either is absent."""
    rows = storage.read_chunks(folder_id)
    chunks = []
    for r in rows:
        meta = r.get("chunk_metadata") or {}
        doc_id = (meta.get("source") or {}).get("filename")
        page = meta.get("page")
        chunks.append(Chunk(chunk_id=r["chunk_id"], text=(r.get("content") or ""),
                            doc_id=doc_id, page=page))
    return chunks


def partition_replace(db, folder_id: str, run_id: Any, dag_id: Any, source_task_id: Any,
                      entity_rows: List[Dict], edge_rows: List[Dict]) -> Tuple[int, int]:
    """One-transaction partition replace for a folder (KG-AC-10). Returns (entity_count, edge_count).
    Runs on the raw psycopg2 connection behind the injected SQLAlchemy session, so it shares the
    task's transaction (the wrapper commits on success / rolls back on failure)."""
    conn = db.connection().connection  # raw psycopg2 conn, same transaction as the session
    with conn.cursor() as cur:
        # DELETE the folder partition — edges first (FK), then entities.
        cur.execute("DELETE FROM public.kg_edges WHERE folder_id = %s", (folder_id,))
        cur.execute("DELETE FROM public.kg_entities WHERE folder_id = %s", (folder_id,))

        uid_to_id: Dict[str, int] = {}
        if entity_rows:
            ent_values = [
                (
                    folder_id, run_id, dag_id, source_task_id,
                    e["entity_uid"], e["entity_type"], e["surface_form"], e.get("source_chunk_id"),
                    e.get("source_doc_id"), e.get("page"), e.get("is_abstract", False),
                    e.get("span_start"), e.get("span_end"), e.get("occurrence_idx", 0),
                    e.get("confidence", 1.0),
                    e.get("extractor"), e.get("ontology_pack"), e.get("ontology_version"),
                    e.get("model_id"), e.get("stage", "staged"),
                )
                for e in entity_rows
            ]
            returned = execute_values(cur, """
                INSERT INTO public.kg_entities
                  (folder_id, run_id, dag_id, source_task_id, entity_uid, entity_type, surface_form,
                   source_chunk_id, source_doc_id, page, is_abstract, span_start, span_end, occurrence_idx, confidence,
                   extractor, ontology_pack, ontology_version, model_id, stage)
                VALUES %s
                ON CONFLICT (entity_uid) DO UPDATE SET
                  entity_type=EXCLUDED.entity_type, surface_form=EXCLUDED.surface_form,
                  run_id=EXCLUDED.run_id, dag_id=EXCLUDED.dag_id, source_task_id=EXCLUDED.source_task_id,
                  source_doc_id=EXCLUDED.source_doc_id, page=EXCLUDED.page, is_abstract=EXCLUDED.is_abstract,
                  confidence=EXCLUDED.confidence, extractor=EXCLUDED.extractor,
                  ontology_pack=EXCLUDED.ontology_pack, ontology_version=EXCLUDED.ontology_version,
                  model_id=EXCLUDED.model_id, stage=EXCLUDED.stage
                RETURNING id, entity_uid
            """, ent_values, page_size=500, fetch=True)
            uid_to_id = {row[1]: row[0] for row in returned}

        edge_count = 0
        valid_edges = [
            e for e in edge_rows
            if e["src_entity_uid"] in uid_to_id and e["dst_entity_uid"] in uid_to_id
        ]
        if valid_edges:
            edge_values = [
                (
                    folder_id, run_id, dag_id, source_task_id,
                    e["edge_uid"], e["relation_type"],
                    uid_to_id[e["src_entity_uid"]], uid_to_id[e["dst_entity_uid"]],
                    e["src_entity_uid"], e["dst_entity_uid"], e.get("confidence", 1.0),
                    e.get("evidence_text"), e.get("source_doc_id"), e.get("page"),
                )
                for e in valid_edges
            ]
            execute_values(cur, """
                INSERT INTO public.kg_edges
                  (folder_id, run_id, dag_id, source_task_id, edge_uid, relation_type,
                   src_entity_id, dst_entity_id, src_entity_uid, dst_entity_uid, confidence,
                   evidence_text, source_doc_id, page)
                VALUES %s
                ON CONFLICT (edge_uid) DO UPDATE SET
                  relation_type=EXCLUDED.relation_type, src_entity_id=EXCLUDED.src_entity_id,
                  dst_entity_id=EXCLUDED.dst_entity_id, run_id=EXCLUDED.run_id,
                  dag_id=EXCLUDED.dag_id, confidence=EXCLUDED.confidence,
                  evidence_text=EXCLUDED.evidence_text, source_doc_id=EXCLUDED.source_doc_id,
                  page=EXCLUDED.page
            """, edge_values, page_size=500)
            edge_count = len(valid_edges)

    return len(entity_rows), edge_count
