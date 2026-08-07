"""spaCy NER entity strategy (ADR-0009). Lazy, by-copy, offline model load (KG-AC-15) — spaCy is
imported and the model loaded only when ``extract`` first runs, never at module import (keeps the
module import-light and torch-free). spaCy labels are mapped to pack types via the pack's
``spacy_labels`` map; an unmapped label is simply not produced (it isn't a pack type). spaCy is a
NODE-only engine — it emits no relations.

For unit tests, an ``nlp`` object may be injected (a fake with ``__call__`` -> doc.ents) so the
label-mapping + span logic is testable without the ~600 MB model.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

from core import Candidate

from .base import Chunk, EntityStrategy, ExtractionConfig


def load_spacy_model(model_path: Optional[str] = None):
    """Lazy, offline, by-copy spaCy model load (KG-AC-15) — the single shared load path so a task
    loads the ~600MB by-copy model at most once, even when both SpacyNerStrategy (engine=spacy)
    and RulesRelationsStrategy (rules_relations_enabled, evolve v8) need it in the same pass."""
    import spacy  # lazy — CNN pipeline, torch-free
    path = model_path or os.environ.get("SPACY_MODEL_PATH")
    if not path:
        raise RuntimeError("SPACY_MODEL_PATH is not set — the by-copy en_core_web_lg model is required")
    return spacy.load(path)  # offline, by-copy (KG-AC-15) — never spacy.download


class SpacyNerStrategy(EntityStrategy):
    layer = "spacy"

    def __init__(self, model_path: Optional[str] = None, nlp: Any = None):
        self._model_path = model_path
        self._nlp = nlp  # injectable for tests

    def _load(self):
        if self._nlp is None:
            self._nlp = load_spacy_model(self._model_path)
        return self._nlp

    def extract(self, chunks: List[Chunk], config: ExtractionConfig, pack) -> List[Candidate]:
        nlp = self._load()
        out: List[Candidate] = []
        for ch in chunks:
            doc = nlp(ch.text)
            for ent in getattr(doc, "ents", []):
                mapped = pack.map_spacy_label(ent.label_)
                if mapped is None:
                    continue  # spaCy label not in this pack's vocabulary -> not produced
                out.append(Candidate(
                    surface_form=ent.text, entity_type=mapped, source_chunk_id=ch.chunk_id,
                    layer="spacy", span_start=ent.start_char, span_end=ent.end_char, confidence=1.0,
                ))
        return out
