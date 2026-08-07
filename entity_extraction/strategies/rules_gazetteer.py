"""Opt-in GLEIF/LEI gazetteer entity strategy (ADR-0009, KG-AC-13). Highest-precedence layer. The
gazetteer INDEX is injected (C6 provides a by-copy SQLite-backed lookup with ``find(text) ->
List[GazetteerHit]``; unit tests inject a fake). When ``entity_linking`` is on and the hit's pack
type may carry an external id, the generalized LEI + source ``gleif_lei`` are written on the row
(feeding cross-run canonicalization). Opt-in: default off; enabled-but-no-index is a non-fatal no-op.
Only CC0/open sources ship — no CUSIP/ISIN.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from core import Candidate

from .base import Chunk, EntityStrategy, ExtractionConfig


@dataclass
class GazetteerHit:
    surface: str
    start: int
    end: int
    entity_type: str
    lei: Optional[str] = None


class RulesGazetteerStrategy(EntityStrategy):
    layer = "gazetteer"

    def __init__(self, gazetteer: Any = None):
        self._gaz = gazetteer  # object with .find(text) -> List[GazetteerHit]

    def extract(self, chunks: List[Chunk], config: ExtractionConfig, pack) -> List[Candidate]:
        if self._gaz is None:
            return []  # opt-in; enabled but index unavailable -> no gazetteer layer (non-fatal)
        out: List[Candidate] = []
        for ch in chunks:
            for hit in self._gaz.find(ch.text):
                if not pack.is_known_type(hit.entity_type):
                    continue
                cand = Candidate(
                    surface_form=hit.surface, entity_type=hit.entity_type, source_chunk_id=ch.chunk_id,
                    layer="gazetteer", span_start=hit.start, span_end=hit.end, confidence=1.0,
                )
                if config.entity_linking and hit.lei and hit.entity_type in pack.gazetteer_link_types:
                    cand.external_id = hit.lei
                    cand.external_id_source = "gleif_lei"
                out.append(cand)
        return out
