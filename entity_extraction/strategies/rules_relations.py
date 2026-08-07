"""DependencyMatcher relation layer (KG-AC-55, ADR-0012 §3 / ADR-0013). Deterministic, confidence
1.0 (clarify 2026-08-06). Patterns are pack-authored (``dep_patterns``); every match carries the
matched sentence as ``evidence_text`` (KG-AC-46 — mandatory). Opt-in via
``ExtractionConfig.rules_relations_enabled``; requires the loaded spaCy model's ``parser`` pipe
(which also supplies sentence boundaries for evidence) — fails loud if absent, never a silent
zero-relation result (KG-AC-42 posture).

**Typing decision (design detail, not spec-mandated — recorded here for the Done writeup):** a
syntactic dep-match has no NER capability of its own, so a match's ``src_type``/``dst_type`` are
taken from the pack relation's declared ``domain[0]``/``range[0]``. This is correct for the v8
sample patterns (single-type domain/range, e.g. ``employs: domain=[Organization], range=[Person]``)
and stays safe under multi-type domain/range because ``build_edge_records``'s dangling-endpoint
drop still rejects a guessed type+surface that doesn't match any actually-written entity — the
same safety net the LLM-sourced relations already rely on. Multi-type domain/range disambiguation
is out of scope (no shipped pattern needs it).
"""
from __future__ import annotations

from typing import Any, List, Optional

from core import Relation

from .base import Chunk, ExtractionConfig


class RulesRelationsStrategy:
    def __init__(self, nlp: Any = None):
        self._nlp = nlp  # injectable for tests; production shares the entity layer's loaded nlp

    def extract(self, chunks: List[Chunk], config: ExtractionConfig, pack) -> List[Relation]:
        if not pack.dep_patterns:
            return []
        nlp = self._nlp
        if nlp is None:
            raise RuntimeError(
                "RulesRelationsStrategy requires a loaded spaCy nlp instance (SPACY_MODEL_PATH)")
        if "parser" not in nlp.pipe_names:
            raise RuntimeError(
                "rules_relations_enabled requires the spaCy model's 'parser' pipe (also needed for "
                f"sentence-boundary evidence); the loaded model's pipes are {nlp.pipe_names!r}")

        from spacy.matcher import DependencyMatcher

        # 2026-08-07 bugfix: spaCy's DependencyMatcher.add(key, patterns) REPLACES any patterns
        # already registered under `key`, it does not accumulate — so two dep_patterns sharing one
        # relation_type used to silently keep only the LAST-registered phrasing (or, with
        # differently-shaped patterns, crash on a node-count mismatch). Each pattern now gets its
        # own unique registration key (independent of how many patterns share a relation_type);
        # `pattern_by_key` maps back to the originating DepPattern (and its relation_type) at match
        # time, so patterns of genuinely different shapes for the same relation coexist correctly.
        matcher = DependencyMatcher(nlp.vocab)
        pattern_by_key = {}
        for idx, dp in enumerate(pack.dep_patterns):
            key = f"{dp.relation_type}#{idx}"
            matcher.add(key, [dp.pattern])
            pattern_by_key[key] = dp

        out: List[Relation] = []
        for ch in chunks:
            doc = nlp(ch.text)
            for match_id, token_ids in matcher(doc):
                dp = pattern_by_key[nlp.vocab.strings[match_id]]
                node_order = [tok["RIGHT_ID"] for tok in dp.pattern]
                by_node = dict(zip(node_order, token_ids))
                src_tok = doc[by_node[dp.src_node]]
                dst_tok = doc[by_node[dp.dst_node]]
                rel = pack.relations[dp.relation_type]
                out.append(Relation(
                    relation_type=dp.relation_type,
                    src_surface=src_tok.text, src_type=rel.domain[0],
                    dst_surface=dst_tok.text, dst_type=rel.range[0],
                    source_chunk_id=ch.chunk_id, confidence=1.0,
                    evidence_text=src_tok.sent.text, extractor="rules",
                ))
        return out
