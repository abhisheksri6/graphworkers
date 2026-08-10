"""Strategy registry + ensemble orchestration (ADR-0009; evolve v5 — relationship intelligence;
evolve v8 ADR-0012 — composable additive deterministic layers).

The entity ensemble is a precedence stack — regex (opt-in, evolve v8) > spaCy > LLM (*amended v11:
the gazetteer tier that used to sit above regex is withdrawn with that capability*) — all in-worker
behind one queue. ``engine`` selects the primary layer: ``spacy`` (its OWN output is entity-only,
KG-AC-44) or ``llm`` (entities AND relations from ONE schema-constrained call per chunk, KG-AC-43 —
supersedes the v4 two-call NER-then-relation path and the withdrawn ``relation_engine`` axis,
KG-AC-16). Evolve v8 adds two OPT-IN, ADDITIVE deterministic layers that run alongside the primary
layer regardless of which `engine` is chosen: ``rules_entities_enabled`` (EntityRuler regex/phrase,
KG-AC-54) and ``rules_relations_enabled`` (DependencyMatcher, KG-AC-55 — so a `spacy`-engine config
CAN carry relations once this layer is on, superseding the old "no relation configuration is
consulted" reading of KG-AC-44). The fixed precedence (core.LAYER_PRECEDENCE) resolves entity span
overlaps at merge time (KG-AC-12); relations from multiple sources are unioned+deduped with
`extractor` provenance (KG-AC-56, run_pipeline). Every emitted relation is validated against the
pack's domain/range (KG-AC-43, KG-AC-16's surviving guarantee); a relation whose endpoint wasn't
extracted is dropped at edge-build (dangling-endpoint drop). Unknown entity types are dropped +
counted (closed vocabulary, KG-AC-14).

Strategy dependencies (LLM client C7, spaCy model/nlp C12) are INJECTED, so this module and the
pure filters below are unit-testable with fakes today.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from candidate_pairs import enumerate_candidate_pairs
from core import (
    Candidate, Relation, assign_occurrence_indices, build_edge_records, build_entity_records,
    build_summary, entity_uid_key_map, filter_bare_pronouns, merge_candidates, merge_edge_records,
)


@dataclass
class Chunk:
    chunk_id: str
    text: str


@dataclass
class ExtractionConfig:
    engine: str = "spacy"                 # spacy | llm
    ontology_pack: str = "generic"
    confidence_threshold: float = 0.0
    promote_top_n: int = 10
    entity_types: Optional[List[str]] = None   # optional subset filter over the pack's types
    connection_id: Optional[str] = None
    coreference_enabled: bool = False      # KG-AC-48 — document-scoped anaphora resolution, opt-in
    rules_entities_enabled: bool = False   # KG-AC-54 — opt-in EntityRuler regex/phrase entity layer
    rules_relations_enabled: bool = False  # KG-AC-55 — opt-in DependencyMatcher relation layer
    relation_strategy: str = "generate"    # KG-AC-59 — generate | classify, decoupled from engine

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionConfig":
        d = d or {}
        return cls(
            engine=d.get("engine", "spacy"),
            ontology_pack=d.get("ontology_pack", "generic"),
            confidence_threshold=float(d.get("confidence_threshold", 0.0)),
            promote_top_n=int(d.get("promote_top_n", 10)),
            entity_types=d.get("entity_types"),
            connection_id=d.get("connection_id"),
            coreference_enabled=bool(d.get("coreference_enabled", False)),
            rules_entities_enabled=bool(d.get("rules_entities_enabled", False)),
            rules_relations_enabled=bool(d.get("rules_relations_enabled", False)),
            relation_strategy=d.get("relation_strategy", "generate"),
        )


class EntityStrategy:
    """Base entity strategy. ``layer`` sets its precedence bucket (core.LAYER_PRECEDENCE)."""
    layer: str = "spacy"

    def extract(self, chunks: List[Chunk], config: ExtractionConfig, pack) -> List[Candidate]:
        raise NotImplementedError


# -- pure filters ----------------------------------------------------------
def filter_closed_vocab(candidates: List[Candidate], config: ExtractionConfig, pack) -> Tuple[List[Candidate], int]:
    """KG-AC-14: drop candidates whose type is outside the pack (counting them as unmapped), then
    apply the optional ``entity_types`` subset filter and the confidence threshold. Returns
    (kept, unmapped_type_count). The subset/threshold drops are NOT counted as unmapped."""
    kept: List[Candidate] = []
    unmapped = 0
    subset = set(config.entity_types) if config.entity_types else None
    for c in candidates:
        if not pack.is_known_type(c.entity_type):
            unmapped += 1
            continue
        if subset is not None and c.entity_type not in subset:
            continue
        if c.confidence < config.confidence_threshold:
            continue
        kept.append(c)
    return kept, unmapped


def validate_relations(relations: List[Relation], pack) -> List[Relation]:
    """KG-AC-43 (carries KG-AC-16's surviving guarantee): keep only relations whose type + src/dst
    types satisfy the pack's domain/range (a subtype of a declared domain/range type is accepted);
    illegal pairings dropped. Deterministic."""
    return [r for r in relations if pack.relation_allowed(r.relation_type, r.src_type, r.dst_type)]


# -- orchestration (impure — instantiates the injected strategies) ---------
def run_graph_extraction(chunks: List[Chunk], config: ExtractionConfig, pack, *,
                         spacy_model_path=None, spacy_nlp=None,
                         llm_client=None) -> Tuple[List[Candidate], List[Relation]]:
    """KG-AC-43/44: the active entity layer(s) + relations for this config. ``engine=spacy`` is
    entity-only (zero relations, KG-AC-44). ``engine=llm`` emits entities AND relations from ONE
    call per chunk via ``LlmGraphStrategy`` (KG-AC-43). ``spacy_nlp`` is a pass-through test seam
    mirroring ``SpacyNerStrategy``'s own ``nlp=`` param."""
    from .llm_graph import LlmGraphStrategy
    from .rules_entities import RulesEntitiesStrategy
    from .rules_relations import RulesRelationsStrategy
    from .spacy_ner import SpacyNerStrategy, load_spacy_model

    candidates: List[Candidate] = []
    relations: List[Relation] = []

    if config.rules_entities_enabled:  # KG-AC-54 (evolve v8) — opt-in, additive
        candidates += RulesEntitiesStrategy().extract(chunks, config, pack)

    # KG-AC-55 (evolve v8): the DependencyMatcher relation layer needs a parsed spaCy doc
    # regardless of which entity `engine` is active — share ONE loaded nlp with SpacyNerStrategy
    # (when engine=spacy) so the ~600MB by-copy model is loaded at most once per task.
    shared_nlp = spacy_nlp
    if shared_nlp is None and (config.engine == "spacy" or config.rules_relations_enabled):
        shared_nlp = load_spacy_model(spacy_model_path)

    if config.engine == "spacy":
        candidates += SpacyNerStrategy(nlp=shared_nlp).extract(chunks, config, pack)
        # spaCy's OWN output stays node-only -- no relation config exists to consult (KG-AC-44)
    elif config.engine == "llm":
        graph = LlmGraphStrategy(llm_client=llm_client)
        candidates += graph.extract(chunks, config, pack)
        if config.relation_strategy == "generate":  # KG-AC-59: generate XOR classify — under
            relations = graph.relations             # classify, this SAME call's relations are
            # discarded; entities from it are still used. Classify's own relations come from
            # llm_classify.py via run_pipeline (needs the post-merge entity set), not here.
    else:
        raise ValueError(f"unknown entity engine: {config.engine!r}")

    if config.rules_relations_enabled:  # KG-AC-55/56 (evolve v8) — merged with engine relations in run_pipeline
        relations = relations + RulesRelationsStrategy(nlp=shared_nlp).extract(chunks, config, pack)

    return candidates, relations


def _screen_guardrails(candidates: List[Candidate], screen) -> Tuple[List[Candidate], int]:
    """KG-AC-17: guardrails invoked ONCE per batch; a blocking verdict drops the blocked candidates
    and counts them — the task still succeeds. ``screen(candidates) -> list[bool]`` (True = keep)."""
    keep_flags = screen(candidates)
    kept = [c for c, keep in zip(candidates, keep_flags) if keep]
    return kept, len(candidates) - len(kept)


def run_pipeline(chunks: List[Chunk], config: ExtractionConfig, pack, *, folder_id: str,
                 spacy_model_path=None, spacy_nlp=None, llm_client=None,
                 guardrails_screen=None):
    """End-to-end extraction orchestration (strategies injected; no Celery/DB/HTTP) — the testable
    core of the worker task. Returns (entity_rows, edge_rows, summary, usage, guardrails_blocked).
    Zero candidates ⇒ empty graph with entity_count=0 (KG-AC-33 — a valid, successful empty result).
    Evolve v5: entities+relations come from run_graph_extraction (one LLM pass when engine=llm,
    KG-AC-43); relations validated by domain/range then dangling-endpoint edges dropped at
    build_edge_records (an endpoint the closed-vocab filter or merge dropped is never written).
    KG-AC-48: when coreference_enabled (engine=llm only — piggybacks on the same connection, no new
    requirement), a document-scoped coreference pre-pass rewrites the chunks BEFORE extraction, and
    any bare-pronoun entity that survives anyway is filtered deterministically. This single call site
    means BOTH the production task and the runtime-preview path (which both call run_pipeline) get
    identical coreference behavior — never a divergent code path."""
    if config.coreference_enabled and config.engine == "llm":
        from .coref import resolve_coreferences
        chunks = resolve_coreferences(chunks, llm_client)

    # KG-AC-59/60: classify mode needs a parsed nlp for sentence boundaries too — resolved HERE
    # (once) and threaded into run_graph_extraction via spacy_nlp= so it never double-loads the
    # ~600MB by-copy model when engine=spacy or rules_relations_enabled also need it (same "share
    # ONE loaded nlp" goal run_graph_extraction already applies internally for its own two triggers).
    shared_nlp = spacy_nlp
    if shared_nlp is None and (
        config.engine == "spacy" or config.rules_relations_enabled or config.relation_strategy == "classify"
    ):
        from .spacy_ner import load_spacy_model
        shared_nlp = load_spacy_model(spacy_model_path)

    candidates, raw_relations = run_graph_extraction(
        chunks, config, pack, spacy_model_path=spacy_model_path,
        spacy_nlp=shared_nlp, llm_client=llm_client,
    )
    if config.coreference_enabled:
        candidates = filter_bare_pronouns(candidates)
    # the LLM model id is known only after the client resolves its connection (on first call)
    model_id = getattr(llm_client, "resolved_model", None) if llm_client is not None else None
    kept, unmapped = filter_closed_vocab(candidates, config, pack)
    guardrails_blocked = 0
    if guardrails_screen is not None:
        kept, guardrails_blocked = _screen_guardrails(kept, guardrails_screen)
    assign_occurrence_indices(kept)
    merged = merge_candidates(kept)
    ent_rows = build_entity_records(folder_id, merged, config.ontology_pack, pack.version, model_id=model_id)

    if config.relation_strategy == "classify":
        # KG-AC-60/61: candidate pairs are enumerated over the FINAL merged (post span-overlap-
        # resolution) entity set, never the raw pre-merge candidates — otherwise overlapping
        # mentions from different layers would pollute pairing with duplicate/conflicting spans.
        # `enumerate_candidate_pairs` is imported at MODULE level (top of this file), NOT here:
        # Celery drops cwd from sys.path after the app import, so a lazy absolute import of a
        # top-level worker module raises ModuleNotFoundError at task time — the 2026-08-08
        # production failure (see tests/test_worker_runtime_imports.py). The relative import
        # below is safe: `strategies` is already in sys.modules with a known __path__.
        from .llm_classify import LlmClassifyStrategy
        sentence_spans = {ch.chunk_id: [(s.start_char, s.end_char) for s in shared_nlp(ch.text).sents]
                          for ch in chunks}
        pairs = enumerate_candidate_pairs(merged, sentence_spans, pack)
        chunk_text_by_id = {ch.chunk_id: ch.text for ch in chunks}
        raw_relations = raw_relations + LlmClassifyStrategy(llm_client=llm_client).classify(pairs, chunk_text_by_id)

    relations = validate_relations(raw_relations, pack)
    edge_rows = build_edge_records(folder_id, relations, entity_uid_key_map(ent_rows))
    edge_rows = merge_edge_records(edge_rows)  # KG-AC-56 (evolve v8) — union+dedup multi-source relations

    summary = build_summary(ent_rows, edge_rows, config.ontology_pack, pack.version,
                            unmapped, config.promote_top_n)
    usage = list(getattr(llm_client, "usage", []) or [])
    return ent_rows, edge_rows, summary, usage, guardrails_blocked
