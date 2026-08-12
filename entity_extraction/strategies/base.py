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
    Candidate, Fact, Relation, assign_occurrence_indices, attach_facts_to_entity_records,
    build_edge_records, build_entity_records, build_summary, derive_abstract_entities,
    entity_uid_key_map, evidence_grounded,
    filter_bare_pronouns, mark_reference_only, merge_candidates, merge_edge_records,
    vote_relations,
)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: Optional[str] = None  # KG-AC-73 (v13) — from chunk_metadata.source.filename
    page: Optional[int] = None    # KG-AC-73 (v13) — from chunk_metadata.page


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
    relation_strategy: str = "generate"    # KG-AC-59/65 — generate | classify | entity_scoped
    relation_self_consistency_k: int = 1   # KG-AC-67 — self-consistency voting; 1=off (default),
                                            # bounded to [1,5] at point of use (run_pipeline)
    llm_max_tokens: Optional[int] = None   # KG-AC-87 — Bedrock Converse maxTokens override;
                                            # None = the client's own default (today's 4096, unchanged)
    llm_model: Optional[str] = None        # KG-AC-95 — Bedrock Converse model id override, set on
                                            # the PROFILE (not the shared connection); None = the
                                            # connection's own configured model, then the client default

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
            relation_self_consistency_k=int(d.get("relation_self_consistency_k", 1)),
            llm_max_tokens=int(d["llm_max_tokens"]) if d.get("llm_max_tokens") is not None else None,
            llm_model=d.get("llm_model"),
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


def validate_relations(
    relations: List[Relation], pack, chunk_text_by_id: dict,
) -> Tuple[List[Relation], int]:
    """KG-AC-43 (carries KG-AC-16's surviving guarantee) + KG-AC-64 (evolve v12 — evidence
    grounding): keep only relations whose type + src/dst types satisfy the pack's domain/range (a
    subtype of a declared domain/range type is accepted; illegal pairings dropped, uncounted) AND —
    for LLM-sourced relations only (``extractor != 'rules'``) — whose evidence occurs verbatim in
    the source chunk (KG-AC-64; deterministic-layer relations are exempt, their evidence is the
    matched sentence by construction). Ungrounded relations are dropped AND counted. Deterministic.
    Returns (kept, ungrounded_count)."""
    kept: List[Relation] = []
    ungrounded = 0
    for r in relations:
        if not pack.relation_allowed(r.relation_type, r.src_type, r.dst_type):
            continue
        if r.extractor != "rules":
            chunk_text = chunk_text_by_id.get(r.source_chunk_id, "")
            if not evidence_grounded(r.evidence_text, chunk_text):
                ungrounded += 1
                continue
        kept.append(r)
    return kept, ungrounded


def validate_facts(
    facts: List[Fact], pack, chunk_text_by_id: dict,
) -> Tuple[List[Fact], int, int]:
    """KG-AC-70 (evolve v13): keep only facts whose `property` is in the pack's declared attribute
    vocabulary AND whose subject's type satisfies that property's declared domain (a subtype of the
    declared domain is accepted) AND whose evidence occurs verbatim in the source chunk (KG-AC-64's
    grounding rule, reused via `core.evidence_grounded`, not duplicated).

    Unlike `validate_relations`' illegal-domain-UNCOUNTED posture, KG-AC-70's own text says "a fact
    failing EITHER gate is dropped and counted" — both an unknown property and an invalid domain
    fold into `unmapped_property_count` (the one vocabulary-mapping counter KG-AC-74 names; there is
    no separate "illegal domain" counter for facts the way relations have none at all). An
    ungrounded evidence counts toward `ungrounded_fact_count`. Deterministic.
    Returns (kept, unmapped_property_count, ungrounded_fact_count)."""
    kept: List[Fact] = []
    unmapped_property = 0
    ungrounded = 0
    for f in facts:
        dp = pack.datatype_properties.get(f.property)
        if dp is None:
            unmapped_property += 1
            continue
        # KG-AC-93 (v15): the pack-declared ANCHOR of an abstract domain also satisfies it — the
        # model is offered such a property under that anchor (it cannot emit the abstract type at
        # all, KG-AC-89), and derivation re-parents the fact onto the minted instance afterwards.
        # Read from the pack's own `identity_from`, never inferred from co-occurrence.
        anchor = pack.anchor_type_for(dp.domain)
        if (f.subject_type != dp.domain
                and not pack.is_descendant(f.subject_type, dp.domain)
                and f.subject_type != anchor):
            unmapped_property += 1
            continue
        chunk_text = chunk_text_by_id.get(f.source_chunk_id, "")
        if not evidence_grounded(f.evidence_text, chunk_text):
            ungrounded += 1
            continue
        kept.append(f)
    return kept, unmapped_property, ungrounded


# -- orchestration (impure — instantiates the injected strategies) ---------
def run_graph_extraction(chunks: List[Chunk], config: ExtractionConfig, pack, *,
                         spacy_model_path=None, spacy_nlp=None,
                         llm_client=None,
                         unresolved_reference_sink: Optional[List[int]] = None,
                         unlocatable_entity_sink: Optional[List[int]] = None,
                         facts_sink: Optional[List[List[Fact]]] = None,
                         ) -> Tuple[List[Candidate], List[Relation]]:
    """KG-AC-43/44: the active entity layer(s) + relations for this config. ``engine=spacy`` is
    entity-only (zero relations, KG-AC-44). ``engine=llm`` emits entities AND relations from ONE
    call per chunk via ``LlmGraphStrategy`` (KG-AC-43). ``spacy_nlp`` is a pass-through test seam
    mirroring ``SpacyNerStrategy``'s own ``nlp=`` param. ``unresolved_reference_sink`` (KG-AC-71,
    v13): an optional mutable out-list — when provided and ``generate`` mode is active, the
    strategy's own ``unresolved_reference_count`` is appended to it. ``unlocatable_entity_sink``
    (KG-AC-72, v13): same out-list mechanism, appended whenever ``engine=llm`` regardless of
    relation_strategy (entities from THIS call are always used, unlike relations under
    classify/entity_scoped). Both are plain out-parameters (not extra return values) so this
    function's existing 2-tuple return shape — depended on by several call sites, including test
    harnesses that unpack it directly — never changes."""
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
        if unlocatable_entity_sink is not None:  # KG-AC-72: entities from THIS call are always
            unlocatable_entity_sink.append(graph.unlocatable_entity_count)  # used, any relation_strategy
        if facts_sink is not None:  # KG-AC-70: facts, like entities, ALWAYS come from this ONE call
            facts_sink.append(graph.facts)  # regardless of relation_strategy — never re-collected
        if config.relation_strategy == "generate":  # KG-AC-59: generate XOR classify — under
            relations = graph.relations             # classify, this SAME call's relations are
            # discarded; entities from it are still used. Classify's own relations come from
            # llm_classify.py via run_pipeline (needs the post-merge entity set), not here.
            if unresolved_reference_sink is not None:
                unresolved_reference_sink.append(graph.unresolved_reference_count)  # KG-AC-71
    else:
        raise ValueError(f"unknown entity engine: {config.engine!r}")

    if config.rules_relations_enabled:  # KG-AC-55/56 (evolve v8) — merged with engine relations in run_pipeline
        relations = relations + RulesRelationsStrategy(nlp=shared_nlp).extract(chunks, config, pack)

    return candidates, relations


def _screen_guardrails(
    candidates: List[Candidate], facts: List[Fact], screen,
) -> Tuple[List[Candidate], List[Fact], int, int]:
    """KG-AC-17/84: guardrails invoked ONCE per batch — entity candidates AND facts are screened
    together in the SAME call (KG-AC-84: never a second guardrails_check call for facts). A blocking
    verdict drops the blocked item(s) and counts them; the task still succeeds. ``screen(items) ->
    list[bool]`` where ``items`` is ``candidates + facts`` concatenated, in that order (True = keep).
    Returns (kept_candidates, kept_facts, blocked_candidate_count, blocked_fact_count)."""
    items = candidates + facts
    keep_flags = screen(items)
    n = len(candidates)
    cand_flags, fact_flags = keep_flags[:n], keep_flags[n:]
    kept_candidates = [c for c, keep in zip(candidates, cand_flags) if keep]
    kept_facts = [f for f, keep in zip(facts, fact_flags) if keep]
    return (kept_candidates, kept_facts,
            len(candidates) - len(kept_candidates), len(facts) - len(kept_facts))


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
    # KG-AC-73 (v13): computed from the ORIGINAL chunks (before any coref rewrite) — reflects the
    # true completeness of the source data's chunk_metadata, independent of coreference resolution.
    chunk_metadata_missing_count = sum(1 for ch in chunks if ch.doc_id is None or ch.page is None)
    chunk_provenance = {ch.chunk_id: (ch.doc_id, ch.page) for ch in chunks}

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

    # KG-AC-71 (v13): accumulates unresolved_reference_count across every LLM relation call made for
    # this pipeline run — run 1 AND every self-consistency repeat, whichever mode is active. A plain
    # mutable list (not a running int) so run_graph_extraction's optional out-parameter and the
    # closures below can all append to the SAME accumulator without a `nonlocal` declaration.
    unresolved_counts: List[int] = []
    # KG-AC-72 (v13): unlike unresolved_counts, this is captured ONLY from run_graph_extraction's
    # own call — entities always come from run 1 alone (self-consistency repeats' entities are
    # discarded, per the structural finding above), so counting a repeat run's unlocatable drops
    # would tally something that never affected the written graph.
    unlocatable_entity_counts: List[int] = []
    # KG-AC-70 (v13): facts, like entities, come from run_graph_extraction's ONE primary call only
    # — never re-collected from a self-consistency repeat (mirrors the entities-from-run-1 rule).
    facts_lists: List[List[Fact]] = []

    candidates, raw_relations = run_graph_extraction(
        chunks, config, pack, spacy_model_path=spacy_model_path,
        spacy_nlp=shared_nlp, llm_client=llm_client,
        unresolved_reference_sink=unresolved_counts,
        unlocatable_entity_sink=unlocatable_entity_counts,
        facts_sink=facts_lists,
    )
    if config.coreference_enabled:
        candidates = filter_bare_pronouns(candidates)
    # the LLM model id is known only after the client resolves its connection (on first call)
    model_id = getattr(llm_client, "resolved_model", None) if llm_client is not None else None
    kept, unmapped = filter_closed_vocab(candidates, config, pack)
    # KG-AC-70 (v13): facts, like entities, come from run_graph_extraction's ONE primary call only
    # (see the accumulator comment above) — read here, before guardrails, so the SAME once-per-batch
    # screening call below can cover both (KG-AC-84).
    raw_facts = facts_lists[0] if facts_lists else []
    guardrails_blocked = 0
    guardrails_blocked_facts = 0
    if guardrails_screen is not None:
        # KG-AC-84: candidates AND facts screened together in ONE call — never a second
        # guardrails_check call for facts (KG-AC-17's once-per-batch posture, extended).
        kept, raw_facts, guardrails_blocked, guardrails_blocked_facts = _screen_guardrails(
            kept, raw_facts, guardrails_screen)
    assign_occurrence_indices(kept)
    merged = merge_candidates(kept)
    ent_rows = build_entity_records(folder_id, merged, config.ontology_pack, pack.version,
                                    model_id=model_id, chunk_provenance=chunk_provenance)

    # KG-AC-64 (evolve v12): every LLM relation mode needs chunk text for evidence grounding, not
    # just classify's candidate-pair prompt — built once, unconditionally.
    chunk_text_by_id = {ch.chunk_id: ch.text for ch in chunks}

    # KG-AC-70 (v13): domain-validate + evidence-ground the run's (already guardrails-screened)
    # facts, then nest survivors onto their subject's kg_entities row. A fact whose subject was
    # itself dropped by filter_closed_vocab/guardrails (unmapped type, blocked, ...) has no matching
    # row in `ent_rows` and is silently excluded by attach_facts_to_entity_records — the SAME
    # dangling-endpoint posture build_edge_records already applies to relations (uncounted; the
    # subject's own drop is what got counted, not a second reason).
    kept_facts, unmapped_property, ungrounded_fact = validate_facts(raw_facts, pack, chunk_text_by_id)
    attach_facts_to_entity_records(ent_rows, kept_facts, chunk_provenance)

    # KG-AC-90/91 (v15): mint pack-declared abstract types deterministically. Placed HERE by
    # contract, not convenience — strictly AFTER attach_facts_to_entity_records (the identity value
    # IS a fact, so nothing is derivable before facts land) and BEFORE build_edge_records (the
    # derived rows must be in ent_rows for their edges to survive the dangling-endpoint drop).
    derived_rows, derived_edges, derived_counters = derive_abstract_entities(
        folder_id, ent_rows, pack, config.ontology_pack, pack.version)
    ent_rows = ent_rows + derived_rows
    # KG-AC-94 (v15): computed from ATTACHED facts, so strictly after derivation's re-parenting —
    # an anchor whose bundle attributes have just moved off it must not be mis-flagged as an
    # unread stub. Derived rows are excluded inside the helper (clarify F2).
    mark_reference_only(ent_rows, pack)

    # KG-AC-67 (evolve v12): self-consistency voting needs the deterministic rules-layer relations
    # (already run ONCE, inside run_graph_extraction) kept SEPARATE from the active LLM relation
    # mode's own output — rules relations are exempt from voting and must never be multiplied by k.
    # `raw_relations` at this point mixes both when engine=llm+generate (run_graph_extraction
    # appended rules relations to whatever generate produced); split them apart.
    rules_relations = [r for r in raw_relations if r.extractor == "rules"]
    llm_relations_run1 = [r for r in raw_relations if r.extractor != "rules"]

    pairs = None  # computed once under classify; reused by repeat runs if k>1
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
        classify_strategy = LlmClassifyStrategy(llm_client=llm_client)
        llm_relations_run1 = classify_strategy.classify(pairs, chunk_text_by_id)
        unresolved_counts.append(classify_strategy.unresolved_reference_count)  # KG-AC-71
    elif config.relation_strategy == "entity_scoped":
        # KG-AC-65 (evolve v12): one call per chunk over the FINAL merged entity set — no
        # candidate-pair enumeration, no sentence-boundary nlp needed at all (unlike classify).
        # Lazy RELATIVE import, same safety reasoning as llm_classify's import above.
        from .llm_entity_scoped import LlmEntityScopedStrategy
        entity_scoped_strategy = LlmEntityScopedStrategy(llm_client=llm_client)
        llm_relations_run1 = entity_scoped_strategy.extract(merged, chunk_text_by_id, pack)
        unresolved_counts.append(entity_scoped_strategy.unresolved_reference_count)  # KG-AC-71
    # else "generate": llm_relations_run1 is already set, from run_graph_extraction's own call.

    # KG-AC-67 (evolve v12): self-consistency voting. k<=1 is an EXACT no-op — byte-identical
    # output, ONE call total, the path every existing profile uses. k>1 repeats ONLY the active LLM
    # relation mode's own call, NEVER the entity/rules-producing run_graph_extraction call — so
    # entities always come from run 1 alone (a structural finding from cb-plan: under `generate`,
    # entities and relations come from the SAME call, so a repeat run necessarily re-extracts
    # entities too, but those are discarded here — only `.relations` is read).
    k = max(1, min(config.relation_self_consistency_k, 5))
    if k <= 1:
        llm_relations = llm_relations_run1
        self_consistency_votes = 1
    else:
        def _one_more_llm_relation_run() -> List[Relation]:
            # KG-AC-71: each repeat call's own unresolved_reference_count is appended to the SAME
            # outer accumulator as run 1's — every LLM relation call this pipeline run makes is
            # counted, not just the first.
            if config.relation_strategy == "classify":
                from .llm_classify import LlmClassifyStrategy
                strategy = LlmClassifyStrategy(llm_client=llm_client)
                result = strategy.classify(pairs, chunk_text_by_id)
                unresolved_counts.append(strategy.unresolved_reference_count)
                return result
            if config.relation_strategy == "entity_scoped":
                from .llm_entity_scoped import LlmEntityScopedStrategy
                strategy = LlmEntityScopedStrategy(llm_client=llm_client)
                result = strategy.extract(merged, chunk_text_by_id, pack)
                unresolved_counts.append(strategy.unresolved_reference_count)
                return result
            # "generate": re-invoke LlmGraphStrategy fresh; its entities are discarded here, only
            # .relations is read (relative import, same safety posture as the imports above).
            from .llm_graph import LlmGraphStrategy
            graph = LlmGraphStrategy(llm_client=llm_client)
            graph.extract(chunks, config, pack)
            unresolved_counts.append(graph.unresolved_reference_count)
            return graph.relations

        runs = [llm_relations_run1] + [_one_more_llm_relation_run() for _ in range(k - 1)]
        llm_relations = vote_relations(runs, k)
        self_consistency_votes = k

    raw_relations = rules_relations + llm_relations
    relations, ungrounded = validate_relations(raw_relations, pack, chunk_text_by_id)
    edge_rows = build_edge_records(folder_id, relations, entity_uid_key_map(ent_rows),
                                   chunk_provenance=chunk_provenance)
    # KG-AC-91: derived edges are already keyed by entity_uid — a derived instance is
    # document-scoped (`source_chunk_id=None`) so it cannot be resolved through
    # entity_uid_key_map's (chunk, type, surface) lookup. Appended, then deduped alongside the
    # model's own edges by the existing merge.
    edge_rows = merge_edge_records(edge_rows + derived_edges)  # KG-AC-56 (v8) — union+dedup

    summary = build_summary(ent_rows, edge_rows, config.ontology_pack, pack.version,
                            unmapped, config.promote_top_n, ungrounded_relation_count=ungrounded,
                            self_consistency_votes=self_consistency_votes,
                            chunk_metadata_missing_count=chunk_metadata_missing_count,
                            unresolved_reference_count=sum(unresolved_counts),
                            unlocatable_entity_count=sum(unlocatable_entity_counts),
                            unmapped_property_count=unmapped_property,
                            ungrounded_fact_count=ungrounded_fact,
                            guardrails_blocked_facts=guardrails_blocked_facts,
                            **derived_counters)
    usage = list(getattr(llm_client, "usage", []) or [])
    return ent_rows, edge_rows, summary, usage, guardrails_blocked
