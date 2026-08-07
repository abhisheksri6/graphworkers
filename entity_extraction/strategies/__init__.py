"""Extraction strategies (ADR-0009; evolve v5): gazetteer (opt-in) > spaCy > LLM entity layers.
``engine=llm`` emits relations too, from the same one-pass call (KG-AC-43). Evolve v8 (ADR-0012)
adds two opt-in, additive deterministic layers that run alongside the above: ``regex``
(RulesEntitiesStrategy, KG-AC-54) and the DependencyMatcher relation layer (KG-AC-55)."""
from .base import (
    Chunk,
    EntityStrategy,
    ExtractionConfig,
    filter_closed_vocab,
    run_graph_extraction,
    run_pipeline,
    validate_relations,
)
from .coref import resolve_coreferences
from .llm_graph import LlmConnectionError, LlmGraphStrategy, LlmOutputError
from .rules_entities import RulesEntitiesStrategy
from .rules_gazetteer import GazetteerHit, RulesGazetteerStrategy
from .spacy_ner import SpacyNerStrategy

__all__ = [
    "Chunk", "ExtractionConfig", "EntityStrategy",
    "LlmConnectionError", "LlmOutputError",
    "filter_closed_vocab", "validate_relations",
    "run_graph_extraction", "run_pipeline",
    "RulesGazetteerStrategy", "GazetteerHit", "SpacyNerStrategy",
    "LlmGraphStrategy", "resolve_coreferences",
    "RulesEntitiesStrategy",
]
