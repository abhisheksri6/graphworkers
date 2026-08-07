"""Ontology packs (ADR-0008) — versioned, domain-agnostic type/relation vocabularies."""
from .loader import (
    DepPattern,
    EntityType,
    OntologyError,
    Pack,
    RegexPattern,
    Relation,
    available_packs,
    load_pack,
)

__all__ = [
    "DepPattern",
    "EntityType",
    "OntologyError",
    "Pack",
    "RegexPattern",
    "Relation",
    "available_packs",
    "load_pack",
]
