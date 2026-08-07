"""Ontology packs (ADR-0008) — versioned, domain-agnostic type/relation vocabularies."""
from .loader import (
    EntityType,
    OntologyError,
    Pack,
    Relation,
    available_packs,
    load_pack,
)

__all__ = [
    "EntityType",
    "OntologyError",
    "Pack",
    "Relation",
    "available_packs",
    "load_pack",
]
