"""Ontology pack schema + loader (ADR-0008). Pure — json + validation only, no spaCy/DB/network.

A pack is a versioned, **domain-agnostic** JSON: entity types (each with a `parent` hierarchy validated
as a DAG, per-engine mapping via `spacy_labels`, `guidance` for the LLM prompt, and an OPTIONAL inert
`iri`) + relations (with `domain`/`range` typing). The runtime vocabulary is CLOSED — `is_known_type`
gates what may be written; unknown types are dropped + counted by the strategies (KG-AC-14). The
`parent` hierarchy drives canonicalization's type reconciliation (KG-AC-23). `generic`/`fibo_core` are
shipped *samples*; a customer authors a custom pack for any domain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_PACK_DIR = Path(__file__).resolve().parent


class OntologyError(ValueError):
    """Malformed pack, unknown pack, or an invalid hierarchy — always fail loud (KG-AC-14)."""


@dataclass
class EntityType:
    type: str
    parent: Optional[str]
    spacy_labels: List[str]
    guidance: str
    iri: Optional[str]


@dataclass
class Relation:
    type: str
    domain: List[str]
    range: List[str]
    guidance: str


class Pack:
    def __init__(self, name: str, version: str, description: str,
                 entity_types: Sequence[EntityType], relations: Sequence[Relation],
                 gazetteer_link_types: Sequence[str]):
        self.name = name
        self.version = version
        self.description = description
        self.entity_types: Dict[str, EntityType] = {et.type: et for et in entity_types}
        self.relations: Dict[str, Relation] = {r.type: r for r in relations}
        self.gazetteer_link_types = set(gazetteer_link_types)
        self._declaration_order = [et.type for et in entity_types]
        self._spacy_map: Dict[str, str] = {}
        for et in entity_types:
            for label in et.spacy_labels:
                self._spacy_map[label] = et.type
        self._validate()

    # -- validation -------------------------------------------------------
    def _validate(self) -> None:
        if not self.version:
            raise OntologyError("pack version is mandatory")
        if not self.entity_types:
            raise OntologyError("pack must declare at least one entity type")
        if len(self._declaration_order) != len(set(self._declaration_order)):
            raise OntologyError("duplicate entity type declared")
        for et in self.entity_types.values():
            if et.parent is not None and et.parent not in self.entity_types:
                raise OntologyError(f"type '{et.type}' parent '{et.parent}' is not declared")
        self._assert_dag()
        for r in self.relations.values():
            for t in list(r.domain) + list(r.range):
                if t not in self.entity_types:
                    raise OntologyError(f"relation '{r.type}' references undeclared type '{t}'")
            if self.gazetteer_link_types - set(self.entity_types):
                raise OntologyError("gazetteer_link_types references an undeclared type")

    def _assert_dag(self) -> None:
        for start in self.entity_types:
            seen = set()
            cur: Optional[str] = start
            while cur is not None:
                if cur in seen:
                    raise OntologyError(f"cycle in parent hierarchy at '{cur}'")
                seen.add(cur)
                cur = self.entity_types[cur].parent

    # -- closed vocabulary (KG-AC-14) -------------------------------------
    def is_known_type(self, entity_type: str) -> bool:
        return entity_type in self.entity_types

    def map_spacy_label(self, label: str) -> Optional[str]:
        """spaCy NER label -> pack type, or None (an unmapped label is dropped + counted)."""
        return self._spacy_map.get(label)

    # -- hierarchy (KG-AC-23) ---------------------------------------------
    def ancestors(self, entity_type: str) -> List[str]:
        chain: List[str] = []
        node = self.entity_types.get(entity_type)
        cur = node.parent if node else None
        while cur is not None:
            chain.append(cur)
            cur = self.entity_types[cur].parent
        return chain

    def is_descendant(self, a: str, b: str) -> bool:
        """True iff a is a (strict) descendant of b."""
        return b in self.ancestors(a)

    def most_specific_type(self, types: Sequence[str]) -> Optional[str]:
        """KG-AC-23: among a cluster's candidate types choose the MOST SPECIFIC per the declared
        `parent` hierarchy — a descendant always beats its ancestor; two types on different branches
        (no ancestor/descendant relation) tie-break by pack DECLARATION ORDER. Unknown types are ignored."""
        cand = [t for t in dict.fromkeys(types) if self.is_known_type(t)]
        if not cand:
            return None
        # drop any candidate that is an ANCESTOR of another candidate (less specific)
        maximal = [t for t in cand if not any(o != t and self.is_descendant(o, t) for o in cand)]
        return min(maximal, key=lambda t: self._declaration_order.index(t))

    # -- relations (KG-AC-16) ---------------------------------------------
    def relation_allowed(self, relation_type: str, src_type: str, dst_type: str) -> bool:
        """A relation is legal iff its type is declared AND the src/dst types satisfy domain/range
        (a subtype of a declared domain/range type is accepted)."""
        r = self.relations.get(relation_type)
        if r is None:
            return False
        return self._type_satisfies(src_type, r.domain) and self._type_satisfies(dst_type, r.range)

    def _type_satisfies(self, t: str, allowed: Sequence[str]) -> bool:
        if t in allowed:
            return True
        return any(a in allowed for a in self.ancestors(t))


def load_pack(name_or_path: str) -> Pack:
    """Load a shipped pack by name (`generic`, `fibo_core`) or an explicit path. Unknown pack ⇒
    OntologyError (fail loud, KG-AC-14)."""
    path = Path(name_or_path)
    if not path.suffix:
        path = _PACK_DIR / f"{name_or_path}.json"
    if not path.exists():
        raise OntologyError(f"unknown ontology pack: {name_or_path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ets = [
            EntityType(type=e["type"], parent=e.get("parent"), spacy_labels=e.get("spacy_labels", []),
                       guidance=e.get("guidance", ""), iri=e.get("iri"))
            for e in data["entity_types"]
        ]
        rels = [
            Relation(type=r["type"], domain=r["domain"], range=r["range"], guidance=r.get("guidance", ""))
            for r in data.get("relations", [])
        ]
        return Pack(
            name=data["name"], version=str(data.get("version", "")), description=data.get("description", ""),
            entity_types=ets, relations=rels, gazetteer_link_types=data.get("gazetteer_link_types", []),
        )
    except KeyError as exc:
        raise OntologyError(f"malformed pack '{name_or_path}': missing key {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OntologyError(f"malformed pack '{name_or_path}': {exc}") from exc


def available_packs() -> List[str]:
    """Names of the shipped sample packs (backend strategy-catalog uses this vocabulary)."""
    return sorted(p.stem for p in _PACK_DIR.glob("*.json"))
