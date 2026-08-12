"""Ontology pack schema + loader (ADR-0008, amended v13 and v14). Pure — json + validation only, no
spaCy/DB/network.

A pack is a versioned, **domain-agnostic** JSON: entity types (each with a `parent` hierarchy validated
as a DAG, per-engine mapping via `spacy_labels`, `guidance` for the LLM prompt, an OPTIONAL `iri`, an
OPTIONAL `abstract` flag — KG-AC-72, default false — and, for an abstract type, an OPTIONAL `derived`
block — KG-AC-88, v14) + relations (with `domain`/`range` typing, an OPTIONAL `iri` — KG-AC-85) + an
OPTIONAL `datatype_properties` attribute vocabulary (KG-AC-69). The runtime vocabulary is CLOSED —
`is_known_type` gates what may be written; unknown types are dropped + counted by the strategies
(KG-AC-14). The `parent` hierarchy drives canonicalization's type reconciliation (KG-AC-23).
`generic`/`fibo_core` are shipped *samples*; a customer authors a custom pack for any domain.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_PACK_DIR = Path(__file__).resolve().parent

# KG-AC-54: the closed set of format-checksum kinds a regex_pattern may declare. A name outside
# this set fails loud at load (ADR-0013) — the algorithm itself is implemented by the runtime
# EntityRuler layer (J3), not here; this is schema validation only.
_KNOWN_CHECKSUMS = frozenset({"lei", "isin"})

# KG-AC-69: the closed set of scalar kinds a datatype_property's `range` may declare. Drives
# core.py's per-kind normalizer (P6) — an undeclared kind fails loud at load, same posture as an
# unknown regex_pattern checksum above.
_KNOWN_RANGE_KINDS = frozenset({"string", "number", "date", "identifier"})


class OntologyError(ValueError):
    """Malformed pack, unknown pack, or an invalid hierarchy — always fail loud (KG-AC-14)."""


@dataclass
class DerivedSpec:
    """KG-AC-88 (v14, ADR-0008 amended): declares how an abstract type is minted DETERMINISTICALLY
    after extraction, replacing the v13 model-synthesis-and-position-reference mechanism.
    ``identity_from`` is ``"<Type>.<datatypeProperty>"`` — a declared entity type + one of that
    type's declared ``datatype_properties`` — the fact value that becomes the derived instance's
    identity. ``mint_when`` is a plain declared entity-type name whose presence in the folder
    triggers minting (no expression grammar — clarify F4: every real case is "this type is
    present", so none is warranted). ``auto_relations`` lists declared relation types carrying
    this abstract type in EITHER its domain or its range — clarify F1: a domain-only rule would
    make `hasCommitment`/`hasCommercialTerms` (abstract type in range only) permanently
    unproducible. Edge direction needs no separate field: a relation's own domain=src/range=dst
    already fully determines it, whichever side the abstract type falls on."""
    identity_from: str
    mint_when: str
    auto_relations: List[str]


@dataclass
class EntityType:
    type: str
    parent: Optional[str]
    spacy_labels: List[str]
    guidance: str
    iri: Optional[str]
    abstract: bool = False  # KG-AC-72 (v13): only an abstract type may be written span-less
    derived: Optional[DerivedSpec] = None  # KG-AC-88 (v14): how to mint it, if at all


@dataclass
class Relation:
    type: str
    domain: List[str]
    range: List[str]
    guidance: str
    iri: Optional[str] = None  # KG-AC-85 (v13): retained now — previously dropped at parse time


@dataclass
class DatatypeProperty:
    """KG-AC-69 (v13): a pack-declared attribute — ``domain`` is one declared entity type,
    ``range`` is a scalar kind (see ``_KNOWN_RANGE_KINDS``). Consumed by the tool-schema builder
    (P6) to enum-constrain ``facts[].property`` and by core.py's per-kind normalizer."""
    property: str
    domain: str
    range: str
    guidance: str
    iri: Optional[str] = None


@dataclass
class RegexPattern:
    """KG-AC-54: a dev-authored EntityRuler regex/phrase pattern (ADR-0013). ``pattern`` is a raw
    regex matched against chunk text; ``checksum`` (optional) names a format-validation kind the
    runtime EntityRuler layer (J3) applies before accepting a match (e.g. an LEI/ISIN check digit) —
    a match failing the checksum is not written."""
    entity_type: str
    pattern: str
    checksum: Optional[str] = None


@dataclass
class DepPattern:
    """KG-AC-55: a dev-authored spaCy DependencyMatcher pattern (ADR-0013) — ``pattern`` is handed
    to ``DependencyMatcher.add`` unmodified by the runtime relation layer (J4). ``src_node``/
    ``dst_node`` name which pattern node (``RIGHT_ID``) becomes the relation's src/dst entity."""
    relation_type: str
    pattern: List[Dict[str, Any]]
    src_node: str
    dst_node: str


class Pack:
    def __init__(self, name: str, version: str, description: str,
                 entity_types: Sequence[EntityType], relations: Sequence[Relation],
                 regex_patterns: Sequence[RegexPattern] = (),
                 dep_patterns: Sequence[DepPattern] = (),
                 datatype_properties: Sequence[DatatypeProperty] = ()):
        self.name = name
        self.version = version
        self.description = description
        self.entity_types: Dict[str, EntityType] = {et.type: et for et in entity_types}
        self.relations: Dict[str, Relation] = {r.type: r for r in relations}
        self.regex_patterns: List[RegexPattern] = list(regex_patterns)
        self.dep_patterns: List[DepPattern] = list(dep_patterns)
        self.datatype_properties: Dict[str, DatatypeProperty] = {
            dp.property: dp for dp in datatype_properties
        }
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
        self._validate_regex_patterns()
        self._validate_dep_patterns()
        self._validate_datatype_properties()
        self._validate_derived()

    def _validate_derived(self) -> None:
        """KG-AC-88 (v14): fail loud naming pack + type. ``identity_from`` must reference a
        declared entity type AND one of that type's declared ``datatype_properties`` (exact
        domain match — this pack has no subtype hierarchy to reconcile against, per ADR-0008);
        ``mint_when`` must reference a declared entity type; every ``auto_relations`` entry must
        be a declared relation carrying THIS abstract type in either its domain or its range
        (clarify F1's regression guard — never domain-only)."""
        for et in self.entity_types.values():
            derived = et.derived
            if derived is None:
                continue
            label = f"pack '{self.name}': entity type '{et.type}' derived block"
            if "." not in derived.identity_from:
                raise OntologyError(
                    f"{label}: identity_from must be '<Type>.<datatypeProperty>', "
                    f"got {derived.identity_from!r}")
            id_type, _, id_prop = derived.identity_from.partition(".")
            if id_type not in self.entity_types:
                raise OntologyError(
                    f"{label}: identity_from references undeclared type '{id_type}'")
            prop = self.datatype_properties.get(id_prop)
            if prop is None or prop.domain != id_type:
                raise OntologyError(
                    f"{label}: identity_from '{derived.identity_from}' is not a declared "
                    f"datatype_properties entry with domain '{id_type}'")
            if derived.mint_when not in self.entity_types:
                raise OntologyError(
                    f"{label}: mint_when references undeclared type '{derived.mint_when}'")
            for rel_name in derived.auto_relations:
                rel = self.relations.get(rel_name)
                if rel is None:
                    raise OntologyError(
                        f"{label}: auto_relations entry '{rel_name}' is not a declared relation")
                if et.type not in rel.domain and et.type not in rel.range:
                    raise OntologyError(
                        f"{label}: auto_relations entry '{rel_name}' carries '{et.type}' in "
                        f"NEITHER its domain {rel.domain} nor its range {rel.range}")

    def _validate_datatype_properties(self) -> None:
        """KG-AC-69: every datatype_properties entry's domain is a declared entity type and its
        range is a known scalar kind — fail loud naming the pack + property (ADR-0008 amended)."""
        for dp in self.datatype_properties.values():
            if dp.domain not in self.entity_types:
                raise OntologyError(
                    f"pack '{self.name}': datatype_properties entry '{dp.property}' "
                    f"references undeclared domain type '{dp.domain}'")
            if dp.range not in _KNOWN_RANGE_KINDS:
                raise OntologyError(
                    f"pack '{self.name}': datatype_properties entry '{dp.property}' "
                    f"names an unknown range kind '{dp.range}' (known: {sorted(_KNOWN_RANGE_KINDS)})")

    def _validate_regex_patterns(self) -> None:
        """KG-AC-54: every regex_pattern's type is declared, its regex compiles, and any
        checksum kind is known — fail loud naming the pack + pattern (ADR-0013)."""
        for rp in self.regex_patterns:
            if rp.entity_type not in self.entity_types:
                raise OntologyError(
                    f"pack '{self.name}': regex_patterns entry references undeclared type '{rp.entity_type}'")
            try:
                re.compile(rp.pattern)
            except re.error as exc:
                raise OntologyError(
                    f"pack '{self.name}': regex_patterns entry for '{rp.entity_type}' "
                    f"is not a valid regex ({rp.pattern!r}): {exc}") from exc
            if rp.checksum is not None and rp.checksum not in _KNOWN_CHECKSUMS:
                raise OntologyError(
                    f"pack '{self.name}': regex_patterns entry for '{rp.entity_type}' "
                    f"names an unknown checksum kind '{rp.checksum}' (known: {sorted(_KNOWN_CHECKSUMS)})")

    def _validate_dep_patterns(self) -> None:
        """KG-AC-55: every dep_pattern's relation is declared and its token graph is well-formed
        (anchor first, every LEFT_ID refers to an already-introduced node, no duplicate RIGHT_ID,
        src/dst_node both resolve) — the same shape spaCy's DependencyMatcher.add() requires, so
        J4 can hand ``pattern`` to it unmodified. Fail loud naming the pack + pattern (ADR-0013)."""
        for dp in self.dep_patterns:
            if dp.relation_type not in self.relations:
                raise OntologyError(
                    f"pack '{self.name}': dep_patterns entry references undeclared relation type "
                    f"'{dp.relation_type}'")
            label = f"pack '{self.name}': dep_patterns entry for '{dp.relation_type}'"
            if not dp.pattern:
                raise OntologyError(f"{label}: pattern must be non-empty")
            anchor = dp.pattern[0]
            if "RIGHT_ID" not in anchor or "RIGHT_ATTRS" not in anchor:
                raise OntologyError(f"{label}: first token must be the anchor (RIGHT_ID + RIGHT_ATTRS)")
            if "LEFT_ID" in anchor or "REL_OP" in anchor:
                raise OntologyError(f"{label}: first token (the anchor) must not carry LEFT_ID/REL_OP")
            defined_ids = {anchor["RIGHT_ID"]}
            for tok in dp.pattern[1:]:
                missing = {"LEFT_ID", "REL_OP", "RIGHT_ID", "RIGHT_ATTRS"} - set(tok)
                if missing:
                    raise OntologyError(f"{label}: token missing required key(s) {sorted(missing)}")
                if tok["LEFT_ID"] not in defined_ids:
                    raise OntologyError(
                        f"{label}: token references undefined LEFT_ID '{tok['LEFT_ID']}'")
                if tok["RIGHT_ID"] in defined_ids:
                    raise OntologyError(f"{label}: duplicate RIGHT_ID '{tok['RIGHT_ID']}'")
                defined_ids.add(tok["RIGHT_ID"])
            if dp.src_node not in defined_ids or dp.dst_node not in defined_ids:
                raise OntologyError(
                    f"{label}: src_node/dst_node must name pattern nodes (have: {sorted(defined_ids)})")

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
            EntityType(
                type=e["type"], parent=e.get("parent"), spacy_labels=e.get("spacy_labels", []),
                guidance=e.get("guidance", ""), iri=e.get("iri"), abstract=e.get("abstract", False),
                derived=(
                    DerivedSpec(
                        identity_from=e["derived"]["identity_from"],
                        mint_when=e["derived"]["mint_when"],
                        auto_relations=list(e["derived"].get("auto_relations", [])),
                    )
                    if e.get("derived") is not None else None
                ),
            )
            for e in data["entity_types"]
        ]
        rels = [
            Relation(type=r["type"], domain=r["domain"], range=r["range"], guidance=r.get("guidance", ""),
                     iri=r.get("iri"))
            for r in data.get("relations", [])
        ]
        datatype_properties = [
            DatatypeProperty(property=p["property"], domain=p["domain"], range=p["range"],
                             guidance=p.get("guidance", ""), iri=p.get("iri"))
            for p in data.get("datatype_properties", [])
        ]
        regex_patterns = [
            RegexPattern(entity_type=p["entity_type"], pattern=p["pattern"], checksum=p.get("checksum"))
            for p in data.get("regex_patterns", [])
        ]
        dep_patterns = [
            DepPattern(relation_type=p["relation_type"], pattern=p["pattern"],
                       src_node=p["src_node"], dst_node=p["dst_node"])
            for p in data.get("dep_patterns", [])
        ]
        return Pack(
            name=data["name"], version=str(data.get("version", "")), description=data.get("description", ""),
            entity_types=ets, relations=rels,
            regex_patterns=regex_patterns, dep_patterns=dep_patterns,
            datatype_properties=datatype_properties,
        )
    except KeyError as exc:
        raise OntologyError(f"malformed pack '{name_or_path}': missing key {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OntologyError(f"malformed pack '{name_or_path}': {exc}") from exc


def available_packs() -> List[str]:
    """Names of the shipped sample packs (backend strategy-catalog uses this vocabulary)."""
    return sorted(p.stem for p in _PACK_DIR.glob("*.json"))
