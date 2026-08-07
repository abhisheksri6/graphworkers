"""KG-AC-57 (evolve v8): empty/absent `entity_types` extracts ALL of the pack's types (no subset
filter applied); a non-empty list narrows to exactly that subset. This is a LOCK-IN
(characterization) test — it passes on the CURRENT code (the falsy-check at
`strategies/base.py::filter_closed_vocab`, `subset = set(config.entity_types) if config.entity_types
else None`, already treats both `[]` and `None` as "no filter") and exists to prevent a future
refactor from silently narrowing "empty" to mean "nothing" instead of "everything" — per the
cb-implement rule, the strict red-first-must-fail requirement does not apply to a lock-in test."""
import pytest

from core import Candidate
from ontologies import load_pack
from strategies.base import ExtractionConfig, filter_closed_vocab


def _candidates():
    pack = load_pack("fibo_core")
    return [
        Candidate("Acme Bank", "Bank", "c1", "spacy", 0, 9),
        Candidate("Jane Roe", "Person", "c1", "spacy", 20, 28),
        Candidate("2026-08-06", "Date", "c1", "spacy", 40, 50),
    ]


@pytest.mark.ac("KG-AC-57")
def test_absent_entity_types_extracts_all_pack_types():
    pack = load_pack("fibo_core")
    config = ExtractionConfig(entity_types=None)
    kept, _unmapped = filter_closed_vocab(_candidates(), config, pack)
    assert {c.entity_type for c in kept} == {"Bank", "Person", "Date"}


@pytest.mark.ac("KG-AC-57")
def test_empty_list_entity_types_extracts_all_pack_types():
    pack = load_pack("fibo_core")
    config = ExtractionConfig(entity_types=[])
    kept, _unmapped = filter_closed_vocab(_candidates(), config, pack)
    assert {c.entity_type for c in kept} == {"Bank", "Person", "Date"}


@pytest.mark.ac("KG-AC-57")
def test_nonempty_entity_types_narrows_to_the_subset():
    pack = load_pack("fibo_core")
    config = ExtractionConfig(entity_types=["Bank"])
    kept, _unmapped = filter_closed_vocab(_candidates(), config, pack)
    assert {c.entity_type for c in kept} == {"Bank"}


@pytest.mark.ac("KG-AC-57")
def test_from_dict_default_entity_types_is_none():
    # the wire-contract path (worker task payload -> ExtractionConfig.from_dict): an entity_types
    # key omitted entirely from the config dict must ALSO resolve to "extract everything".
    config = ExtractionConfig.from_dict({"engine": "spacy", "ontology_pack": "fibo_core"})
    assert config.entity_types is None
