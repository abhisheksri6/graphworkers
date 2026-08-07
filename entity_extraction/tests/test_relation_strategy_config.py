"""KG-AC-59 (worker side): ``relation_strategy`` is an opt-in enum {generate, classify}, default
``generate``, decoupled from ``engine`` (ADR-0014 — the L1 config contract for Workstream L)."""
import pytest

from strategies.base import ExtractionConfig


@pytest.mark.ac("KG-AC-59")
def test_default_relation_strategy_is_generate():
    assert ExtractionConfig().relation_strategy == "generate"


@pytest.mark.ac("KG-AC-59")
def test_from_dict_default_relation_strategy_is_generate():
    # the wire-contract path (worker task payload -> ExtractionConfig.from_dict): a config with no
    # relation_strategy key must resolve to the default, same as every other optional field here.
    config = ExtractionConfig.from_dict({"engine": "llm", "ontology_pack": "fibo_core"})
    assert config.relation_strategy == "generate"


@pytest.mark.ac("KG-AC-59")
def test_from_dict_relation_strategy_classify_round_trips():
    config = ExtractionConfig.from_dict(
        {"engine": "llm", "ontology_pack": "fibo_core", "relation_strategy": "classify"})
    assert config.relation_strategy == "classify"


@pytest.mark.ac("KG-AC-59")
def test_relation_strategy_decoupled_from_engine():
    # classify is legal alongside engine=spacy too -- it classifies over whatever entities the
    # enabled layers produced, not tied to the LLM generate-mode one-pass call (ADR-0014).
    config = ExtractionConfig.from_dict(
        {"engine": "spacy", "ontology_pack": "generic", "relation_strategy": "classify"})
    assert config.engine == "spacy" and config.relation_strategy == "classify"
