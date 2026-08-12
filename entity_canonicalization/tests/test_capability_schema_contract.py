"""KG-AC-3 (worker side): the canonicalization CAPABILITY_SCHEMA is a well-formed v1 contract —
consumes entity_records (stage staged), produces entity_records (stage canonicalized). The DB parity
(== the seeded system_capabilities row) is proved in backend/tests/test_kg_registry.py + the F4 build
verified it byte-identical."""
import pytest

from capability_schema import CAPABILITY_SCHEMA


@pytest.mark.ac("KG-AC-3")
def test_worker_identity():
    assert CAPABILITY_SCHEMA["worker"] == "entity_canonicalization"
    assert CAPABILITY_SCHEMA["version"] == "1.0"


@pytest.mark.ac("KG-AC-3")
def test_staged_in_canonicalized_out():
    ins = CAPABILITY_SCHEMA["artifact_inputs"]["staged_entities"]
    outs = CAPABILITY_SCHEMA["artifact_outputs"]["canonicalized_entities"]
    assert ins["artifact_type"] == "entity_records" and ins["stage"] == "staged"
    assert outs["artifact_type"] == "entity_records" and outs["stage"] == "canonicalized"


@pytest.mark.ac("KG-AC-3")
def test_scalar_outputs_present():
    out = CAPABILITY_SCHEMA["output_fields"]
    for f in ("canonical_count", "merged_count", "minted_count"):
        assert out[f]["always_present"] is True


@pytest.mark.ac("KG-AC-82")
def test_canonicalized_entities_output_declares_the_canonical_graph_shape():
    # P16 (KG-AC-82): the contract must name the canonical-graph fields P10-P13 actually put on
    # kg_canonical_entities (canonical_key/canonical_name/aliases/attributes+status) -- before this
    # task the description said only "transitioned staged->canonicalized with canonical_id", which
    # was true at v11 but silently stale after P10-P13 landed.
    desc = CAPABILITY_SCHEMA["artifact_outputs"]["canonicalized_entities"]["description"]
    for term in ("canonical_key", "canonical_name", "aliases", "attributes",
                 "single_source", "consistent", "conflicting"):
        assert term in desc, f"canonical-graph shape term {term!r} missing from the output contract"
