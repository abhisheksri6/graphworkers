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
