"""KG-AC-4 (worker side): the kg_export CAPABILITY_SCHEMA is a well-formed v1 contract — consumes
entity_records (stage canonicalized), writes to an external store (no CB artifact_output). DB parity
(== the seeded row) is proved in backend/tests/test_kg_registry.py + verified byte-identical at build."""
import pytest

from capability_schema import CAPABILITY_SCHEMA


@pytest.mark.ac("KG-AC-4")
def test_worker_identity():
    assert CAPABILITY_SCHEMA["worker"] == "kg_export"
    assert CAPABILITY_SCHEMA["version"] == "1.0"


@pytest.mark.ac("KG-AC-4")
def test_consumes_canonicalized_entity_records_no_artifact_output():
    ins = CAPABILITY_SCHEMA["artifact_inputs"]["canonicalized_entities"]
    assert ins["artifact_type"] == "entity_records" and ins["stage"] == "canonicalized"
    assert "artifact_outputs" not in CAPABILITY_SCHEMA  # external Neo4j — no CB artifact out


@pytest.mark.ac("KG-AC-4")
def test_scalar_outputs_present():
    out = CAPABILITY_SCHEMA["output_fields"]
    assert out["node_count"]["always_present"] is True
    assert out["relationship_count"]["always_present"] is True
