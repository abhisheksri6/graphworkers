"""KG-AC-8 (thin callback: counts + top-N only, NO entity/edge lists) and KG-AC-9 (per-folder state
scalars in folders[] for the M1 run_state merge). The success payload is asserted byte-identical to
the shared canonical fixture `tests/fixtures/callback_success.json` — the SAME fixture the backend
M1–M4 handler test (B4) loads, keeping worker and backend in lockstep.
"""
import json
from pathlib import Path

import pytest

from callback import build_callback, flatten_state

_FIXTURE = Path(__file__).parent / "fixtures" / "callback_success.json"

# The exact inputs the shared fixture is built from (mirrored in the backend B4 test).
_SUMMARY = {
    "entity_count": 3,
    "edge_count": 1,
    "distinct_types": 2,
    "top_entities": [{"entity_type": "Organization", "surface_form": "Acme Corp", "mention_count": 2}],
    "ontology_pack": "fibo_core",
    "ontology_version": "1.0",
    "unmapped_type_count": 1,
    "ungrounded_relation_count": 0,
}
_USAGE = [{"charge_category": "llm", "model": "amazon.nova-pro-v1:0",
           "input_tokens": 100, "output_tokens": 50, "connection_id": "conn-1"}]


def _build():
    return build_callback("task-123", "folder-789", "kg_pipeline", "run-456",
                          status="success", summary=_SUMMARY, usage=_USAGE)


@pytest.mark.ac("KG-AC-8")
def test_success_payload_matches_shared_fixture():
    expected = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert _build() == expected


@pytest.mark.ac("KG-AC-8")
def test_callback_carries_no_bulk_entity_or_edge_list():
    payload = _build()
    # thin: counts + top-N only; the bulk rows live in Postgres, never on the wire
    assert "entities" not in payload
    assert "edges" not in payload
    assert set(payload["top_entities"][0]) == {"entity_type", "surface_form", "mention_count"}


@pytest.mark.ac("KG-AC-9")
def test_folders_carry_state_scalars_for_m1_merge():
    payload = _build()
    assert len(payload["folders"]) == 1
    state = payload["folders"][0]
    assert state["folder_id"] == "folder-789"
    for field in ("entity_count", "edge_count", "distinct_types", "top_entities",
                  "ontology_pack", "ontology_version", "unmapped_type_count",
                  "ungrounded_relation_count"):
        assert field in state, field
    assert state["entity_count"] == 3


@pytest.mark.ac("KG-AC-9")
def test_failure_callback_is_loud_and_carries_no_summary():
    payload = build_callback("t", "f", "d", "r", status="failed", error_message="no chunks for folder f")
    assert payload["status"] == "failed"
    assert payload["error_message"] == "no chunks for folder f"
    assert "folders" not in payload and "entity_count" not in payload


@pytest.mark.ac("KG-AC-9")
def test_flatten_state_shape():
    st = flatten_state("f1", _SUMMARY)
    assert st["folder_id"] == "f1"
    assert st["entity_count"] == 3
    assert "linked_count" not in st  # v11: withdrawn with the gazetteer capability (KG-AC-9)
