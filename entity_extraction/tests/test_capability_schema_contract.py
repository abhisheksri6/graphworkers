"""KG-AC-21 (worker side): the worker's CAPABILITY_SCHEMA is a well-formed v1 contract that
declares entity_extraction as an ARTIFACT-plane producer — consumes text_chunks, produces the
new entity_records type (stage staged).

The other half of KG-AC-21 ("identical to the seeded system_capabilities row") is F1/B2: the
migration embeds this exact schema and a byte-parity check (scratchpad/kg_c2_parity.py, run at
build time) confirms worker dict == DB row. This file proves the worker-side contract offline,
without a broker or DB (import-light module).
"""
import pytest

from capability_schema import CAPABILITY_SCHEMA


@pytest.mark.ac("KG-AC-21")
def test_worker_identity_matches_naming_standard():
    # variant-None: worker == queue == capability_type == 'entity_extraction' (WORKER_NAMING).
    assert CAPABILITY_SCHEMA["worker"] == "entity_extraction"
    assert CAPABILITY_SCHEMA["version"] == "1.0"


@pytest.mark.ac("KG-AC-21")
def test_artifact_bindings_text_chunks_in_entity_records_out():
    inputs = CAPABILITY_SCHEMA["artifact_inputs"]
    outputs = CAPABILITY_SCHEMA["artifact_outputs"]
    # consumes text_chunks (required slot)
    assert inputs["chunks"]["artifact_type"] == "text_chunks"
    assert inputs["chunks"]["required"] == "always"
    # produces the NEW entity_records type at stage 'staged' (ADR-0007)
    assert outputs["entities"]["artifact_type"] == "entity_records"
    assert outputs["entities"]["stage"] == "staged"
    assert outputs["entities"]["always_present"] is True


@pytest.mark.ac("KG-AC-21")
def test_required_input_fields_declared():
    inp = CAPABILITY_SCHEMA["input_fields"]
    assert inp["folder_id"]["required"] == "always"
    assert inp["folder_id"]["implicit"] is True
    assert inp["entity_extraction_config.engine"]["required"] == "always"
    assert inp["entity_extraction_config.ontology_pack"]["required"] == "always"


@pytest.mark.ac("KG-AC-21")
def test_state_scalar_outputs_present():
    out = CAPABILITY_SCHEMA["output_fields"]
    # the KG-AC-9 state-plane scalar summary the callback promotes
    for field in (
        "entity_count", "edge_count", "distinct_types",
        "ontology_pack", "ontology_version", "unmapped_type_count", "ungrounded_relation_count",
        "self_consistency_votes", "chunk_metadata_missing_count", "unresolved_reference_count",
        "unlocatable_entity_count", "unmapped_property_count", "ungrounded_fact_count",
        "guardrails_blocked_facts",
    ):
        assert out[field]["always_present"] is True, field
    # top_entities is bounded top-N and may be empty -> not always_present
    assert out["top_entities"]["always_present"] is False


# ---- v11 (spec v11, KG-AC-9/12): gazetteer config surface + linked_count withdrawn ----
@pytest.mark.ac("KG-AC-12")
def test_gazetteer_input_fields_absent():
    inp = CAPABILITY_SCHEMA["input_fields"]
    for key in (
        "entity_extraction_config.gazetteer_enabled",
        "entity_extraction_config.gazetteer_sources",
        "entity_extraction_config.entity_linking",
    ):
        assert key not in inp, key


@pytest.mark.ac("KG-AC-9")
def test_linked_count_output_field_absent():
    assert "linked_count" not in CAPABILITY_SCHEMA["output_fields"]


# ---- KG-AC-45 (evolve v5): relation_engine field removed, coreference_enabled added ----
@pytest.mark.ac("KG-AC-45")
def test_relation_engine_field_absent():
    assert "entity_extraction_config.relation_engine" not in CAPABILITY_SCHEMA["input_fields"]


@pytest.mark.ac("KG-AC-45")
def test_coreference_enabled_field_present():
    field = CAPABILITY_SCHEMA["input_fields"]["entity_extraction_config.coreference_enabled"]
    assert field["type"] == "boolean"
    assert field["required"] == "optional"


# ---- J2 (spec v8): composable-layer toggles — ADR-0012, KG-AC-53 -----------
@pytest.mark.ac("KG-AC-53")
def test_composable_layer_toggles_present_and_optional():
    inp = CAPABILITY_SCHEMA["input_fields"]
    for key in (
        "entity_extraction_config.rules_entities_enabled",
        "entity_extraction_config.rules_relations_enabled",
    ):
        field = inp[key]
        assert field["type"] == "boolean"
        assert field["required"] == "optional"  # opt-in — clarify 2026-08-06, defaults false


# ---- L1 (spec v10): relation_strategy config — ADR-0014, KG-AC-59 ----------
@pytest.mark.ac("KG-AC-59")
def test_relation_strategy_field_present_optional_enum():
    field = CAPABILITY_SCHEMA["input_fields"]["entity_extraction_config.relation_strategy"]
    assert field["type"] == "string"
    assert field["required"] == "optional"
    assert "generate" in field["description"].lower()
    assert "classify" in field["description"].lower()


# ---- N3/N4 (spec v12): entity_scoped mode + self-consistency voting — KG-AC-65/67/68 -------
@pytest.mark.ac("KG-AC-65")
def test_relation_strategy_description_names_entity_scoped():
    field = CAPABILITY_SCHEMA["input_fields"]["entity_extraction_config.relation_strategy"]
    assert "entity_scoped" in field["description"].lower()


@pytest.mark.ac("KG-AC-67")
def test_relation_self_consistency_k_field_present_optional_bounded():
    field = CAPABILITY_SCHEMA["input_fields"]["entity_extraction_config.relation_self_consistency_k"]
    assert field["type"] == "integer"
    assert field["required"] == "optional"
    assert "1" in field["description"] and "5" in field["description"]


@pytest.mark.ac("KG-AC-57")
def test_entity_types_field_description_carries_empty_means_all_hint():
    # bug found + fixed live 2026-08-06: the UI hint was only ever added to the backend
    # strategy-catalog under the wrong key ("help", which the FE adapter ignores) — the field the
    # FE actually renders is THIS description, on the canonical (KG-AC-21-parity) capability_schema.
    field = CAPABILITY_SCHEMA["input_fields"]["entity_extraction_config.entity_types"]
    assert "empty" in field["description"].lower()
