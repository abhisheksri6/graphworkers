"""P7 (spec v13, KG-AC-84): guardrails screening extended to facts. Facts pass through the SAME
once-per-batch guardrails call entity candidates already use (KG-AC-17's posture) — never a second
guardrails_check call. A blocked fact is dropped and counted (`guardrails_blocked_facts`); the task
still succeeds; entity screening behaviour is unchanged."""
import pytest

from ontologies import DatatypeProperty, EntityType, Pack
from strategies.base import Chunk, ExtractionConfig, run_pipeline


def _agreement_pack():
    return Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("Agreement", None, [], "", None)],
        relations=[],
        datatype_properties=[DatatypeProperty("agreementId", "Agreement", "identifier", "")],
    )


class _Llm:
    resolved_model = "amazon.nova-pro-v1:0"
    usage = []

    def complete_tool(self, **_kwargs):
        return {
            "entities": [{"type": "Agreement", "surface": "The Agreement"}],
            "relations": [],
            "facts": [{"subject_id": 0, "property": "agreementId", "value": "IMA-2025-018",
                       "evidence": "The Agreement IMA-2025-018 is entered."}],
        }


_CHUNK = Chunk("c1", "The Agreement IMA-2025-018 is entered.")


@pytest.mark.ac("KG-AC-84")
def test_guardrails_screen_called_once_with_both_candidates_and_facts():
    calls = []

    def screen(items):
        calls.append(list(items))
        return [True] * len(items)

    run_pipeline([_CHUNK], ExtractionConfig(engine="llm", ontology_pack="x"), _agreement_pack(),
                folder_id="f", llm_client=_Llm(), guardrails_screen=screen)
    assert len(calls) == 1  # KG-AC-17's once-per-batch posture — never a second call for facts
    assert len(calls[0]) == 2  # 1 entity candidate + 1 fact, screened together


@pytest.mark.ac("KG-AC-84")
def test_blocked_fact_is_dropped_and_counted_task_still_succeeds():
    # block only the fact (2nd item); keep the entity candidate (1st item)
    def screen(items):
        return [True, False]

    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [_CHUNK], ExtractionConfig(engine="llm", ontology_pack="x"), _agreement_pack(),
        folder_id="f", llm_client=_Llm(), guardrails_screen=screen,
    )
    assert summary["guardrails_blocked_facts"] == 1
    assert ent_rows[0]["attributes"] == []  # the blocked fact never attaches
    assert blocked == 0  # the entity candidate was NOT blocked


@pytest.mark.ac("KG-AC-84")
def test_blocked_entity_candidate_still_counted_same_as_before():
    def screen(items):
        return [False, True]  # block the entity, keep the fact

    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [_CHUNK], ExtractionConfig(engine="llm", ontology_pack="x"), _agreement_pack(),
        folder_id="f", llm_client=_Llm(), guardrails_screen=screen,
    )
    assert blocked == 1  # entity screening behaviour unchanged
    assert summary["guardrails_blocked_facts"] == 0
    assert summary["entity_count"] == 0


@pytest.mark.ac("KG-AC-84")
def test_no_guardrails_screen_means_zero_blocked_facts():
    ent_rows, edge_rows, summary, usage, blocked = run_pipeline(
        [_CHUNK], ExtractionConfig(engine="llm", ontology_pack="x"), _agreement_pack(),
        folder_id="f", llm_client=_Llm(),
    )
    assert summary["guardrails_blocked_facts"] == 0
    assert len(ent_rows[0]["attributes"]) == 1
