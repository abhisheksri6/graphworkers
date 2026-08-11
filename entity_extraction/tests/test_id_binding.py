"""P4 (spec v13, KG-AC-71): identifier-based binding. A relation item references an entity by
`src_id`/`dst_id` — the 0-based position of that entity within the SAME LLM call's own response —
never a re-typed `src`/`src_type`/`dst`/`dst_type` surface pair. This is the fix for a real,
previously-silent, previously-uncounted failure mode: any drift in the model's re-stated surface
form (case, whitespace, a paraphrase) used to make `build_edge_records`'s `(chunk, type, surface)`
lookup miss, silently dropping the edge. Under id-based binding the surface/type are copied
directly from the `Candidate` the strategy already built — they can no longer mismatch.

Covers all three relation-producing strategies: `LlmGraphStrategy` (generate — entities AND
relations from one call, ids reference THIS response's own entities[]), `LlmClassifyStrategy`
(classify — already id-based via `pair_index` into a deterministic pre-computed batch; only the
counter was missing), `LlmEntityScopedStrategy` (entity_scoped — ids reference the GIVEN,
already-known entity list, not a response the model itself produced).
"""
import pytest

from candidate_pairs import CandidatePair
from core import Candidate
from ontologies import load_pack
from strategies.llm_classify import LlmClassifyStrategy
from strategies.llm_entity_scoped import LlmEntityScopedStrategy
from strategies.llm_graph import LlmGraphStrategy
from strategies.base import Chunk, ExtractionConfig

FIBO = load_pack("fibo_core")


class _FakeLlmClient:
    def __init__(self, response):
        self._response = response
        self.calls = 0

    def complete_tool(self, *, system_text, user_text, tool_name, tool_description, tool_schema):
        self.calls += 1
        return self._response


# ---- LlmGraphStrategy (generate mode) --------------------------------------------------------
@pytest.mark.ac("KG-AC-71")
def test_llm_graph_binds_by_id_even_when_model_would_have_mismatched_surface():
    # the model's relation item carries NO surface/type at all -- it can only reference position.
    # If binding still depended on a re-typed surface, this would be untestable as a mismatch case;
    # id-based binding makes the class of bug structurally impossible instead.
    response = {
        "entities": [
            {"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
            {"type": "Bond", "surface": "Acme 5% 2030", "confidence": 0.8},
        ],
        "relations": [
            {"type": "issues", "src_id": 0, "dst_id": 1, "evidence": "Acme Corp issues Acme 5% 2030."},
        ],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    strat.extract([Chunk("c1", "Acme Corp issues Acme 5% 2030.")], ExtractionConfig(engine="llm"), FIBO)
    assert len(strat.relations) == 1
    r = strat.relations[0]
    assert r.src_surface == "Acme Corp" and r.src_type == "Organization"
    assert r.dst_surface == "Acme 5% 2030" and r.dst_type == "Bond"
    assert strat.unresolved_reference_count == 0


@pytest.mark.ac("KG-AC-71")
def test_llm_graph_id_outside_response_own_entities_is_unresolved_and_counted():
    response = {
        "entities": [{"type": "Organization", "surface": "Acme Corp", "confidence": 0.9}],
        "relations": [
            # dst_id=1 doesn't exist -- only ONE entity (index 0) was returned in this response.
            {"type": "issues", "src_id": 0, "dst_id": 1, "evidence": "Acme Corp issues something."},
        ],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    strat.extract([Chunk("c1", "Acme Corp issues something.")], ExtractionConfig(engine="llm"), FIBO)
    assert strat.relations == []
    assert strat.unresolved_reference_count == 1


@pytest.mark.ac("KG-AC-71")
def test_llm_graph_index_is_raw_response_position_not_post_filter_position():
    # entities[0] is invalid (no surface) and skipped from the Candidate list, so entities[1]'s
    # RAW position is 1 -- the same position the model was told to use, even though it is the
    # FIRST item in the filtered `entities` list the strategy builds internally.
    response = {
        "entities": [
            {"type": "Organization", "surface": None, "confidence": 0.9},  # invalid, skipped
            {"type": "Bond", "surface": "Acme 5% 2030", "confidence": 0.8},
            {"type": "Person", "surface": "Jane Roe", "confidence": 0.7},
        ],
        "relations": [
            {"type": "hasRating", "src_id": 1, "dst_id": 2, "evidence": "text"},
        ],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    # chunk text must literally contain both surfaces so KG-AC-72's unlocatable-span drop doesn't
    # remove them before this test's own id-position assertion is ever reached.
    strat.extract([Chunk("c1", "Acme 5% 2030 has a rating held by Jane Roe.")],
                  ExtractionConfig(engine="llm"), FIBO)
    assert len(strat.relations) == 1
    assert strat.relations[0].src_surface == "Acme 5% 2030"
    assert strat.relations[0].dst_surface == "Jane Roe"
    assert strat.unresolved_reference_count == 0
    assert strat.unlocatable_entity_count == 0


@pytest.mark.ac("KG-AC-71")
def test_llm_graph_id_referencing_a_skipped_raw_item_is_unresolved():
    response = {
        "entities": [
            {"type": "Organization", "surface": None, "confidence": 0.9},  # invalid, skipped
            {"type": "Bond", "surface": "Acme 5% 2030", "confidence": 0.8},
        ],
        "relations": [
            {"type": "issues", "src_id": 0, "dst_id": 1, "evidence": "text"},  # src_id=0 was skipped
        ],
    }
    strat = LlmGraphStrategy(llm_client=_FakeLlmClient(response))
    strat.extract([Chunk("c1", "text")], ExtractionConfig(engine="llm"), FIBO)
    assert strat.relations == []
    assert strat.unresolved_reference_count == 1


# ---- LlmClassifyStrategy (classify mode) -----------------------------------------------------
def _pair(chunk_id="c1", src="Acme Corp", src_type="Organization", dst="Acme 5% 2030",
          dst_type="Bond", allowed=("issues",), sentence_span=(0, 30)):
    return CandidatePair(
        chunk_id=chunk_id, src_surface=src, src_type=src_type, src_span=(0, 9),
        dst_surface=dst, dst_type=dst_type, dst_span=(20, 28),
        sentence_span=sentence_span, allowed_relation_types=list(allowed),
    )


@pytest.mark.ac("KG-AC-71")
def test_llm_classify_out_of_range_pair_index_is_unresolved_and_counted():
    client = _FakeLlmClient({"labels": [
        {"pair_index": 5, "relation_type": "issues", "evidence": "text"},  # only 1 pair given (index 0)
    ]})
    strat = LlmClassifyStrategy(llm_client=client)
    relations = strat.classify([_pair()], {"c1": "Acme Corp issues Acme 5% 2030."})
    assert relations == []
    assert strat.unresolved_reference_count == 1


@pytest.mark.ac("KG-AC-71")
def test_llm_classify_valid_pair_index_resolves_and_is_not_counted():
    client = _FakeLlmClient({"labels": [
        {"pair_index": 0, "relation_type": "issues", "evidence": "text"},
    ]})
    strat = LlmClassifyStrategy(llm_client=client)
    relations = strat.classify([_pair()], {"c1": "Acme Corp issues Acme 5% 2030."})
    assert len(relations) == 1
    assert strat.unresolved_reference_count == 0


# ---- LlmEntityScopedStrategy (entity_scoped mode) --------------------------------------------
@pytest.mark.ac("KG-AC-71")
def test_llm_entity_scoped_binds_by_id_from_the_given_entity_list():
    entities = [
        Candidate(surface_form="Acme Corp", entity_type="Organization", source_chunk_id="c1", layer="llm"),
        Candidate(surface_form="Jane Roe", entity_type="Person", source_chunk_id="c1", layer="llm"),
    ]
    client = _FakeLlmClient({"relations": [
        {"type": "employs", "src_id": 0, "dst_id": 1, "evidence": "Acme Corp employs Jane Roe."},
    ]})
    strat = LlmEntityScopedStrategy(llm_client=client)
    relations = strat.extract(entities, {"c1": "Acme Corp employs Jane Roe."}, FIBO)
    assert len(relations) == 1
    assert relations[0].src_surface == "Acme Corp" and relations[0].src_type == "Organization"
    assert relations[0].dst_surface == "Jane Roe" and relations[0].dst_type == "Person"
    assert strat.unresolved_reference_count == 0


@pytest.mark.ac("KG-AC-71")
def test_llm_entity_scoped_id_outside_given_entities_is_unresolved_and_counted():
    # >= 2 entities so the call actually happens (KG-AC-65/33: <2 entities makes no LLM call at all).
    entities = [
        Candidate(surface_form="Acme Corp", entity_type="Organization", source_chunk_id="c1", layer="llm"),
        Candidate(surface_form="Jane Roe", entity_type="Person", source_chunk_id="c1", layer="llm"),
    ]
    client = _FakeLlmClient({"relations": [
        {"type": "employs", "src_id": 0, "dst_id": 7, "evidence": "text"},  # dst_id=7 doesn't exist
    ]})
    strat = LlmEntityScopedStrategy(llm_client=client)
    relations = strat.extract(entities, {"c1": "text"}, FIBO)
    assert relations == []
    assert strat.unresolved_reference_count == 1
