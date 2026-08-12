"""KG-AC-35: the ambiguous match band is adjudicated by the LLM via the canonicalization connection;
each call emits a usage[] entry; and if no connection resolves at all, canonicalization fails LOUD
(never a silent skip). Pure (fake client).

**Amended 2026-08-12 (P23, owner-directed, same day as the live incident this fixes):**
`connection_id` was previously the pipeline node's own inline config, absent which every ambiguous
pair failed the whole batch — and there is currently no node-UI surface to set it (deliberately
left that way for now, per the owner). `_resolve_connection_id` fell back to
`settings.default_llm_connection_id` when absent.

**Amended again same day (P24, owner-directed):** every pipeline found with an EXPLICIT
`connection_id` had it WRONG (pointed at a non-Bedrock connection — the actual production
incident). Since there is no UI surface to ever set this field correctly, and the field has never
once been observed holding a correct value, `settings.default_llm_connection_id` now ALWAYS wins
— not just a fallback when absent, an override even when the pipeline's inline config sets
something else. Set the env var to `""` to go back to reading the pipeline's own `connection_id`
(fails loud if that's absent too, via `_no_connection_adjudicator`)."""
import pytest

from core import Mention, normalize_surface
from entity_canonicalization_worker import (
    _no_connection_adjudicator, _resolve_connection_id, make_llm_adjudicator, settings,
)


def _m(uid, etype, surface):
    m = Mention(entity_uid=uid, entity_type=etype, surface_form=surface)
    m.normalized_form = normalize_surface(surface)
    return m


class _FakeLlm:
    def __init__(self, answer):
        self._answer = answer
        self.usage = []

    def complete(self, prompt):
        self.usage.append({"charge_category": "llm", "model": "amazon.nova-pro-v1:0",
                           "input_tokens": 20, "output_tokens": 1, "connection_id": "c"})
        return self._answer


@pytest.mark.ac("KG-AC-35")
def test_adjudicator_yes_merges_and_captures_usage():
    client = _FakeLlm("yes")
    adjudicate = make_llm_adjudicator(client)
    result = adjudicate(_m("1", "Organization", "Acme Systems"), _m("2", "Organization", "Acme Solutions"))
    assert result is True
    assert len(client.usage) == 1 and client.usage[0]["charge_category"] == "llm"


@pytest.mark.ac("KG-AC-35")
def test_adjudicator_no_does_not_merge():
    assert make_llm_adjudicator(_FakeLlm("no"))(
        _m("1", "Organization", "Acme Systems"), _m("2", "Organization", "Beta Corp")) is False


@pytest.mark.ac("KG-AC-35")
def test_no_connection_adjudicator_fails_loud():
    with pytest.raises(RuntimeError):
        _no_connection_adjudicator(_m("1", "Org", "A"), _m("2", "Org", "B"))


# ---- KG-AC-35 amended twice (2026-08-12, P23 then P24): connection_id resolution ---------------
@pytest.mark.ac("KG-AC-35")
def test_resolve_connection_id_the_configured_default_overrides_explicit_config():
    # P24: the configured default now ALWAYS wins, even over a pipeline's own explicit
    # connection_id -- every explicit value found in production had it wrong (no UI surface to set
    # it correctly), so the pipeline's own choice is no longer trusted.
    assert _resolve_connection_id({"connection_id": "Some Other Connection"}) == settings.default_llm_connection_id
    assert _resolve_connection_id({"connection_id": "Ollama LLM Profile"}) == settings.default_llm_connection_id


@pytest.mark.ac("KG-AC-35")
def test_resolve_connection_id_uses_the_default_when_config_is_absent_too():
    # No connection_id in the profile at all (the current, no-UI-surface reality for new pipelines)
    # -> the same default, not None.
    assert _resolve_connection_id({}) == settings.default_llm_connection_id
    assert _resolve_connection_id({"fuzzy_floor": 0.8}) == settings.default_llm_connection_id


@pytest.mark.ac("KG-AC-35")
def test_resolve_connection_id_falls_back_to_explicit_config_only_when_default_is_blanked():
    # The escape hatch: an operator who sets the env var to "" gets the PRE-P24 behavior back --
    # the pipeline's own connection_id is read again (and still fails loud if that's absent too,
    # via _no_connection_adjudicator -- covered by test_no_connection_adjudicator_fails_loud above).
    settings.default_llm_connection_id = ""
    try:
        assert _resolve_connection_id({"connection_id": "Some Other Connection"}) == "Some Other Connection"
        assert _resolve_connection_id({}) is None
    finally:
        settings.default_llm_connection_id = "AWS LLM Connection"


@pytest.mark.ac("KG-AC-35")
def test_resolve_connection_id_default_env_value_is_the_known_working_aws_connection():
    # Pins the actual default (not just "truthy") -- this is the exact profile name
    # entity_extraction's own pipeline node already uses successfully (verified live 2026-08-12).
    assert settings.default_llm_connection_id == "AWS LLM Connection"
