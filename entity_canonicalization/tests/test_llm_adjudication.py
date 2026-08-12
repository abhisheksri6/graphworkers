"""KG-AC-35: the ambiguous match band is adjudicated by the LLM via the canonicalization connection;
each call emits a usage[] entry; and if an ambiguous pair is reached with NO connection configured
AND no default resolves either, canonicalization fails LOUD (never a silent skip). Pure (fake client).

**Amended 2026-08-12 (owner-directed, same day as the live incident this fixes):** the canon
node's `connection_id` was previously the ONLY source — absent it, every ambiguous pair failed the
whole batch, and there is currently no node-UI surface to set it (deliberately left that way for
now, per the owner). `_resolve_connection_id` now falls back to `settings.default_llm_connection_id`
(env-overridable, defaults to the same "AWS LLM Connection" profile `entity_extraction` already
uses successfully) so a NEW pipeline works out of the box without hand-editing its inline JSON.
KG-AC-35's own "loud failure if absent" clause still holds — it now means "if absent AND the
default itself resolves to nothing" (an operator can force the old behavior by setting the env var
to an empty string)."""
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


# ---- KG-AC-35 amended (2026-08-12): connection_id defaulting -----------------------------------
@pytest.mark.ac("KG-AC-35")
def test_resolve_connection_id_uses_explicit_config_value_when_present():
    # An explicit connection_id in the profile always wins over the default -- the pipeline's own
    # choice is never silently overridden.
    assert _resolve_connection_id({"connection_id": "Some Other Connection"}) == "Some Other Connection"


@pytest.mark.ac("KG-AC-35")
def test_resolve_connection_id_falls_back_to_the_configured_default_when_absent():
    # No connection_id in the profile at all (the current, no-UI-surface reality for new pipelines)
    # -> falls back to settings.default_llm_connection_id, not None.
    assert _resolve_connection_id({}) == settings.default_llm_connection_id
    assert _resolve_connection_id({"fuzzy_floor": 0.8}) == settings.default_llm_connection_id


@pytest.mark.ac("KG-AC-35")
def test_resolve_connection_id_treats_empty_string_the_same_as_absent():
    # A profile carrying connection_id="" (e.g. a stale/blanked designer field) is not a deliberate
    # choice of "no connection" -- falls back the same as a missing key, same posture as `cfg.get(x)
    # or default` treats any falsy value.
    assert _resolve_connection_id({"connection_id": ""}) == settings.default_llm_connection_id


@pytest.mark.ac("KG-AC-35")
def test_resolve_connection_id_default_env_value_is_the_known_working_aws_connection():
    # Pins the actual default (not just "truthy") -- this is the exact profile name
    # entity_extraction's own pipeline node already uses successfully (verified live 2026-08-12).
    assert settings.default_llm_connection_id == "AWS LLM Connection"
