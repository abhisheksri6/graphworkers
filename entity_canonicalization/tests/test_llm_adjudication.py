"""KG-AC-35: the ambiguous match band is adjudicated by the LLM via the canonicalization connection;
each call emits a usage[] entry; and if an ambiguous pair is reached with NO connection configured,
canonicalization fails LOUD (never a silent skip). Pure (fake client)."""
import pytest

from core import Mention, normalize_surface
from entity_canonicalization_worker import _no_connection_adjudicator, make_llm_adjudicator


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
