"""KG-AC-37: bounded, deterministic top_entities promotion — ranked by mention count, ties broken
by first-appearance order, never exceeding promote_top_n (hard max 20). Bulk entities never ride
the state plane; only this bounded summary does."""
import pytest

from core import promote_top_entities


def _rows(pairs):
    # pairs: list of (entity_type, surface_form) in appearance order
    return [{"entity_type": t, "surface_form": s} for t, s in pairs]


@pytest.mark.ac("KG-AC-37")
def test_ranked_by_mention_count_desc():
    rows = _rows([("Org", "A"), ("Org", "B"), ("Org", "A"), ("Org", "A"), ("Org", "B")])
    top = promote_top_entities(rows, promote_top_n=10)
    assert [t["surface_form"] for t in top] == ["A", "B"]  # A=3, B=2
    assert top[0]["mention_count"] == 3


@pytest.mark.ac("KG-AC-37")
def test_ties_broken_by_first_appearance():
    # A and B both appear twice; A appears first -> A ranks first.
    rows = _rows([("Org", "A"), ("Org", "B"), ("Org", "B"), ("Org", "A")])
    top = promote_top_entities(rows, promote_top_n=10)
    assert [t["surface_form"] for t in top] == ["A", "B"]


@pytest.mark.ac("KG-AC-37")
def test_never_exceeds_promote_top_n():
    rows = _rows([("Org", f"E{i}") for i in range(15)])
    top = promote_top_entities(rows, promote_top_n=5)
    assert len(top) == 5


@pytest.mark.ac("KG-AC-37")
def test_hard_max_caps_at_20():
    rows = _rows([("Org", f"E{i}") for i in range(30)])
    top = promote_top_entities(rows, promote_top_n=100)  # request > hard_max
    assert len(top) == 20


@pytest.mark.ac("KG-AC-37")
def test_deterministic_repeatable():
    rows = _rows([("Org", "A"), ("Person", "X"), ("Org", "A"), ("Person", "X"), ("Date", "2023")])
    a = promote_top_entities(rows, promote_top_n=10)
    b = promote_top_entities(rows, promote_top_n=10)
    assert a == b
