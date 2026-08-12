"""P10 (spec v13, KG-AC-77): ``aliases`` — the cluster's distinct surface forms EXCLUDING the chosen
``canonical_name``, ordered by descending mention count then alphabetically. Stated purpose: "a
reader can see every string that resolved to this instance" — auditability, not just display."""
import pytest

from core import Mention, choose_canonical_name


def _m(surface, doc=None, chunk=None, span=None):
    return Mention(entity_uid=f"u-{surface}-{doc}-{chunk}-{span}", entity_type="Organization",
                   surface_form=surface, source_doc_id=doc, source_chunk_id=chunk, span_start=span)


@pytest.mark.ac("KG-AC-77")
def test_aliases_exclude_the_chosen_canonical_name():
    _, aliases = choose_canonical_name([_m("Acme"), _m("Acme Corporation"), _m("Acme Corp")])
    assert "Acme Corporation" not in aliases
    assert set(aliases) == {"Acme", "Acme Corp"}


@pytest.mark.ac("KG-AC-77")
def test_aliases_ordered_by_descending_mention_count():
    mentions = ([_m("Acme Corporation")] +
               [_m("Acme")] * 3 +
               [_m("Acme Corp")] * 5)
    _, aliases = choose_canonical_name(mentions)
    assert aliases == ["Acme Corp", "Acme"]  # 5 mentions before 3


@pytest.mark.ac("KG-AC-77")
def test_aliases_tie_broken_alphabetically():
    mentions = [_m("Zebra Corporation"), _m("Acme"), _m("Beta")]  # Acme/Beta each 1 mention
    _, aliases = choose_canonical_name(mentions)
    assert aliases == ["Acme", "Beta"]


@pytest.mark.ac("KG-AC-77")
def test_single_mention_cluster_has_no_aliases():
    _, aliases = choose_canonical_name([_m("Solo Corp")])
    assert aliases == []


@pytest.mark.ac("KG-AC-77")
def test_every_distinct_surface_recoverable_from_name_plus_aliases():
    # the auditability property the AC exists for: canonical_name + aliases together must equal
    # EXACTLY the cluster's distinct surface set, nothing dropped, nothing invented.
    mentions = [_m("Acme"), _m("Acme Corporation"), _m("Acme Corp"), _m("ACME")]
    name, aliases = choose_canonical_name(mentions)
    assert {name, *aliases} == {"Acme", "Acme Corporation", "Acme Corp", "ACME"}
