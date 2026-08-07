"""KG-AC-13 (strategy side): the opt-in gazetteer writes a generalized external_id (LEI) +
external_id_source='gleif_lei' when a mention matches AND entity_linking is on AND the pack type may
carry an external id. Opt-in: default off; enabled-but-no-index is a non-fatal no-op. The real
by-copy GLEIF/LEI SQLite index + the CC0-only / no-CUSIP-ISIN guarantee are C6/C12 (needs the data)."""
import pytest

from ontologies import load_pack
from strategies import Chunk, ExtractionConfig, GazetteerHit, RulesGazetteerStrategy

FIBO = load_pack("fibo_core")


class _FakeGazetteer:
    def __init__(self, hits):
        self._hits = hits

    def find(self, text):
        return list(self._hits)


def _cfg(**kw):
    return ExtractionConfig(engine="spacy", ontology_pack="fibo_core", gazetteer_enabled=True, **kw)


@pytest.mark.ac("KG-AC-13")
def test_external_id_written_when_linking_on():
    gaz = _FakeGazetteer([GazetteerHit("Acme Bank NA", 0, 12, "Bank", lei="5493001KJTIIGC8Y1R12")])
    cands = RulesGazetteerStrategy(gaz).extract([Chunk("c1", "Acme Bank NA ...")],
                                                _cfg(entity_linking=True), FIBO)
    assert len(cands) == 1
    c = cands[0]
    assert c.layer == "gazetteer"
    assert c.entity_type == "Bank"
    assert c.external_id == "5493001KJTIIGC8Y1R12"
    assert c.external_id_source == "gleif_lei"


@pytest.mark.ac("KG-AC-13")
def test_no_external_id_when_linking_off():
    gaz = _FakeGazetteer([GazetteerHit("Acme Bank NA", 0, 12, "Bank", lei="5493001KJTIIGC8Y1R12")])
    cands = RulesGazetteerStrategy(gaz).extract([Chunk("c1", "Acme Bank NA ...")],
                                                _cfg(entity_linking=False), FIBO)
    assert cands[0].external_id is None
    assert cands[0].external_id_source is None


@pytest.mark.ac("KG-AC-13")
def test_opt_in_no_index_is_noop():
    # gazetteer_enabled but no index injected -> non-fatal empty result
    cands = RulesGazetteerStrategy(gazetteer=None).extract([Chunk("c1", "x")], _cfg(), FIBO)
    assert cands == []


@pytest.mark.ac("KG-AC-13")
def test_unknown_type_hit_skipped():
    gaz = _FakeGazetteer([GazetteerHit("Weird", 0, 5, "NotAPackType", lei="X")])
    cands = RulesGazetteerStrategy(gaz).extract([Chunk("c1", "Weird")], _cfg(entity_linking=True), FIBO)
    assert cands == []
