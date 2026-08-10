"""KG-AC-54: the EntityRuler regex/phrase entity layer (ADR-0012 §2) — pack-authored
`regex_patterns` yield typed candidates at confidence 1.0; a checksum-declared pattern (LEI/ISIN)
is accepted only when the format's check digits are valid. The valid/invalid LEI and ISIN fixtures
below are derived by an INDEPENDENT reference implementation of the public ISO 17442 / ISO 6166
check-digit algorithms (not imported from the module under test), so the test proves correctness
against the spec, not just self-consistency with the implementation.
"""
import pytest

from core import Candidate
from ontologies import EntityType, Pack, RegexPattern
from strategies.base import Chunk, ExtractionConfig
from strategies.rules_entities import RulesEntitiesStrategy


# ---- independent reference implementation (test-side ground truth) --------
def _letters_to_digits(s: str) -> str:
    return "".join(ch if ch.isdigit() else str(ord(ch.upper()) - ord("A") + 10) for ch in s)


def _valid_lei(prefix18: str) -> str:
    """ISO 17442: check digits = 98 - (mod97 of the prefix+'00' numeric expansion)."""
    numeric = _letters_to_digits(prefix18 + "00")
    remainder = 0
    for ch in numeric:
        remainder = (remainder * 10 + int(ch)) % 97
    check = 98 - remainder
    return f"{prefix18}{check:02d}"


def _valid_isin(country_and_id9: str) -> str:
    """ISO 6166: Luhn check digit over the letters-expanded 11-char body."""
    numeric = _letters_to_digits(country_and_id9)
    digits = [int(d) for d in numeric]
    total = 0
    # Luhn over the body treating it as if the (not-yet-known) check digit were appended:
    # double every digit at an EVEN position counting from the right of the FULL 12-char string,
    # i.e. every digit at an ODD position of the 11-char body (0-indexed from the right).
    for i, d in enumerate(reversed(digits)):
        pos_from_right_in_full = i + 1  # the check digit itself occupies position 0
        if pos_from_right_in_full % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    check = (10 - (total % 10)) % 10
    return f"{country_and_id9}{check}"


_VALID_LEI = _valid_lei("529900HNOAA1KXQJUQ")  # 18-char prefix; _valid_lei appends 2 check digits
_INVALID_LEI = _VALID_LEI[:-1] + str((int(_VALID_LEI[-1]) + 1) % 10)  # corrupt the last check digit
_VALID_ISIN = _valid_isin("US037833100")
_INVALID_ISIN = _VALID_ISIN[:-1] + str((int(_VALID_ISIN[-1]) + 1) % 10)
assert len(_VALID_LEI) == 20 and len(_INVALID_LEI) == 20
assert len(_VALID_ISIN) == 12 and len(_INVALID_ISIN) == 12
assert _VALID_ISIN == "US0378331005"  # cross-checked against Apple Inc.'s real, published ISIN


def _lei_pack():
    return Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("Organization", None, [], "", None)], relations=[],
        regex_patterns=[RegexPattern(entity_type="Organization", pattern=r"\b[A-Z0-9]{18}[0-9]{2}\b", checksum="lei")],
    )


@pytest.mark.ac("KG-AC-54")
def test_regex_layer_yields_candidate_on_valid_checksum():
    strategy = RulesEntitiesStrategy()
    chunks = [Chunk("c1", f"The LEI is {_VALID_LEI} for this entity.")]
    out = strategy.extract(chunks, ExtractionConfig(), _lei_pack())
    assert len(out) == 1
    c = out[0]
    assert isinstance(c, Candidate)
    assert c.surface_form == _VALID_LEI
    assert c.entity_type == "Organization"
    assert c.layer == "regex"
    assert c.confidence == 1.0
    assert c.source_chunk_id == "c1"


@pytest.mark.ac("KG-AC-54")
def test_regex_layer_drops_match_on_invalid_checksum():
    strategy = RulesEntitiesStrategy()
    chunks = [Chunk("c1", f"The LEI is {_INVALID_LEI} for this entity.")]
    out = strategy.extract(chunks, ExtractionConfig(), _lei_pack())
    assert out == []


@pytest.mark.ac("KG-AC-54")
def test_isin_checksum_kind_also_supported():
    pack = Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("FinancialInstrument", None, [], "", None)], relations=[],
        regex_patterns=[RegexPattern(entity_type="FinancialInstrument", pattern=r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b", checksum="isin")],
    )
    strategy = RulesEntitiesStrategy()
    good = strategy.extract([Chunk("c1", f"ISIN {_VALID_ISIN} traded today.")], ExtractionConfig(), pack)
    bad = strategy.extract([Chunk("c1", f"ISIN {_INVALID_ISIN} traded today.")], ExtractionConfig(), pack)
    assert len(good) == 1 and good[0].surface_form == _VALID_ISIN
    assert bad == []


@pytest.mark.ac("KG-AC-54")
def test_no_checksum_accepts_any_regex_match():
    pack = Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("Date", None, [], "", None)], relations=[],
        regex_patterns=[RegexPattern(entity_type="Date", pattern=r"\b\d{4}-\d{2}-\d{2}\b")],
    )
    strategy = RulesEntitiesStrategy()
    out = strategy.extract([Chunk("c1", "Filed on 2026-08-06 in the report.")], ExtractionConfig(), pack)
    assert len(out) == 1
    assert out[0].surface_form == "2026-08-06" and out[0].entity_type == "Date"


@pytest.mark.ac("KG-AC-54")
def test_multiple_matches_across_chunks():
    pack = Pack(
        name="x", version="1", description="",
        entity_types=[EntityType("Date", None, [], "", None)], relations=[],
        regex_patterns=[RegexPattern(entity_type="Date", pattern=r"\b\d{4}-\d{2}-\d{2}\b")],
    )
    strategy = RulesEntitiesStrategy()
    chunks = [Chunk("c1", "Start 2026-01-01."), Chunk("c2", "End 2026-12-31.")]
    out = strategy.extract(chunks, ExtractionConfig(), pack)
    assert {(c.source_chunk_id, c.surface_form) for c in out} == {("c1", "2026-01-01"), ("c2", "2026-12-31")}


@pytest.mark.ac("KG-AC-54")
def test_no_patterns_yields_no_candidates():
    pack = Pack(name="x", version="1", description="",
                entity_types=[EntityType("Organization", None, [], "", None)], relations=[])
    out = RulesEntitiesStrategy().extract([Chunk("c1", "Acme Corp filed a report.")], ExtractionConfig(), pack)
    assert out == []
