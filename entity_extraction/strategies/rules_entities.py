"""EntityRuler regex/phrase entity layer (KG-AC-54, ADR-0012 §2 / ADR-0013). Deterministic,
confidence 1.0 (clarify 2026-08-06). Patterns are pack-authored (``regex_patterns``); a
checksum-declared pattern (LEI/ISIN) is written only when the match's format check digits are
valid — an invalid-checksum match is silently dropped, never written as a false-positive entity.
Opt-in via ``ExtractionConfig.rules_entities_enabled``.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List

from core import Candidate

from .base import Chunk, EntityStrategy, ExtractionConfig


def _letters_to_digits(s: str) -> str:
    return "".join(ch if ch.isdigit() else str(ord(ch.upper()) - ord("A") + 10) for ch in s)


def _lei_valid(candidate: str) -> bool:
    """ISO 17442 check digits (ISO 7064 MOD 97-10): the 20-char LEI's numeric expansion mod 97 == 1."""
    if len(candidate) != 20:
        return False
    remainder = 0
    for ch in _letters_to_digits(candidate):
        remainder = (remainder * 10 + int(ch)) % 97
    return remainder == 1


def _isin_valid(candidate: str) -> bool:
    """ISO 6166 check digit: Luhn over the 12-char ISIN's letters-expanded numeric form."""
    if len(candidate) != 12:
        return False
    digits = [int(d) for d in _letters_to_digits(candidate)]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:  # every second digit counting from the rightmost (the check digit itself)
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_CHECKSUM_VALIDATORS: Dict[str, Callable[[str], bool]] = {"lei": _lei_valid, "isin": _isin_valid}


class RulesEntitiesStrategy(EntityStrategy):
    layer = "regex"

    def extract(self, chunks: List[Chunk], config: ExtractionConfig, pack) -> List[Candidate]:
        out: List[Candidate] = []
        for rp in pack.regex_patterns:
            regex = re.compile(rp.pattern)
            validator = _CHECKSUM_VALIDATORS.get(rp.checksum) if rp.checksum else None
            for ch in chunks:
                for m in regex.finditer(ch.text):
                    surface = m.group(0)
                    if validator is not None and not validator(surface):
                        continue
                    out.append(Candidate(
                        surface_form=surface, entity_type=rp.entity_type, source_chunk_id=ch.chunk_id,
                        layer="regex", span_start=m.start(), span_end=m.end(), confidence=1.0,
                    ))
        return out
