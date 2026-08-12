"""P11 (spec v13, KG-AC-78): attribute merge across a canonicalised cluster's mentions, with
CONFLICT RETENTION — never silent last-write-wins. Two mentions asserting the SAME normalized
value for a property collapse to one fact whose provenance list carries both sources. Two mentions
asserting DIFFERENT values are BOTH retained, each with its own provenance, flagged `conflicting`
(a genuine cross-document disagreement is a finding, not noise to resolve away).

**Amended by P13 (KG-AC-80):** the boolean `conflicting` field is replaced by the full `status`
string (`single_source`/`consistent`/`conflicting`) — the exact, pre-announced extension P11's own
Done writeup scoped: "`conflicting: bool` is P13-compatible by construction: `True` maps directly
to `status: 'conflicting'`; `False` is exactly the set P13 refines." Not a surprise, the planned
continuation."""
import pytest

from core import merge_attributes


def _mention_attrs(*facts):
    """facts: list of dicts with property/value/normalized_value/evidence/source_doc_id/page —
    the SAME shape entity_extraction's attach_facts_to_entity_records writes onto kg_entities.attributes."""
    return list(facts)


def _f(prop, value, normalized_value=None, doc="d1", page=1, evidence="e"):
    return {"property": prop, "value": value, "normalized_value": normalized_value or value,
            "evidence": evidence, "source_doc_id": doc, "page": page}


# ---- same value collapses, provenance carries both sources --------------------------------------
@pytest.mark.ac("KG-AC-78")
def test_same_normalized_value_collapses_to_one_fact_with_both_provenance():
    mentions_attrs = [
        _mention_attrs(_f("governingLaw", "England and Wales", doc="docA")),
        _mention_attrs(_f("governingLaw", "England and Wales", doc="docB")),
    ]
    merged = merge_attributes(mentions_attrs)
    assert list(merged.keys()) == ["governingLaw"]
    entries = merged["governingLaw"]
    assert len(entries) == 1  # collapsed, not duplicated
    assert entries[0]["status"] == "consistent"  # 2 distinct docs, agreeing
    docs = {p["source_doc_id"] for p in entries[0]["provenance"]}
    assert docs == {"docA", "docB"}  # BOTH sources present


@pytest.mark.ac("KG-AC-78")
def test_normalized_value_is_the_equality_key_not_raw_value():
    # "15 March 2025" and "2025-03-15" are the SAME date once normalized -- must collapse, even
    # though the raw strings differ (mirrors KG-AC-70's own normalize_fact_value distinction).
    mentions_attrs = [
        _mention_attrs(_f("effectiveDate", "15 March 2025", normalized_value="2025-03-15", doc="docA")),
        _mention_attrs(_f("effectiveDate", "2025-03-15", normalized_value="2025-03-15", doc="docB")),
    ]
    merged = merge_attributes(mentions_attrs)
    assert len(merged["effectiveDate"]) == 1
    assert merged["effectiveDate"][0]["status"] == "consistent"  # 2 distinct docs, agreeing


# ---- different values: BOTH retained, flagged conflicting, never last-write-wins -----------------
@pytest.mark.ac("KG-AC-78")
def test_different_values_both_retained_and_flagged_conflicting():
    mentions_attrs = [
        _mention_attrs(_f("effectiveDate", "15 March 2025", normalized_value="2025-03-15", doc="docA")),
        _mention_attrs(_f("effectiveDate", "20 March 2025", normalized_value="2025-03-20", doc="docB")),
    ]
    merged = merge_attributes(mentions_attrs)
    entries = merged["effectiveDate"]
    assert len(entries) == 2  # NEITHER value discarded
    assert all(e["status"] == "conflicting" for e in entries)
    values = {e["normalized_value"] for e in entries}
    assert values == {"2025-03-15", "2025-03-20"}


@pytest.mark.ac("KG-AC-78")
def test_no_last_write_wins_regardless_of_input_order():
    # the defect this AC exists to prevent: reversing the input must NOT change which value "wins"
    # -- because nothing should win. Both orderings retain both values.
    a = [_mention_attrs(_f("effectiveDate", "15 March 2025", normalized_value="2025-03-15")),
         _mention_attrs(_f("effectiveDate", "20 March 2025", normalized_value="2025-03-20"))]
    b = list(reversed(a))
    merged_a = {e["normalized_value"] for e in merge_attributes(a)["effectiveDate"]}
    merged_b = {e["normalized_value"] for e in merge_attributes(b)["effectiveDate"]}
    assert merged_a == merged_b == {"2025-03-15", "2025-03-20"}


@pytest.mark.ac("KG-AC-78")
def test_conflicting_value_provenance_stays_scoped_to_its_own_value():
    # docA's evidence must NOT leak onto the docB-asserted value, and vice versa.
    mentions_attrs = [
        _mention_attrs(_f("effectiveDate", "15 March 2025", normalized_value="2025-03-15", doc="docA")),
        _mention_attrs(_f("effectiveDate", "20 March 2025", normalized_value="2025-03-20", doc="docB")),
    ]
    merged = merge_attributes(mentions_attrs)
    by_value = {e["normalized_value"]: e for e in merged["effectiveDate"]}
    assert [p["source_doc_id"] for p in by_value["2025-03-15"]["provenance"]] == ["docA"]
    assert [p["source_doc_id"] for p in by_value["2025-03-20"]["provenance"]] == ["docB"]


# ---- multiple properties, multiple mentions per property -----------------------------------------
@pytest.mark.ac("KG-AC-78")
def test_multiple_properties_merge_independently():
    mentions_attrs = [
        _mention_attrs(
            _f("governingLaw", "England and Wales", doc="docA"),
            _f("agreementId", "IMA-2025-018", doc="docA"),
        ),
        _mention_attrs(_f("governingLaw", "England and Wales", doc="docB")),
    ]
    merged = merge_attributes(mentions_attrs)
    assert set(merged.keys()) == {"governingLaw", "agreementId"}
    assert len(merged["governingLaw"]) == 1
    assert len(merged["agreementId"]) == 1
    assert merged["agreementId"][0]["status"] == "single_source"  # only docA ever asserted it


@pytest.mark.ac("KG-AC-78")
def test_three_way_split_all_distinct_values_retained():
    mentions_attrs = [
        _mention_attrs(_f("status", "Active", doc="docA")),
        _mention_attrs(_f("status", "Terminated", doc="docB")),
        _mention_attrs(_f("status", "Suspended", doc="docC")),
    ]
    merged = merge_attributes(mentions_attrs)
    assert len(merged["status"]) == 3
    assert all(e["status"] == "conflicting" for e in merged["status"])


# ---- empty / no-facts cases ------------------------------------------------------------------------
@pytest.mark.ac("KG-AC-78")
def test_no_facts_anywhere_yields_empty_merge():
    assert merge_attributes([[], []]) == {}


@pytest.mark.ac("KG-AC-78")
def test_empty_mention_list_yields_empty_merge():
    assert merge_attributes([]) == {}


@pytest.mark.ac("KG-AC-78")
def test_repeated_mention_of_same_value_within_one_document_stays_a_single_provenance_entry_per_fact():
    # two facts from the SAME document/page/evidence (e.g. re-extracted at two occurrence indices)
    # must not silently duplicate provenance entries -- each source fact keeps its own record, but
    # exact duplicates (identical doc+page+evidence) collapse rather than padding the list.
    mentions_attrs = [
        _mention_attrs(_f("governingLaw", "England and Wales", doc="docA", page=1, evidence="e1")),
        _mention_attrs(_f("governingLaw", "England and Wales", doc="docA", page=1, evidence="e1")),
    ]
    merged = merge_attributes(mentions_attrs)
    assert len(merged["governingLaw"][0]["provenance"]) == 1
    # ONE distinct source document (docA) despite two identical facts -> single_source, not consistent
    assert merged["governingLaw"][0]["status"] == "single_source"


# ================================================================================================
# P13 (spec v13, KG-AC-80): the full status vocabulary — single_source / consistent / conflicting.
# ================================================================================================
@pytest.mark.ac("KG-AC-80")
def test_one_document_is_single_source_never_consistent():
    # KG-AC-80's own emphasis: single-source must NOT be reported as "consistent", which would
    # overstate the evidence of one unchallenged reading.
    mentions_attrs = [_mention_attrs(_f("governingLaw", "England and Wales", doc="docA"))]
    merged = merge_attributes(mentions_attrs)
    assert merged["governingLaw"][0]["status"] == "single_source"


@pytest.mark.ac("KG-AC-80")
def test_two_distinct_documents_agreeing_is_consistent():
    mentions_attrs = [
        _mention_attrs(_f("governingLaw", "England and Wales", doc="docA")),
        _mention_attrs(_f("governingLaw", "England and Wales", doc="docB")),
    ]
    merged = merge_attributes(mentions_attrs)
    assert merged["governingLaw"][0]["status"] == "consistent"


@pytest.mark.ac("KG-AC-80")
def test_three_distinct_documents_agreeing_is_still_consistent_not_a_fourth_status():
    mentions_attrs = [
        _mention_attrs(_f("governingLaw", "England and Wales", doc="docA")),
        _mention_attrs(_f("governingLaw", "England and Wales", doc="docB")),
        _mention_attrs(_f("governingLaw", "England and Wales", doc="docC")),
    ]
    merged = merge_attributes(mentions_attrs)
    assert merged["governingLaw"][0]["status"] == "consistent"


@pytest.mark.ac("KG-AC-80")
def test_lopsided_split_is_conflicting_not_a_fourth_status():
    # clarify 2026-08-11 (baked into KG-AC-80's own text): conflicting whenever >=2 distinct
    # normalized values exist, HOWEVER LOPSIDED -- 3-against-1 is still conflicting, not a
    # majority-wins status, and NEITHER side's document count changes that classification.
    mentions_attrs = [
        _mention_attrs(_f("effectiveDate", "15 March 2025", normalized_value="2025-03-15", doc="docA")),
        _mention_attrs(_f("effectiveDate", "15 March 2025", normalized_value="2025-03-15", doc="docB")),
        _mention_attrs(_f("effectiveDate", "15 March 2025", normalized_value="2025-03-15", doc="docC")),
        _mention_attrs(_f("effectiveDate", "20 March 2025", normalized_value="2025-03-20", doc="docD")),
    ]
    merged = merge_attributes(mentions_attrs)
    entries = merged["effectiveDate"]
    assert len(entries) == 2
    assert all(e["status"] == "conflicting" for e in entries)
    # each distinct value STILL carries its own supporting document set -- the split is visible,
    # not collapsed into a single number
    by_value = {e["normalized_value"]: e for e in entries}
    assert {p["source_doc_id"] for p in by_value["2025-03-15"]["provenance"]} == {"docA", "docB", "docC"}
    assert {p["source_doc_id"] for p in by_value["2025-03-20"]["provenance"]} == {"docD"}


@pytest.mark.ac("KG-AC-80")
def test_no_known_source_doc_id_defaults_to_single_source_not_a_fourth_status():
    # defensive: a fact with no source_doc_id at all (e.g. a degenerate synthetic call) must not
    # crash the distinct-count logic or invent a status outside the closed vocabulary.
    mentions_attrs = [_mention_attrs(_f("governingLaw", "England and Wales", doc=None))]
    merged = merge_attributes(mentions_attrs)
    assert merged["governingLaw"][0]["status"] == "single_source"
