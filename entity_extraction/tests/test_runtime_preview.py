"""P9 (spec v13, KG-AC-75): runtime preview diagnostic parity. The store-only runtime preview
(KG-AC-20) is the operator's only single-document debugging surface — before v13 it returned
relations as `{relation_type, src, dst}` only, hiding both the evidence and the reason anything
was discarded. It now ALSO carries each relation's `evidence_text`, the extracted `facts[]`,
per-item document/page provenance, and every KG-AC-74 drop counter (+ P7's
`guardrails_blocked_facts`) — the SAME `run_pipeline` engine production uses, never a divergent
code path. Store-only posture is unchanged: no run_state, no M1-M4, no kg rows (verified by the
existing store-only tests elsewhere; not re-proven here).

Uses the `investment_fibo` pack (real datatype_properties + relations) — NOT `fibo_core`/
`fibo_custom`, which are planned for removal (owner instruction 2026-08-11)."""
import pytest

from runtime import run_preview


class _FakeLlmClient:
    resolved_model = "fake-model"
    usage = []

    def complete_tool(self, **_kwargs):
        return {
            "entities": [
                {"type": "Agreement", "surface": "The Agreement"},
                {"type": "Investor", "surface": "Acme Investor"},
                {"type": "LegalEntity", "surface": "Acme LLC"},
            ],
            "relations": [
                {"type": "hasLegalEntity", "src_id": 1, "dst_id": 2,
                 "evidence": "Acme Investor is Acme LLC."},
            ],
            "facts": [
                {"subject_id": 0, "property": "agreementId", "value": "IMA-2025-018",
                 "evidence": "The Agreement IMA-2025-018 was signed."},
            ],
        }


_TEXT = "The Agreement IMA-2025-018 was signed. Acme Investor is Acme LLC."


def _preview(config_dict=None, llm_client=None):
    return run_preview(
        config_dict or {"engine": "llm", "ontology_pack": "investment_fibo"}, _TEXT,
        llm_client=llm_client or _FakeLlmClient(),
    )


@pytest.mark.ac("KG-AC-75")
def test_relations_carry_evidence_text():
    out = _preview()
    # v15: the preview now also carries DERIVED edges (this sample has an Agreement with an
    # agreementId plus parties, so a hub is minted and attached) -- assert on the MODEL-emitted
    # relation specifically rather than on a total count, which is what this AC is about.
    model_rel = next(r for r in out["relations"] if r["relation_type"] == "hasLegalEntity")
    assert model_rel["evidence_text"] == "Acme Investor is Acme LLC."


@pytest.mark.ac("KG-AC-91")
def test_preview_shows_derived_edges_end_to_end():
    # end-to-end proof that the derivation pass is wired into run_pipeline (not just unit-tested):
    # a derived hub reaches the preview with its definitional edges and no fabricated evidence.
    out = _preview()
    derived_edges = [r for r in out["relations"] if r["evidence_text"] is None]
    assert derived_edges, "derived edges must reach the preview"
    assert any(r["relation_type"] == "governedBy" for r in derived_edges)
    hub = next(e for e in out["entities"] if e["entity_type"] == "InvestmentRelationship")
    assert hub["surface_form"] == "IMA-2025-018"  # identity value, never a composed name


@pytest.mark.ac("KG-AC-75")
def test_relations_and_entities_carry_document_page_provenance():
    out = _preview()
    # a runtime preview has no real document -- both fields are present (never omitted), null
    # (KG-AC-73's own null-on-absence rule, since a synthetic "sample" chunk carries no metadata)
    assert "source_doc_id" in out["relations"][0] and "page" in out["relations"][0]
    assert "source_doc_id" in out["entities"][0] and "page" in out["entities"][0]
    assert out["entities"][0]["source_doc_id"] is None and out["entities"][0]["page"] is None


@pytest.mark.ac("KG-AC-75")
def test_extracted_facts_are_surfaced():
    out = _preview()
    assert len(out["facts"]) == 1
    fact = out["facts"][0]
    assert fact["property"] == "agreementId"
    assert fact["value"] == "IMA-2025-018"
    assert fact["normalized_value"] == "IMA-2025-018"
    assert fact["evidence_text"] == "The Agreement IMA-2025-018 was signed."
    assert fact["subject_type"] == "Agreement" and fact["subject_surface"] == "The Agreement"


@pytest.mark.ac("KG-AC-75")
def test_every_kg_ac_74_counter_plus_guardrails_blocked_facts_is_present():
    out = _preview()
    for field in ("unmapped_property_count", "unresolved_reference_count",
                 "unlocatable_entity_count", "ungrounded_fact_count", "guardrails_blocked_facts"):
        assert field in out, field
        assert out[field] == 0  # nothing was dropped in this scenario


@pytest.mark.ac("KG-AC-75")
def test_today_s_fields_are_unaffected():
    out = _preview()
    assert out["status"] == "success"
    assert out["entities"][0]["surface_form"] == "The Agreement"
    assert out["relations"][0]["relation_type"] == "hasLegalEntity"
    assert out["relations"][0]["src"] == "Acme Investor"
    assert out["relations"][0]["dst"] == "Acme LLC"
    assert out["ontology_pack"] == "investment_fibo"
    assert out["unmapped_type_count"] == 0
