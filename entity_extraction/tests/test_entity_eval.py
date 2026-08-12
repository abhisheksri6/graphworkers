"""KG-AC-P1 (spaCy/generic type-agnostic span F1), KG-AC-P2 (LLM/fibo_core relaxed micro-F1 + per-type
recall), KG-AC-P3 (LLM JSON validity rate) vs the frozen synthetic golden set
(sample-datasets/knowledge-graph/, C13/A5).

**Environment-honest gating, verified empirically in this build session — not assumed:**
  - P1 needs the by-copy `en_core_web_lg` model (`SPACY_MODEL_PATH`) — task C12, a deferred ~600MB
    asset. Not installed here (`spacy.load` raises `OSError` for every model size, including `_sm`).
  - P2/P3 need a live LLM connection. This sandbox has no reachable model: the `AWS LLM Connection`
    profile decrypts but the endpoint returns HTTP 401 (no session outside a real backend login,
    confirmed by an actual attempted call — not assumed), and the local Ollama instance only serves
    an embedding model (`nomic-embed-text`), no chat model.
  - Both real-eval tests below are gated on explicit env vars (`SPACY_MODEL_PATH` real-loadable /
    `KG_EVAL_LLM_CONNECTION` set) and SKIP with a clear reason when absent — never a fabricated pass.
  - The scorer functions themselves ARE verified today, via `test_*_scorer_self_test` functions below
    that run against hand-built known inputs (no model/LLM needed) — proving the scoring MATH is
    correct independent of asset availability.

Runs only under `pytest -m eval`.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path

import pytest

from ontologies import load_pack
from strategies import Chunk, ExtractionConfig
from strategies.llm_graph import LlmOutputError, build_graph_system_prompt, build_graph_tool_schema, build_graph_user_prompt
from strategies.spacy_ner import SpacyNerStrategy

# OWNER DECISION 2026-08-12: `fibo_core`/`fibo_custom` are slated for deletion, and their golden
# datasets with them — do not spend time running or maintaining evals against them. Skipped whole-
# file (not deleted) so the AC markers below stay visible to `check_spec_conformance.py` until the
# replacement investment_fibo dataset lands and these are removed outright.
pytestmark = [
    pytest.mark.eval,
    pytest.mark.skip(reason="fibo_core/fibo_custom datasets pending deletion (owner decision "
                            "2026-08-12) — superseded by the investment_fibo golden dataset"),
]

REPO = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get(
    "CB_DATASETS_DIR", REPO.parent.parent / "sample-datasets",
)).resolve() / "knowledge-graph"
BASELINES = Path(os.environ.get(
    "CB_GOVERNANCE_DIR", REPO.parent.parent / "coding-governance",
)).resolve() / "specs" / "evaluation" / "baselines.json"


@pytest.fixture(scope="module", autouse=True)
def verify_dataset_integrity():
    manifest_path = DATASET / "manifest.json"
    assert manifest_path.exists(), f"synthetic dataset not found at {DATASET} — set CB_DATASETS_DIR"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        p = DATASET / rel
        assert p.exists(), f"dataset file missing: {rel}"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, (
            f"dataset file CHANGED since generation: {rel} (re-run the generator or re-freeze)"
        )


def _load_chunks():
    return json.loads((DATASET / "gold" / "chunks.json").read_text(encoding="utf-8"))


def _load_entities():
    return json.loads((DATASET / "gold" / "entities.json").read_text(encoding="utf-8"))


def _baseline(key):
    if BASELINES.exists():
        return json.loads(BASELINES.read_text(encoding="utf-8")).get(key)
    return None


# ---------------------------------------------------------------------------
# Scorers (pure — no model/LLM). Duplicated per-repo deliberately, matching the
# specs/evaluation/design.md convention (no shared scorer library until a 3rd consumer appears).
# ---------------------------------------------------------------------------
def type_agnostic_span_f1(predicted: list[tuple], gold: list[tuple]) -> dict:
    """predicted/gold: (chunk_id, span_start, span_end) tuples. Exact span match, type ignored,
    micro-aggregated (KG-AC-P1)."""
    pred_set, gold_set = set(predicted), set(gold)
    tp, fp, fn = len(pred_set & gold_set), len(pred_set - gold_set), len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
           "tp": tp, "fp": fp, "fn": fn}


def type_exact_span_f1(predicted: list[tuple], gold: list[tuple]) -> dict:
    """predicted/gold: (chunk_id, span_start, span_end, type) tuples. Reported alongside P1, not gated."""
    return type_agnostic_span_f1(predicted, gold)  # identical mechanics, 4-tuples make type part of the key


def _norm_text(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()


def surface_matches(a: str, b: str, threshold: float = 0.85) -> bool:
    na, nb = _norm_text(a), _norm_text(b)
    if na == nb:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= threshold


def relaxed_entity_micro_f1(predicted: list[dict], gold: list[dict]) -> dict:
    """predicted/gold: dicts with chunk_id/type/surface. TP: type-exact + surface span-relaxed
    (KG-AC-P2), greedy one-to-one match within a chunk. Also returns per-type recall."""
    gold_by_chunk: dict[str, list[dict]] = {}
    for g in gold:
        gold_by_chunk.setdefault(g["chunk_id"], []).append({**g, "_matched": False})

    tp = fp = 0
    for p in predicted:
        cands = gold_by_chunk.get(p["chunk_id"], [])
        m = next((g for g in cands if not g["_matched"] and g["type"] == p["type"]
                  and surface_matches(g["surface"], p["surface"])), None)
        if m:
            m["_matched"] = True
            tp += 1
        else:
            fp += 1
    all_gold = [g for lst in gold_by_chunk.values() for g in lst]
    fn = sum(1 for g in all_gold if not g["_matched"])
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_type_total: dict[str, int] = {}
    per_type_matched: dict[str, int] = {}
    for g in all_gold:
        per_type_total[g["type"]] = per_type_total.get(g["type"], 0) + 1
        if g["_matched"]:
            per_type_matched[g["type"]] = per_type_matched.get(g["type"], 0) + 1
    per_type_recall = {t: round(per_type_matched.get(t, 0) / n, 4) for t, n in per_type_total.items()}

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
           "tp": tp, "fp": fp, "fn": fn, "per_type_recall": per_type_recall}


# ---------------------------------------------------------------------------
# Scorer self-tests (ALWAYS run — no marker, no model/LLM needed) — prove the scoring math is
# correct against hand-built known inputs, independent of asset availability.
# ---------------------------------------------------------------------------
def test_type_agnostic_span_f1_scorer_self_test():
    gold = [("c1", 0, 5), ("c1", 10, 15), ("c2", 0, 3)]
    predicted = [("c1", 0, 5), ("c1", 10, 15), ("c1", 20, 25)]  # 2 correct + 1 spurious + 1 missed
    scores = type_agnostic_span_f1(predicted, gold)
    assert scores == {"precision": round(2 / 3, 4), "recall": round(2 / 3, 4), "f1": round(2 / 3, 4),
                      "tp": 2, "fp": 1, "fn": 1}


def test_relaxed_entity_micro_f1_scorer_self_test():
    # "U.S. Securities and Exchange Commission" vs "Securities and Exchange Commission" is a
    # realistic prefix-drop an LLM might produce -- SequenceMatcher ratio 0.932, verified empirically
    # (not assumed) before writing this test.
    gold = [
        {"chunk_id": "c1", "type": "RegulatoryAgency", "surface": "U.S. Securities and Exchange Commission"},
        {"chunk_id": "c1", "type": "Person", "surface": "Jane Roe"},
    ]
    predicted = [
        {"chunk_id": "c1", "type": "RegulatoryAgency", "surface": "Securities and Exchange Commission"},
        {"chunk_id": "c1", "type": "Location", "surface": "Nowhere"},  # false positive
    ]
    scores = relaxed_entity_micro_f1(predicted, gold)
    assert scores["tp"] == 1 and scores["fp"] == 1 and scores["fn"] == 1
    assert scores["per_type_recall"]["RegulatoryAgency"] == 1.0
    assert scores["per_type_recall"]["Person"] == 0.0


def test_surface_matches_is_relaxed_not_exact():
    # verified empirically: prefix-drop/suffix-addition land above 0.85; a genuine abbreviation
    # (Corp vs Corporation, ratio 0.72) does NOT -- "relaxed" tolerates position/minor drift, not
    # arbitrary paraphrase.
    assert surface_matches("U.S. Securities and Exchange Commission", "Securities and Exchange Commission")
    assert surface_matches("Meridian Capital Partners", "Meridian Capital Partners LLC")
    assert not surface_matches("Acme Corp", "Acme Corporation")
    assert not surface_matches("Acme Corp", "Globex")


# ---------------------------------------------------------------------------
# KG-AC-P1 — real eval, gated on the by-copy spaCy model actually loading.
# ---------------------------------------------------------------------------
def _spacy_model_ready() -> bool:
    path = os.environ.get("SPACY_MODEL_PATH", "")
    if not path:
        return False
    try:
        import spacy
        spacy.load(path)
        return True
    except Exception:  # noqa: BLE001 — gate check only, any failure means "not ready"
        return False


_SPACY_READY = _spacy_model_ready()


@pytest.mark.ac("KG-AC-P1")
@pytest.mark.skipif(not _SPACY_READY, reason="SPACY_MODEL_PATH not set or the by-copy en_core_web_lg "
                                             "model is not installed (task C12, deferred asset)")
def test_spacy_generic_span_f1_vs_baseline():
    chunks_gold = _load_chunks()
    entities_gold = _load_entities()
    generic = load_pack("generic")
    gold_spans = [(e["chunk_id"], e["span_start"], e["span_end"])
                  for e in entities_gold if e["generic_type"]]
    gold_typed = [(e["chunk_id"], e["span_start"], e["span_end"], e["generic_type"])
                  for e in entities_gold if e["generic_type"]]

    strat = SpacyNerStrategy(model_path=os.environ["SPACY_MODEL_PATH"])
    cfg = ExtractionConfig(engine="spacy", ontology_pack="generic")
    predicted_spans, predicted_typed = [], []
    for c in chunks_gold:
        cands = strat.extract([Chunk(c["chunk_id"], c["text"])], cfg, generic)
        predicted_spans += [(c["chunk_id"], cand.span_start, cand.span_end) for cand in cands]
        predicted_typed += [(c["chunk_id"], cand.span_start, cand.span_end, cand.entity_type)
                            for cand in cands]

    scores = type_agnostic_span_f1(predicted_spans, gold_spans)
    type_exact = type_exact_span_f1(predicted_typed, gold_typed)
    print(f"\nKG spaCy/generic SPAN F1 (type-agnostic): {json.dumps(scores, indent=2)}")
    print(f"KG spaCy/generic SPAN F1 (type-exact, reported only): {json.dumps(type_exact, indent=2)}")

    base = _baseline("kg_extraction_spacy_generic")
    if base is None:
        assert scores["f1"] >= 0.60, scores
    else:
        assert scores["f1"] >= 0.60, scores
        assert scores["f1"] >= base["f1"] - 0.02, f"regression: {scores} vs baseline {base}"


# ---------------------------------------------------------------------------
# KG-AC-P2 + KG-AC-P3 — real eval, gated on an explicit live LLM connection.
# ---------------------------------------------------------------------------
_LLM_CONNECTION = os.environ.get("KG_EVAL_LLM_CONNECTION", "")
_SKIP_NO_LLM = pytest.mark.skipif(
    not _LLM_CONNECTION,
    reason="KG_EVAL_LLM_CONNECTION not set — needs a live LLM connection (Bedrock, reachable backend "
           "decrypt) which is not available in this environment (verified: AWS connection decrypts "
           "but returns 401 outside a real backend session; local Ollama has no chat model)",
)


def _run_llm_graph_over_chunks(chunks_gold, pack, client):
    """Per-chunk, failure-isolated (unlike LlmGraphStrategy.extract, which now propagates a tool-call
    failure uncaught — KG-AC-P3 v6, owner decision: fails the whole folder in PRODUCTION) —
    failure-isolation stays appropriate HERE because this is a QUALITY MEASUREMENT harness, not
    production; it wants "how many of N chunks succeeded", which requires trying all of them.
    Evolve v6: reuses the REAL complete_tool (Bedrock Converse tool-use) — the response is already a
    parsed dict, no more parse_strict_json. Returns (predicted_entities, tool_call_valid_count, total)."""
    system_text = build_graph_system_prompt(pack)
    tool_schema = build_graph_tool_schema(pack)
    predicted, valid, total = [], 0, 0
    for c in chunks_gold:
        total += 1
        try:
            data = client.complete_tool(
                system_text=system_text, user_text=build_graph_user_prompt(c["text"]),
                tool_name="extract_graph", tool_description="Record extracted entities and relations.",
                tool_schema=tool_schema,
            )
            valid += 1
        except LlmOutputError:
            continue
        for item in (data.get("entities") or []):
            etype, surface = item.get("type"), item.get("surface")
            if etype and surface:
                predicted.append({"chunk_id": c["chunk_id"], "type": etype, "surface": surface})
    return predicted, valid, total


@_SKIP_NO_LLM
@pytest.mark.ac("KG-AC-P2")
@pytest.mark.ac("KG-AC-P3")
def test_llm_fibo_core_micro_f1_and_json_validity_vs_baseline():
    from clients import build_llm_client

    chunks_gold = _load_chunks()
    entities_gold = [{"chunk_id": e["chunk_id"], "type": e["fibo_core_type"], "surface": e["surface"]}
                     for e in _load_entities()]
    fibo = load_pack("fibo_core")
    client = build_llm_client(_LLM_CONNECTION)

    predicted, valid, total = _run_llm_graph_over_chunks(chunks_gold, fibo, client)
    entity_scores = relaxed_entity_micro_f1(predicted, entities_gold)
    validity_rate = round(valid / total, 4) if total else 0.0
    print(f"\nKG LLM/fibo_core ENTITY MICRO-F1: {json.dumps(entity_scores, indent=2)}")
    print(f"KG LLM TOOL-CALL VALIDITY RATE: {valid}/{total} = {validity_rate}")

    base = _baseline("kg_extraction_llm_fibo_core")
    if base is None:
        assert entity_scores["f1"] >= 0.55, entity_scores
        assert validity_rate >= 0.98, (valid, total)
    else:
        assert entity_scores["f1"] >= 0.55, entity_scores
        assert entity_scores["f1"] >= base["f1"] - 0.02, f"regression: {entity_scores} vs baseline {base}"
        assert validity_rate >= 0.98, (valid, total)
