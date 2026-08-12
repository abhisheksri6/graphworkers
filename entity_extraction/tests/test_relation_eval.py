"""KG-AC-P7 (relation quality — H6) vs the frozen synthetic golden set
(sample-datasets/knowledge-graph/, A5): relation micro-F1, TP = relation_type exact + BOTH endpoints
match under KG-AC-P2's rule (type-exact + surface span-relaxed), measured at extraction output
(pre-canonicalization). Promotes the withdrawn report-only KG-AC-P4.

**Same environment-honest gating as test_entity_eval.py's P2/P3** (duplicated here deliberately —
specs/evaluation/design.md's stated convention: no shared scorer library until a 3rd consumer
appears): this sandbox has no reachable LLM connection (verified empirically, not assumed — see
that file's module docstring for the exact evidence). The real eval SKIPS with a clear reason; the
scorer itself is proven via a self-test using known inputs, no LLM needed.

Runs only under `pytest -m eval`.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path

import pytest

from core import assign_occurrence_indices, merge_candidates
from ontologies import load_pack
from strategies import Chunk, ExtractionConfig, filter_closed_vocab, run_graph_extraction
from strategies.llm_classify import LlmClassifyStrategy
from strategies.llm_entity_scoped import LlmEntityScopedStrategy
from strategies.llm_graph import LlmOutputError, build_graph_system_prompt, build_graph_tool_schema, build_graph_user_prompt
from strategies.spacy_ner import load_spacy_model
from candidate_pairs import enumerate_candidate_pairs

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


def _load_relations():
    return json.loads((DATASET / "gold" / "relations.json").read_text(encoding="utf-8"))


def _baseline():
    if BASELINES.exists():
        return json.loads(BASELINES.read_text(encoding="utf-8")).get("kg_relation_llm_fibo_core")
    return None


# ---------------------------------------------------------------------------
# Scorer (pure — no LLM). Duplicated from test_entity_eval.py's surface_matches deliberately
# (specs/evaluation/design.md convention: no shared scorer library until a 3rd consumer appears).
# ---------------------------------------------------------------------------
def _norm_text(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()


def surface_matches(a: str, b: str, threshold: float = 0.85) -> bool:
    na, nb = _norm_text(a), _norm_text(b)
    if na == nb:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= threshold


def relation_micro_f1(predicted: list[dict], gold: list[dict]) -> dict:
    """predicted/gold: dicts with chunk_id, relation_type, src_type, src_surface, dst_type,
    dst_surface. TP: relation_type exact AND both endpoints match under KG-AC-P2's rule (type-exact
    + surface span-relaxed) (KG-AC-P7). Greedy one-to-one within a chunk. Also returns per-relation-
    type precision/recall."""
    gold_by_chunk: dict[str, list[dict]] = {}
    for g in gold:
        gold_by_chunk.setdefault(g["chunk_id"], []).append({**g, "_matched": False})

    tp = fp = 0
    per_type_tp: dict[str, int] = {}
    per_type_fp: dict[str, int] = {}
    for p in predicted:
        cands = gold_by_chunk.get(p["chunk_id"], [])
        m = next((g for g in cands if not g["_matched"]
                  and g["relation_type"] == p["relation_type"]
                  and g["src_type"] == p["src_type"] and surface_matches(g["src_surface"], p["src_surface"])
                  and g["dst_type"] == p["dst_type"] and surface_matches(g["dst_surface"], p["dst_surface"])),
                 None)
        if m:
            m["_matched"] = True
            tp += 1
            per_type_tp[p["relation_type"]] = per_type_tp.get(p["relation_type"], 0) + 1
        else:
            fp += 1
            per_type_fp[p["relation_type"]] = per_type_fp.get(p["relation_type"], 0) + 1

    all_gold = [g for lst in gold_by_chunk.values() for g in lst]
    fn = sum(1 for g in all_gold if not g["_matched"])
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_type_gold_total: dict[str, int] = {}
    per_type_gold_matched: dict[str, int] = {}
    for g in all_gold:
        per_type_gold_total[g["relation_type"]] = per_type_gold_total.get(g["relation_type"], 0) + 1
        if g["_matched"]:
            per_type_gold_matched[g["relation_type"]] = per_type_gold_matched.get(g["relation_type"], 0) + 1
    per_type_recall = {t: round(per_type_gold_matched.get(t, 0) / n, 4)
                       for t, n in per_type_gold_total.items()}
    per_type_precision = {
        t: round(per_type_tp.get(t, 0) / (per_type_tp.get(t, 0) + per_type_fp.get(t, 0)), 4)
        for t in set(per_type_tp) | set(per_type_fp)
    }

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
           "tp": tp, "fp": fp, "fn": fn,
           "per_relation_type_recall": per_type_recall, "per_relation_type_precision": per_type_precision}


def relation_edge_key(r: dict) -> tuple:
    """Canonical, hashable identity for a relation edge. Set-membership needs an exact key, unlike
    the fuzzy `surface_matches` used for TP scoring above — whitespace/case-normalized only."""
    return (r["chunk_id"], r["relation_type"], r["src_type"], _norm_text(r["src_surface"]),
            r["dst_type"], _norm_text(r["dst_surface"]))


def relation_edge_set_variance(runs: list[list[dict]]) -> dict:
    """Run-to-run stability of a relation extractor over the SAME input (KG-AC-P8's "variance
    reported" — report-only, no threshold set yet). `runs`: one predicted-edge list per repeated
    run over the same chunks. Pairwise Jaccard similarity of each run's canonical edge-key set,
    averaged across all pairs; `instability = 1 - mean_jaccard` (0 = identical every run, 1 =
    disjoint every run)."""
    if len(runs) < 2:
        raise ValueError("variance needs >=2 runs to compare")
    edge_sets = [{relation_edge_key(r) for r in run} for run in runs]
    counts = [len(s) for s in edge_sets]
    pair_jaccards = []
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            a, b = edge_sets[i], edge_sets[j]
            union = a | b
            pair_jaccards.append(1.0 if not union else len(a & b) / len(union))
    mean_jaccard = round(sum(pair_jaccards) / len(pair_jaccards), 4)
    count_mean = round(sum(counts) / len(counts), 4)
    count_variance = round(sum((c - count_mean) ** 2 for c in counts) / len(counts), 4)
    return {"num_runs": len(runs), "per_run_edge_counts": counts, "count_mean": count_mean,
           "count_variance": count_variance, "mean_pairwise_jaccard": mean_jaccard,
           "instability": round(1.0 - mean_jaccard, 4)}


# ---------------------------------------------------------------------------
# Scorer self-test (ALWAYS runs — no marker, no LLM needed).
# ---------------------------------------------------------------------------
def test_relation_edge_set_variance_scorer_self_test():
    run_a = [{"chunk_id": "c1", "relation_type": "issues", "src_type": "Organization",
              "src_surface": "Acme Corp", "dst_type": "Bond", "dst_surface": "Acme 5% 2030 Notes"}]
    run_b = list(run_a)  # identical run -> jaccard 1.0 against run_a
    run_c = [{"chunk_id": "c1", "relation_type": "employs", "src_type": "Organization",
              "src_surface": "Acme Corp", "dst_type": "Person", "dst_surface": "Jane Roe"}]  # disjoint
    scores = relation_edge_set_variance([run_a, run_b, run_c])
    assert scores["num_runs"] == 3
    assert scores["per_run_edge_counts"] == [1, 1, 1]
    # pairs: (a,b)=1.0, (a,c)=0.0, (b,c)=0.0 -> mean = 1/3
    assert scores["mean_pairwise_jaccard"] == round(1 / 3, 4)
    assert scores["instability"] == round(1 - 1 / 3, 4)
    assert scores["count_mean"] == 1.0 and scores["count_variance"] == 0.0


def test_relation_micro_f1_scorer_self_test():
    gold = [
        {"chunk_id": "c1", "relation_type": "issues",
         "src_type": "Organization", "src_surface": "Acme Corp",
         "dst_type": "Bond", "dst_surface": "Acme 5% 2030 Notes"},
        {"chunk_id": "c1", "relation_type": "employs",
         "src_type": "Organization", "src_surface": "Acme Corp",
         "dst_type": "Person", "dst_surface": "Jane Roe"},
    ]
    predicted = [
        # correct 'issues' relation, exact match
        {"chunk_id": "c1", "relation_type": "issues",
         "src_type": "Organization", "src_surface": "Acme Corp",
         "dst_type": "Bond", "dst_surface": "Acme 5% 2030 Notes"},
        # wrong relation_type -> false positive; the real 'employs' relation is missed (false negative)
        {"chunk_id": "c1", "relation_type": "manages",
         "src_type": "Organization", "src_surface": "Acme Corp",
         "dst_type": "Person", "dst_surface": "Jane Roe"},
    ]
    scores = relation_micro_f1(predicted, gold)
    assert scores["tp"] == 1 and scores["fp"] == 1 and scores["fn"] == 1
    assert scores["per_relation_type_recall"]["issues"] == 1.0
    assert scores["per_relation_type_recall"]["employs"] == 0.0


# ---------------------------------------------------------------------------
# KG-AC-P7 — real eval, gated on an explicit live LLM connection (same gate as P2/P3).
# ---------------------------------------------------------------------------
_LLM_CONNECTION = os.environ.get("KG_EVAL_LLM_CONNECTION", "")
_SKIP_NO_LLM = pytest.mark.skipif(
    not _LLM_CONNECTION,
    reason="KG_EVAL_LLM_CONNECTION not set — needs a live LLM connection (Bedrock, reachable backend "
           "decrypt) which is not available in this environment (verified: AWS connection decrypts "
           "but returns 401 outside a real backend session; local Ollama has no chat model)",
)


def _run_llm_graph_relations_over_chunks(chunks_gold, pack, client):
    """Per-chunk, failure-isolated (mirrors test_entity_eval.py's entity version — MEASUREMENT
    harness only; production now propagates a tool-call failure uncaught, KG-AC-P3 v6). Evolve v6:
    reuses the REAL complete_tool (Bedrock Converse tool-use). Returns predicted relation dicts +
    tool-call-valid/total counts (shared metric with KG-AC-P3, not re-gated here — proven in
    test_entity_eval.py). Evolve v13 (KG-AC-71): relations reference entities by src_id/dst_id
    (0-based position in this SAME response's entities[]) — resolved here the same way
    `LlmGraphStrategy.extract()` resolves them in production, so this measurement harness stays
    faithful to what actually ships."""
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
        by_raw_index = {}
        for raw_idx, item in enumerate(data.get("entities") or []):
            etype, surface = item.get("type"), item.get("surface")
            if etype and surface:
                by_raw_index[raw_idx] = (surface, etype)
        for item in (data.get("relations") or []):
            rtype = item.get("type")
            if not rtype:
                continue
            src = by_raw_index.get(item.get("src_id"))
            dst = by_raw_index.get(item.get("dst_id"))
            if src is None or dst is None:  # KG-AC-71: unresolved reference, not scored as a prediction
                continue
            predicted.append({
                "chunk_id": c["chunk_id"], "relation_type": rtype,
                "src_surface": src[0], "src_type": src[1],
                "dst_surface": dst[0], "dst_type": dst[1],
            })
    return predicted, valid, total


@_SKIP_NO_LLM
@pytest.mark.ac("KG-AC-P7")
def test_llm_relation_micro_f1_vs_baseline():
    from clients import build_llm_client

    chunks_gold = _load_chunks()
    relations_gold = _load_relations()
    fibo = load_pack("fibo_core")
    client = build_llm_client(_LLM_CONNECTION)

    predicted, _valid, _total = _run_llm_graph_relations_over_chunks(chunks_gold, fibo, client)
    scores = relation_micro_f1(predicted, relations_gold)
    print(f"\nKG LLM RELATION MICRO-F1: {json.dumps(scores, indent=2)}")

    base = _baseline()
    if base is None:
        assert scores["f1"] >= 0.55, scores
    else:
        assert scores["f1"] >= 0.55, scores
        assert scores["f1"] >= base["f1"] - 0.02, f"regression: {scores} vs baseline {base}"


# ---------------------------------------------------------------------------
# KG-AC-P8 prep — run-to-run variance, generate mode ONLY (report-only, no threshold).
# Classify mode (Workstream L, gated behind this file per the v10 sequencing gate) doesn't exist
# yet, so this cannot be the full KG-AC-P8 proof (that needs classify vs generate, L4) — it
# establishes generate mode's OWN stability baseline now, so L4 has something to compare against
# the moment llm_classify.py (L3) lands. Deliberately NOT @pytest.mark.ac("KG-AC-P8") — marking it
# would falsely claim an AC that isn't provable until the classify side exists (CB-OBS-15 lesson:
# don't let a marker outrun what the test actually proves).
# ---------------------------------------------------------------------------
@_SKIP_NO_LLM
def test_llm_relation_generate_mode_edge_set_variance():
    from clients import build_llm_client

    chunks_gold = _load_chunks()
    fibo = load_pack("fibo_core")
    client = build_llm_client(_LLM_CONNECTION)

    runs = []
    for _ in range(3):
        predicted, _valid, _total = _run_llm_graph_relations_over_chunks(chunks_gold, fibo, client)
        runs.append(predicted)

    scores = relation_edge_set_variance(runs)
    print(f"\nKG LLM RELATION GENERATE-MODE EDGE-SET VARIANCE (report-only, KG-AC-P8 prep): "
          f"{json.dumps(scores, indent=2)}")


# ---------------------------------------------------------------------------
# L4 (spec v10, KG-AC-P8): classify vs generate on the golden slice — the eval this whole
# sequencing gate (A5 + C13) was blocking. Now that L1-L3 (classify mode itself) exist, this can
# finally exercise it end-to-end against the frozen dataset.
#
# **Doubly asset-gated (beyond every eval above):** classify mode needs a live LLM connection (to
# do the classifying) AND the by-copy spaCy model (KG-AC-60's sentence-boundary computation, needed
# regardless of `engine`) — the FIRST eval in this worker to need both at once. Neither is
# available in this sandbox (see this file's and test_entity_eval.py's module docstrings for the
# verified-absent evidence) — gated with a clear skip reason, never a fabricated pass. The
# comparison/variance LOGIC reuses the already-self-tested `relation_micro_f1`/
# `relation_edge_set_variance` above — no new scoring math, so no new self-test needed here.
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
_SKIP_NO_LLM_OR_SPACY = pytest.mark.skipif(
    not (_LLM_CONNECTION and _SPACY_READY),
    reason="KG-AC-P8 needs BOTH a live LLM connection (classify mode's classifying call) AND the "
           "by-copy spaCy model (KG-AC-60 sentence boundaries, required regardless of engine) — "
           "neither is available in this sandbox (verified: see test_entity_eval.py's and this "
           "file's module docstrings for the exact evidence)",
)


def _run_llm_classify_relations_over_chunks(chunks_gold, pack, client, nlp):
    """Classify mode end-to-end, at the SAME Relation-level granularity as
    `_run_llm_graph_relations_over_chunks` above (pre-build_edge_records — an apples-to-apples
    comparison against generate mode, which is also scored before the uid/DB-row transform):
    entities via the engine=llm one-pass call, merged, candidate pairs enumerated over the merged
    set (KG-AC-60), then classified (KG-AC-61/62). Mirrors `strategies.base.run_pipeline`'s wiring
    but stops short of building DB rows, matching this file's evaluation granularity."""
    chunks = [Chunk(c["chunk_id"], c["text"]) for c in chunks_gold]
    cfg = ExtractionConfig(engine="llm", ontology_pack=pack.name, relation_strategy="classify")
    candidates, _raw_relations = run_graph_extraction(chunks, cfg, pack, spacy_nlp=nlp, llm_client=client)
    kept, _unmapped = filter_closed_vocab(candidates, cfg, pack)
    assign_occurrence_indices(kept)
    merged = merge_candidates(kept)

    sentence_spans = {ch.chunk_id: [(s.start_char, s.end_char) for s in nlp(ch.text).sents] for ch in chunks}
    pairs = enumerate_candidate_pairs(merged, sentence_spans, pack)
    chunk_text_by_id = {ch.chunk_id: ch.text for ch in chunks}
    relations = LlmClassifyStrategy(llm_client=client).classify(pairs, chunk_text_by_id)

    predicted = [{"chunk_id": r.source_chunk_id, "relation_type": r.relation_type,
                 "src_type": r.src_type, "src_surface": r.src_surface,
                 "dst_type": r.dst_type, "dst_surface": r.dst_surface} for r in relations]
    return predicted, relations


@_SKIP_NO_LLM_OR_SPACY
@pytest.mark.ac("KG-AC-P8")
def test_classify_relation_micro_f1_vs_generate_baseline():
    from clients import build_llm_client

    chunks_gold = _load_chunks()
    relations_gold = _load_relations()
    fibo = load_pack("fibo_core")
    client = build_llm_client(_LLM_CONNECTION)
    nlp = load_spacy_model(os.environ["SPACY_MODEL_PATH"])

    predicted, _raw = _run_llm_classify_relations_over_chunks(chunks_gold, fibo, client, nlp)
    scores = relation_micro_f1(predicted, relations_gold)
    print(f"\nKG LLM CLASSIFY RELATION MICRO-F1: {json.dumps(scores, indent=2)}")

    base = _baseline()  # kg_relation_llm_fibo_core — the COMMITTED generate-mode baseline
    if base is None:
        # no generate baseline recorded yet (P7 has never run for real in any environment) — floor
        # only, same 0.55 floor P7 itself uses, until a real generate score exists to compare against.
        assert scores["f1"] >= 0.55, scores
    else:
        assert scores["f1"] >= base["f1"], f"classify regressed vs generate baseline: {scores} vs {base}"


@_SKIP_NO_LLM_OR_SPACY
@pytest.mark.ac("KG-AC-P8")
def test_classify_relation_edge_set_variance_side_by_side_with_generate():
    # "variance reported" — classify mode's OWN run-to-run stability, directly comparable to
    # generate mode's (test_llm_relation_generate_mode_edge_set_variance above). Report-only.
    from clients import build_llm_client

    chunks_gold = _load_chunks()
    fibo = load_pack("fibo_core")
    client = build_llm_client(_LLM_CONNECTION)
    nlp = load_spacy_model(os.environ["SPACY_MODEL_PATH"])

    runs = []
    for _ in range(3):
        _predicted, raw_relations = _run_llm_classify_relations_over_chunks(chunks_gold, fibo, client, nlp)
        runs.append([{"chunk_id": r.source_chunk_id, "relation_type": r.relation_type,
                     "src_type": r.src_type, "src_surface": r.src_surface,
                     "dst_type": r.dst_type, "dst_surface": r.dst_surface} for r in raw_relations])

    scores = relation_edge_set_variance(runs)
    print(f"\nKG LLM RELATION CLASSIFY-MODE EDGE-SET VARIANCE (report-only, KG-AC-P8): "
          f"{json.dumps(scores, indent=2)}")
    # report-only per KG-AC-P8 ("variance reported", no threshold set) -- no assertion on the value


# ---------------------------------------------------------------------------
# N8 (spec v12, KG-AC-P9/P10): entity_scoped vs classify/generate, and k=3 vs k=1 stability.
# Gated identically to KG-AC-P8's classify tests above (requirements.md's own "same gate as
# KG-AC-P9" wording, KG-AC-P10) -- entity_scoped itself needs no sentence-boundary nlp (KG-AC-65's
# whole point is no same-sentence gate), but the frozen AC text ties both new v12 evals to the same
# combined live-LLM + by-copy-spaCy gate as KG-AC-P8 for environment-gating symmetry across every
# v12 relation-mode eval in this file -- honored as written, not loosened based on this file's own
# read of what entity_scoped structurally requires.
# ---------------------------------------------------------------------------
def _run_llm_entity_scoped_relations_over_chunks(chunks_gold, pack, client):
    """entity_scoped mode end-to-end, at the SAME Relation-level granularity as
    `_run_llm_classify_relations_over_chunks`/`_run_llm_graph_relations_over_chunks` above
    (pre-build_edge_records). Mirrors `strategies.base.run_pipeline`'s entity_scoped branch: one
    call per chunk over the FINAL merged entity set, no candidate-pair enumeration, no sentence
    boundaries needed."""
    chunks = [Chunk(c["chunk_id"], c["text"]) for c in chunks_gold]
    cfg = ExtractionConfig(engine="llm", ontology_pack=pack.name, relation_strategy="entity_scoped")
    candidates, _raw_relations = run_graph_extraction(chunks, cfg, pack, llm_client=client)
    kept, _unmapped = filter_closed_vocab(candidates, cfg, pack)
    assign_occurrence_indices(kept)
    merged = merge_candidates(kept)

    chunk_text_by_id = {ch.chunk_id: ch.text for ch in chunks}
    relations = LlmEntityScopedStrategy(llm_client=client).extract(merged, chunk_text_by_id, pack)

    predicted = [{"chunk_id": r.source_chunk_id, "relation_type": r.relation_type,
                 "src_type": r.src_type, "src_surface": r.src_surface,
                 "dst_type": r.dst_type, "dst_surface": r.dst_surface} for r in relations]
    return predicted, relations


@_SKIP_NO_LLM_OR_SPACY
@pytest.mark.ac("KG-AC-P9")
def test_entity_scoped_relation_micro_f1_vs_classify_and_generate_baseline():
    from clients import build_llm_client

    chunks_gold = _load_chunks()
    relations_gold = _load_relations()
    fibo = load_pack("fibo_core")
    client = build_llm_client(_LLM_CONNECTION)

    predicted, _raw = _run_llm_entity_scoped_relations_over_chunks(chunks_gold, fibo, client)
    scores = relation_micro_f1(predicted, relations_gold)
    print(f"\nKG LLM ENTITY_SCOPED RELATION MICRO-F1: {json.dumps(scores, indent=2)}")

    base = _baseline()  # kg_relation_llm_fibo_core -- the COMMITTED generate-mode baseline
    classify_base = None
    if BASELINES.exists():
        classify_base = json.loads(BASELINES.read_text(encoding="utf-8")).get("kg_relation_classify_fibo_core")

    # KG-AC-P9: entity_scoped exists to raise recall over BOTH other modes -- failing to beat
    # either means the mode does not justify its added per-chunk-call cost. No committed baseline
    # for either sibling mode yet (neither has run for real in any environment) -- floor only, the
    # same 0.55 floor P7/P8 themselves use until real numbers exist to compare against.
    assert scores["f1"] >= 0.55, scores
    if base is not None:
        assert scores["f1"] >= base["f1"], f"entity_scoped below generate baseline: {scores} vs {base}"
    if classify_base is not None:
        assert scores["f1"] >= classify_base["f1"], (
            f"entity_scoped below classify baseline: {scores} vs {classify_base}")


@_SKIP_NO_LLM_OR_SPACY
@pytest.mark.ac("KG-AC-P9")
def test_entity_scoped_relation_edge_set_variance_side_by_side():
    # "variance reported" alongside generate/classify's own (test_llm_relation_generate_mode_edge_
    # set_variance / test_classify_relation_edge_set_variance_side_by_side_with_generate above).
    # Report-only, same as those two.
    from clients import build_llm_client

    chunks_gold = _load_chunks()
    fibo = load_pack("fibo_core")
    client = build_llm_client(_LLM_CONNECTION)

    runs = []
    for _ in range(3):
        _predicted, raw_relations = _run_llm_entity_scoped_relations_over_chunks(chunks_gold, fibo, client)
        runs.append([{"chunk_id": r.source_chunk_id, "relation_type": r.relation_type,
                     "src_type": r.src_type, "src_surface": r.src_surface,
                     "dst_type": r.dst_type, "dst_surface": r.dst_surface} for r in raw_relations])

    scores = relation_edge_set_variance(runs)
    print(f"\nKG LLM RELATION ENTITY_SCOPED-MODE EDGE-SET VARIANCE (report-only): "
          f"{json.dumps(scores, indent=2)}")


@_SKIP_NO_LLM_OR_SPACY
@pytest.mark.ac("KG-AC-P10")
def test_self_consistency_k3_instability_below_k1():
    # KG-AC-P10: run-to-run relation edge-set instability at relation_self_consistency_k=3 versus
    # the same config at k=1, on entity_scoped mode (v12's own new mode -- the most natural surface
    # to prove voting actually helps, since it is the mode this workstream added the voting FOR).
    # k=1 is a single run (no voting -- the existing 3-independent-runs variance measurement above,
    # `test_entity_scoped_relation_edge_set_variance_side_by_side`, IS the k=1 baseline: 3 raw,
    # unvoted extraction runs over the same chunks). k=3 instability is measured the same way but
    # over 3 INDEPENDENT ***voted*** outputs (each itself the majority vote of its own 3 sub-runs)
    # -- voting should make each voted output closer to the others than raw single runs are.
    from clients import build_llm_client
    from core import vote_relations

    chunks_gold = _load_chunks()
    fibo = load_pack("fibo_core")
    client = build_llm_client(_LLM_CONNECTION)

    def _one_voted_run(k: int):
        sub_runs = [_run_llm_entity_scoped_relations_over_chunks(chunks_gold, fibo, client)[1]
                   for _ in range(k)]
        return vote_relations(sub_runs, k)

    k1_runs = [[{"chunk_id": r.source_chunk_id, "relation_type": r.relation_type,
                "src_type": r.src_type, "src_surface": r.src_surface,
                "dst_type": r.dst_type, "dst_surface": r.dst_surface}
               for r in _one_voted_run(1)] for _ in range(3)]
    k3_runs = [[{"chunk_id": r.source_chunk_id, "relation_type": r.relation_type,
                "src_type": r.src_type, "src_surface": r.src_surface,
                "dst_type": r.dst_type, "dst_surface": r.dst_surface}
               for r in _one_voted_run(3)] for _ in range(3)]

    k1_scores = relation_edge_set_variance(k1_runs)
    k3_scores = relation_edge_set_variance(k3_runs)
    print(f"\nKG SELF-CONSISTENCY INSTABILITY k=1: {json.dumps(k1_scores, indent=2)}")
    print(f"KG SELF-CONSISTENCY INSTABILITY k=3: {json.dumps(k3_scores, indent=2)}")

    assert k3_scores["instability"] < k1_scores["instability"], (
        f"k=3 voting did not reduce instability: k=3 {k3_scores} vs k=1 {k1_scores}")
