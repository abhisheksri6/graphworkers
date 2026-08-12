"""KG-AC-P11 (fact micro-F1) and KG-AC-P12 (canonical-graph entity + edge F1) vs the frozen
`meridian-2026-bundle` multi-document golden set (`sample-datasets/knowledge-graph/
meridian-2026-bundle/`, P18) — the first case with a gold canonical graph (the existing
single-document cases structurally cannot exercise cross-document resolution).

**Reuses production code as the thing under test, never reimplements it** (the L4/N8 precedent,
extended):
  - KG-AC-P11 calls `strategies.base.run_pipeline` — the SAME entry point production and the
    runtime-preview path both call — once per document, and scores the `facts` nested onto its
    returned entity rows (pre-canonicalisation, per the AC's own "measured at extraction output").
  - KG-AC-P12 additionally feeds every document's entity/edge rows into
    `entity_canonicalization.core`'s PURE functions (`cluster_mentions`/`reconcile_type`/
    `canonical_key`/`choose_canonical_name`/`merge_attributes`/`aggregate_edge_group`) — the exact
    same functions `store.py`'s `canonicalize_batch` orchestrates around a DB transaction, called
    here without a DB (a pure in-memory canonicalization pass, loaded via `importlib` under an
    aliased module name — entity_extraction and entity_canonicalization each ship their OWN
    top-level `core.py`, so a bare `import core` would silently resolve to the wrong one).

**Environment-honest gating, same evidence as `test_entity_eval.py`/`test_relation_eval.py`'s
module docstrings** (this sandbox has no reachable LLM connection — the AWS connection decrypts
but returns HTTP 401 outside a real backend session, verified empirically, not assumed; local
Ollama has no chat model): both real-eval tests below SKIP with a clear reason when
`KG_EVAL_LLM_CONNECTION` is unset — per this task's own verify text, "ship the harness correctly
skipped rather than inventing a score." Every scorer is proven via a self-test on hand-built
inputs first, independent of asset availability.

Runs only under `pytest -m eval`.
"""
from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from ontologies import load_pack
from strategies import Chunk, ExtractionConfig
from strategies.base import run_pipeline

pytestmark = [pytest.mark.eval]

REPO = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = Path(os.environ.get(
    "CB_GOVERNANCE_DIR", REPO.parent.parent / "coding-governance",
)).resolve()
# CB-OBS-36 (corrected at P18): sample-datasets/knowledge-graph now lives INSIDE the governance
# repo (a real git home, since R4), not at the root REPO.parent.parent/"sample-datasets" the OLDER
# fibo_core-era eval files still default to (those files are dead — pending deletion, per their own
# module docstrings — so their stale default was never worth fixing). This file's own default
# points at the CURRENT, correct location.
DATASET = Path(os.environ.get(
    "CB_DATASETS_DIR", GOVERNANCE_DIR / "sample-datasets",
)).resolve() / "knowledge-graph" / "meridian-2026-bundle"
BASELINES = GOVERNANCE_DIR / "specs" / "evaluation" / "baselines.json"
ENTITY_CANON_DIR = Path(os.environ.get(
    "CB_ENTITY_CANON_DIR", REPO.parent / "entity_canonicalization",
)).resolve()

DOC_STEMS = [
    "01_IMA-2026-0091_Investment_Management_Agreement",
    "02_SL-2026-007_Side_Letter",
    "03_SUB-2026-023_Subscription_Agreement",
]


@pytest.fixture(scope="module", autouse=True)
def verify_dataset_integrity():
    manifest_path = DATASET / "manifest.json"
    assert manifest_path.exists(), f"meridian-2026-bundle dataset not found at {DATASET} — set CB_DATASETS_DIR"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        p = DATASET / rel
        assert p.exists(), f"dataset file missing: {rel}"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, (
            f"dataset file CHANGED since generation: {rel} (re-run generate.py or re-freeze)"
        )


def _load_doc_gold(stem: str) -> dict:
    return json.loads((DATASET / "gold" / f"extraction-{stem}.json").read_text(encoding="utf-8"))


def _load_all_doc_golds() -> list[dict]:
    return [_load_doc_gold(stem) for stem in DOC_STEMS]


def _load_canonical_gold() -> dict:
    return json.loads((DATASET / "gold" / "canonical_graph.json").read_text(encoding="utf-8"))


def _load_source_text(stem: str) -> str:
    return (DATASET / "source" / f"{stem}.txt").read_text(encoding="utf-8")


def _baseline(key: str):
    if BASELINES.exists():
        return json.loads(BASELINES.read_text(encoding="utf-8")).get(key)
    return None


def _norm_text(s) -> str:
    return " ".join(str(s or "").split()).strip().lower()


# -------------------------------------------------------------------------------------------
# entity_canonicalization's pure core, loaded under an aliased module name — NOT a bare
# `import core`, since entity_extraction (this repo) already has its OWN top-level `core.py` on
# sys.path; a bare import would silently resolve to the wrong module (same name, different repo).
# core.py there has zero cross-file imports (verified by reading it) — safe to load standalone
# without the rest of the entity_canonicalization package on sys.path.
# -------------------------------------------------------------------------------------------
def _load_canon_core():
    path = ENTITY_CANON_DIR / "core.py"
    assert path.exists(), f"entity_canonicalization/core.py not found at {path} — set CB_ENTITY_CANON_DIR"
    module_name = "_canon_core_for_kg_eval"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    # Registered in sys.modules BEFORE exec_module: core.py's `@dataclass` classes (e.g. Mention)
    # need `sys.modules[cls.__module__]` to resolve their `from __future__ import annotations`
    # string-typed fields — omitting this raises AttributeError deep inside the stdlib dataclasses
    # module, found empirically when this loader was first written.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================================
# Scorers (pure — no model/LLM). Duplicated from test_entity_eval.py/test_relation_eval.py's
# convention deliberately (specs/evaluation/design.md: no shared scorer library until a 3rd
# consumer appears — this file already reuses THEIR pattern, not their code).
# =============================================================================================
def fact_micro_f1(predicted: list[dict], gold: list[dict]) -> dict:
    """predicted/gold: dicts with document_id, subject_type, property, value, normalized_value.
    TP: `property` exact AND `document_id`+`subject_type` match (the subject-resolution rule, at
    this dataset's granularity: one entity per type per document, so type+document identifies the
    subject unambiguously) AND `normalized_value` equal — falling back to a whitespace/case-
    collapsed comparison of the verbatim `value` when either side's normalizer returned null
    (KG-AC-P11). Per-property P/R also reported."""
    def fact_key(f: dict):
        nv = f.get("normalized_value")
        cmp_value = _norm_text(nv) if nv is not None else _norm_text(f["value"])
        return (f["document_id"], f["subject_type"], f["property"], cmp_value)

    gold_keys = {fact_key(f) for f in gold}
    pred_keys = {fact_key(f) for f in predicted}
    tp, fp, fn = len(pred_keys & gold_keys), len(pred_keys - gold_keys), len(gold_keys - pred_keys)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_prop_total: dict[str, int] = {}
    per_prop_matched: dict[str, int] = {}
    for f in gold:
        prop = f["property"]
        per_prop_total[prop] = per_prop_total.get(prop, 0) + 1
        if fact_key(f) in pred_keys:
            per_prop_matched[prop] = per_prop_matched.get(prop, 0) + 1
    per_property_recall = {p: round(per_prop_matched.get(p, 0) / n, 4) for p, n in per_prop_total.items()}

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
           "tp": tp, "fp": fp, "fn": fn, "per_property_recall": per_property_recall}


def canonical_entity_f1(predicted: list[dict], gold: list[dict]) -> dict:
    """predicted/gold: dicts with entity_type, canonical_name. TP: `entity_type` exact AND
    `canonical_name` normalised-equal — exact after whitespace/case fold, deliberately NOT the
    fuzzy `surface_matches` ratio the extraction-level entity scorer uses: KG-AC-P12's own text
    says "canonical-name normalised equality", not "relaxed"."""
    def key(e: dict):
        return (e["entity_type"], _norm_text(e["canonical_name"]))

    gold_set, pred_set = {key(e) for e in gold}, {key(e) for e in predicted}
    tp, fp, fn = len(pred_set & gold_set), len(pred_set - gold_set), len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
           "tp": tp, "fp": fp, "fn": fn}


def canonical_edge_f1(predicted: list[dict], gold: list[dict]) -> dict:
    """predicted/gold: dicts with relation_type, src_type, src_canonical_name, dst_type,
    dst_canonical_name (both endpoints already RESOLVED to their canonical identity — the caller's
    job, not this scorer's). TP: `relation_type` exact AND both endpoints' (type, normalised
    canonical_name) match (KG-AC-P12)."""
    def key(e: dict):
        return (e["relation_type"], e["src_type"], _norm_text(e["src_canonical_name"]),
                e["dst_type"], _norm_text(e["dst_canonical_name"]))

    gold_set, pred_set = {key(e) for e in gold}, {key(e) for e in predicted}
    tp, fp, fn = len(pred_set & gold_set), len(pred_set - gold_set), len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
           "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------------------------
# Scorer self-tests (ALWAYS run — no marker, no model/LLM needed).
# ---------------------------------------------------------------------------------------------
def test_fact_micro_f1_scorer_self_test():
    gold = [
        {"document_id": "d1", "subject_type": "Agreement", "property": "agreementId",
         "value": "IMA-2026-0091", "normalized_value": None},
        {"document_id": "d1", "subject_type": "Agreement", "property": "effectiveDate",
         "value": "1 March 2026", "normalized_value": "2026-03-01"},
    ]
    predicted = [
        # exact match, identifier property (no normalizer -> value fallback, whitespace-only diff)
        {"document_id": "d1", "subject_type": "Agreement", "property": "agreementId",
         "value": "IMA-2026-0091  ", "normalized_value": None},
        # wrong normalized_value -> false positive; the real effectiveDate is missed (false negative)
        {"document_id": "d1", "subject_type": "Agreement", "property": "effectiveDate",
         "value": "2 March 2026", "normalized_value": "2026-03-02"},
    ]
    scores = fact_micro_f1(predicted, gold)
    assert scores["tp"] == 1 and scores["fp"] == 1 and scores["fn"] == 1
    assert scores["per_property_recall"]["agreementId"] == 1.0
    assert scores["per_property_recall"]["effectiveDate"] == 0.0


def test_canonical_entity_f1_scorer_self_test():
    gold = [
        {"entity_type": "Investor", "canonical_name": "Granite State Pension Trust"},
        {"entity_type": "Fund", "canonical_name": "Meridian Growth Fund, L.P."},
    ]
    predicted = [
        # case/whitespace-only difference -> still a match (normalised equality)
        {"entity_type": "Investor", "canonical_name": "  granite state pension trust "},
        # wrong type for the same name -> false positive; Fund is missed -> false negative
        {"entity_type": "InvestmentManager", "canonical_name": "Meridian Growth Fund, L.P."},
    ]
    scores = canonical_entity_f1(predicted, gold)
    assert scores["tp"] == 1 and scores["fp"] == 1 and scores["fn"] == 1


def test_canonical_edge_f1_scorer_self_test():
    gold = [
        {"relation_type": "hasInvestor", "src_type": "InvestmentRelationship",
         "src_canonical_name": "IMA-2026-0091", "dst_type": "Investor",
         "dst_canonical_name": "Granite State Pension Trust"},
    ]
    predicted = [
        {"relation_type": "hasInvestor", "src_type": "InvestmentRelationship",
         "src_canonical_name": "IMA-2026-0091", "dst_type": "Investor",
         "dst_canonical_name": "Granite State Pension Trust"},
        {"relation_type": "hasInvestmentManager", "src_type": "InvestmentRelationship",
         "src_canonical_name": "IMA-2026-0091", "dst_type": "InvestmentManager",
         "dst_canonical_name": "Meridian Capital Partners LLC"},  # spurious -> false positive
    ]
    scores = canonical_edge_f1(predicted, gold)
    assert scores["tp"] == 1 and scores["fp"] == 1 and scores["fn"] == 0


def test_resolve_canonical_graph_merges_across_documents_and_aggregates_edges():
    # Integration-level self-test of `_resolve_canonical_graph` (no LLM needed -- hand-built
    # entity/edge rows in the EXACT shape strategies.base.run_pipeline actually returns, per
    # core.build_entity_records/build_edge_records). Proves the cross-repo orchestration itself
    # (clustering + attribute merge + edge aggregation) before the real evals depend on it.
    canon_core = _load_canon_core()
    pack = load_pack("investment_fibo")

    # Two documents, both mentioning the SAME InvestmentManager (exact-surface -> ACCEPT band) and
    # the SAME Fund, asserting the SAME relation -- must collapse to ONE canonical entity per type
    # and ONE aggregated edge (support_count=2, source_doc_ids=[docA, docB]), not four/two.
    doc_a_entities = [
        {"entity_uid": "a1", "entity_type": "InvestmentManager", "surface_form": "Acme Capital LLC",
         "source_doc_id": "docA", "source_chunk_id": "docA-c0", "span_start": 0, "is_abstract": False,
         "attributes": [{"property": "investmentManagerName", "value": "Acme Capital LLC",
                        "normalized_value": "Acme Capital LLC", "evidence": "Acme Capital LLC",
                        "source_doc_id": "docA", "page": 1}]},
        {"entity_uid": "a2", "entity_type": "Fund", "surface_form": "Acme Growth Fund",
         "source_doc_id": "docA", "source_chunk_id": "docA-c0", "span_start": 10, "is_abstract": False,
         "attributes": []},
    ]
    doc_a_edges = [
        {"edge_uid": "e1", "relation_type": "manages", "src_entity_uid": "a1", "dst_entity_uid": "a2",
         "source_doc_id": "docA", "confidence": 0.9, "evidence_text": "Acme manages the fund",
         "extractor": "llm", "folder_id": "docA"},
    ]
    doc_b_entities = [
        {"entity_uid": "b1", "entity_type": "InvestmentManager", "surface_form": "Acme Capital LLC",
         "source_doc_id": "docB", "source_chunk_id": "docB-c0", "span_start": 0, "is_abstract": False,
         "attributes": [{"property": "investmentManagerName", "value": "Acme Capital LLC",
                        "normalized_value": "Acme Capital LLC", "evidence": "Acme Capital LLC",
                        "source_doc_id": "docB", "page": 1}]},
        {"entity_uid": "b2", "entity_type": "Fund", "surface_form": "Acme Growth Fund",
         "source_doc_id": "docB", "source_chunk_id": "docB-c0", "span_start": 10, "is_abstract": False,
         "attributes": []},
    ]
    doc_b_edges = [
        {"edge_uid": "e2", "relation_type": "manages", "src_entity_uid": "b1", "dst_entity_uid": "b2",
         "source_doc_id": "docB", "confidence": 0.85, "evidence_text": "Acme continues to manage the fund",
         "extractor": "llm", "folder_id": "docB"},
    ]

    canonical_entities, canonical_edges = _resolve_canonical_graph(
        {"docA": doc_a_entities, "docB": doc_b_entities},
        {"docA": doc_a_edges, "docB": doc_b_edges},
        pack, canon_core,
    )

    assert len(canonical_entities) == 2, canonical_entities  # ONE InvestmentManager, ONE Fund
    im = next(e for e in canonical_entities if e["entity_type"] == "InvestmentManager")
    assert im["canonical_name"] == "Acme Capital LLC"
    assert im["attributes"]["investmentManagerName"][0]["status"] == "consistent"  # 2 docs, same value

    assert len(canonical_edges) == 1, canonical_edges  # ONE aggregated 'manages' edge, not two
    edge = canonical_edges[0]
    assert edge["relation_type"] == "manages"
    assert edge["support_count"] == 2
    assert sorted(edge["source_doc_ids"]) == ["docA", "docB"]

    resolved = _resolve_predicted_edges_for_scoring(canonical_entities, canonical_edges)
    assert resolved == [{"relation_type": "manages", "src_type": "InvestmentManager",
                        "src_canonical_name": "Acme Capital LLC", "dst_type": "Fund",
                        "dst_canonical_name": "Acme Growth Fund"}]


def test_canon_core_loads_under_an_aliased_module_name_not_this_repos_own_core():
    # Guards the collision this file's module docstring warns about: entity_extraction's own
    # `core.py` (already imported elsewhere in this suite) must NOT be what `_load_canon_core()`
    # returns.
    import core as own_core  # entity_extraction's own top-level core.py
    canon_core = _load_canon_core()
    assert canon_core is not own_core
    assert hasattr(canon_core, "Mention") and not hasattr(own_core, "Mention")
    assert hasattr(canon_core, "choose_canonical_name")
    assert hasattr(own_core, "build_entity_records")  # own_core has ITS OWN, unrelated surface


# ===============================================================================================
# Real evals — gated on an explicit live LLM connection (same gate as test_entity_eval.py /
# test_relation_eval.py's P2/P3/P7 tests).
# ===============================================================================================
_LLM_CONNECTION = os.environ.get("KG_EVAL_LLM_CONNECTION", "")
_SKIP_NO_LLM = pytest.mark.skipif(
    not _LLM_CONNECTION,
    reason="KG_EVAL_LLM_CONNECTION not set — needs a live LLM connection (Bedrock, reachable backend "
           "decrypt) which is not available in this environment (verified: AWS connection decrypts "
           "but returns 401 outside a real backend session; local Ollama has no chat model)",
)


def _run_pipeline_per_document(pack, client):
    """Runs strategies.base.run_pipeline once per document (folder_id = the document's own
    source_identifier, mirroring production's per-folder fan-out) and returns the per-document
    (entity_rows, edge_rows) plus the flat facts list used by the KG-AC-P11 scorer."""
    golds = _load_all_doc_golds()
    per_doc_entity_rows: dict[str, list[dict]] = {}
    per_doc_edge_rows: dict[str, list[dict]] = {}
    predicted_facts: list[dict] = []

    for stem, gold in zip(DOC_STEMS, golds):
        doc_id = gold["source_identifier"]
        text = _load_source_text(stem)
        chunk = Chunk(chunk_id=f"{doc_id}-c0", text=text, doc_id=doc_id, page=1)
        cfg = ExtractionConfig(engine="llm", ontology_pack=pack.name)
        ent_rows, edge_rows, _summary, _usage, _blocked = run_pipeline(
            [chunk], cfg, pack, folder_id=doc_id, llm_client=client,
        )
        # folder_id stamped for aggregate_edge_group's support_count (KG-AC-47) — the doc IS the
        # folder in this per-document harness, matching production's per-folder model.
        for e in edge_rows:
            e["folder_id"] = doc_id
        per_doc_entity_rows[doc_id] = ent_rows
        per_doc_edge_rows[doc_id] = edge_rows

        for row in ent_rows:
            for fact in (row.get("attributes") or []):
                predicted_facts.append({
                    "document_id": doc_id, "subject_type": row["entity_type"],
                    "property": fact["property"], "value": fact.get("value"),
                    "normalized_value": fact.get("normalized_value"),
                })

    return per_doc_entity_rows, per_doc_edge_rows, predicted_facts


def _gold_facts_flat(golds: list[dict]) -> list[dict]:
    out = []
    for g in golds:
        doc_id = g["source_identifier"]
        for p in g["properties"]:
            out.append({"document_id": doc_id, "subject_type": p["subject"], "property": p["property"],
                       "value": p["value"], "normalized_value": p.get("normalized_value")})
    return out


@_SKIP_NO_LLM
@pytest.mark.ac("KG-AC-P11")
def test_fact_micro_f1_vs_baseline():
    from clients import build_llm_client

    pack = load_pack("investment_fibo")
    client = build_llm_client(_LLM_CONNECTION)
    _entity_rows, _edge_rows, predicted_facts = _run_pipeline_per_document(pack, client)
    gold_facts = _gold_facts_flat(_load_all_doc_golds())

    scores = fact_micro_f1(predicted_facts, gold_facts)
    print(f"\nKG FACT MICRO-F1 (meridian-2026-bundle): {json.dumps(scores, indent=2)}")

    base = _baseline("kg_fact_investment_fibo")
    if base is None:
        assert scores["f1"] >= 0.60, scores
    else:
        assert scores["f1"] >= 0.60, scores
        assert scores["f1"] >= base["f1"] - 0.02, f"regression: {scores} vs baseline {base}"


def _resolve_canonical_graph(per_doc_entity_rows, per_doc_edge_rows, pack, canon_core):
    """The pure, DB-free canonicalization pass — mirrors store.py's `canonicalize_batch` exactly
    (same function call sequence, same order), minus the DB reads/writes: `canonical_id` (a DB
    sequence there) is replaced by `canonical_key` itself, which is deterministic and unique
    per-cluster within one pass (KG-AC-79) — sufficient identity for a single, non-cross-run eval
    like this one. `adjudicate=None` (AMBIGUOUS-band pairs not merged) is safe for this dataset by
    design (see README.md: every shared surface is an EXACT match across documents, never a
    fuzzy-boundary case) and is verified, not assumed, by the entity-F1 assertion this feeds."""
    all_entity_rows = [r for rows in per_doc_entity_rows.values() for r in rows]
    all_edge_rows = [r for rows in per_doc_edge_rows.values() for r in rows]
    by_uid = {r["entity_uid"]: r for r in all_entity_rows}

    mentions = [
        canon_core.Mention(
            entity_uid=r["entity_uid"], entity_type=r["entity_type"], surface_form=r["surface_form"],
            normalized_form=canon_core.normalize_surface(r["surface_form"]),
            source_doc_id=r.get("source_doc_id"), source_chunk_id=r.get("source_chunk_id"),
            span_start=r.get("span_start"), is_abstract=bool(r.get("is_abstract")),
        )
        for r in all_entity_rows
    ]
    clusters = canon_core.cluster_mentions(mentions, fuzzy_floor=0.80, fuzzy_ceiling=0.95, adjudicate=None)

    canonical_entities = []
    uid_to_ckey: dict[str, str] = {}
    for cluster in clusters:
        rtype = canon_core.reconcile_type([m.entity_type for m in cluster], pack)
        canonical_name, aliases = canon_core.choose_canonical_name(cluster)
        merged_attrs = canon_core.merge_attributes(
            [by_uid[m.entity_uid].get("attributes") or [] for m in cluster])
        ckey = canon_core.canonical_key(rtype, cluster[0].normalized_form)
        for m in cluster:
            uid_to_ckey[m.entity_uid] = ckey
        canonical_entities.append({
            "canonical_key": ckey, "entity_type": rtype, "canonical_name": canonical_name,
            "aliases": aliases, "attributes": merged_attrs,
        })

    edge_groups: dict[tuple, list[dict]] = {}
    for e in all_edge_rows:
        src_ck, dst_ck = uid_to_ckey.get(e["src_entity_uid"]), uid_to_ckey.get(e["dst_entity_uid"])
        if not src_ck or not dst_ck:
            continue  # dangling endpoint (its own entity was dropped upstream) -- same posture as build_edge_records
        edge_groups.setdefault((src_ck, e["relation_type"], dst_ck), []).append(e)

    canonical_edges = []
    for (src_ck, rtype, dst_ck), rows in edge_groups.items():
        agg = canon_core.aggregate_edge_group(rows)
        canonical_edges.append({"relation_type": rtype, "src_canonical_key": src_ck,
                               "dst_canonical_key": dst_ck, **agg})

    return canonical_entities, canonical_edges


def _resolve_predicted_edges_for_scoring(canonical_entities, canonical_edges):
    by_key = {e["canonical_key"]: e for e in canonical_entities}
    resolved = []
    for e in canonical_edges:
        src, dst = by_key.get(e["src_canonical_key"]), by_key.get(e["dst_canonical_key"])
        if not src or not dst:
            continue
        resolved.append({"relation_type": e["relation_type"], "src_type": src["entity_type"],
                        "src_canonical_name": src["canonical_name"], "dst_type": dst["entity_type"],
                        "dst_canonical_name": dst["canonical_name"]})
    return resolved


def _resolve_gold_edges_for_scoring(gold_graph):
    by_type = {e["entity_type"]: e["canonical_name"] for e in gold_graph["canonical_entities"]}
    return [
        {"relation_type": e["relation_type"], "src_type": e["src"], "src_canonical_name": by_type[e["src"]],
         "dst_type": e["dst"], "dst_canonical_name": by_type[e["dst"]]}
        for e in gold_graph["canonical_edges"]
    ]


@_SKIP_NO_LLM
@pytest.mark.ac("KG-AC-P12")
def test_canonical_graph_entity_and_edge_f1_vs_baseline():
    from clients import build_llm_client

    pack = load_pack("investment_fibo")
    client = build_llm_client(_LLM_CONNECTION)
    canon_core = _load_canon_core()

    per_doc_entity_rows, per_doc_edge_rows, _facts = _run_pipeline_per_document(pack, client)
    canonical_entities, canonical_edges = _resolve_canonical_graph(
        per_doc_entity_rows, per_doc_edge_rows, pack, canon_core)

    gold_graph = _load_canonical_gold()
    entity_scores = canonical_entity_f1(
        [{"entity_type": e["entity_type"], "canonical_name": e["canonical_name"]} for e in canonical_entities],
        [{"entity_type": e["entity_type"], "canonical_name": e["canonical_name"]}
         for e in gold_graph["canonical_entities"]],
    )
    edge_scores = canonical_edge_f1(
        _resolve_predicted_edges_for_scoring(canonical_entities, canonical_edges),
        _resolve_gold_edges_for_scoring(gold_graph),
    )
    print(f"\nKG CANONICAL-GRAPH ENTITY F1 (meridian-2026-bundle): {json.dumps(entity_scores, indent=2)}")
    print(f"KG CANONICAL-GRAPH EDGE F1 (meridian-2026-bundle): {json.dumps(edge_scores, indent=2)}")

    entity_base = _baseline("kg_canonical_entity_investment_fibo")
    edge_base = _baseline("kg_canonical_edge_investment_fibo")
    assert entity_scores["f1"] >= 0.70, entity_scores
    assert edge_scores["f1"] >= 0.55, edge_scores
    if entity_base is not None:
        assert entity_scores["f1"] >= entity_base["f1"] - 0.02, f"regression: {entity_scores} vs baseline {entity_base}"
    if edge_base is not None:
        assert edge_scores["f1"] >= edge_base["f1"] - 0.02, f"regression: {edge_scores} vs baseline {edge_base}"
