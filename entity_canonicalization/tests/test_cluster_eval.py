"""KG-AC-P6 (canonicalization cluster quality) vs the frozen synthetic golden set
(sample-datasets/knowledge-graph/, D4/A5). Pairwise precision/recall/F1 on the labeled duplicate
set. No external dependency (pure Python core — blocking/matching/union-find) — this eval genuinely
RUNS in any environment, unlike the LLM/model-dependent entity/relation evals.

**Adjudicator choice, stated plainly:** the dataset's one duplicate cluster spans both an
exact-normalized ACCEPT pair (fuzzy_score=1.0) and two AMBIGUOUS-band pairs (fuzzy_score=0.840,
between the default floor=0.80/ceiling=0.95) that need adjudication in production. This environment
has no reachable LLM (verified: the AWS Bedrock connection decrypts but returns 401 unauthenticated
outside a real backend session, and no local chat model is installed). Two runs are scored:
  - **gold-oracle adjudicator** (adjudicate(a,b) = the TRUE same-cluster answer from the labeled
    set) — isolates KG-AC-P6's actual target (blocking + matching + union-find correctness) from
    LLM adjudication quality, which has no AC/floor of its own and is untestable here. This is the
    GATING score.
  - **no adjudicator** (adjudicate=None, ambiguous pairs never merge) — reported, not gated: the
    honest "what this looks like without a live LLM" data point.

Runs only under `pytest -m eval`.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from core import Mention, cluster_mentions, normalize_surface

pytestmark = pytest.mark.eval

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


def _load_gold():
    entities = json.loads((DATASET / "gold" / "entities.json").read_text(encoding="utf-8"))
    duplicates = json.loads((DATASET / "gold" / "duplicates.json").read_text(encoding="utf-8"))
    return entities, duplicates["clusters"]


def _mention_key(doc_id: str, surface: str) -> str:
    return f"{doc_id}::{surface}"


def _build_mentions(entities):
    mentions = []
    for e in entities:
        key = _mention_key(e["doc_id"], e["surface"])
        m = Mention(entity_uid=key, entity_type=e["fibo_core_type"], surface_form=e["surface"])
        m.normalized_form = normalize_surface(e["surface"])
        mentions.append(m)
    return mentions


def _gold_cluster_id_map(entities, clusters):
    """Every (doc_id, surface) -> a gold cluster id. Mentions not in any labeled cluster are their
    own singleton (keyed by entity_uid, so they never accidentally collide with a real cluster)."""
    key_to_cluster: dict[str, str] = {}
    for i, cluster in enumerate(clusters):
        for m in cluster:
            key_to_cluster[_mention_key(m["doc_id"], m["surface"])] = f"gold-cluster-{i}"
    mapping = {}
    for e in entities:
        key = _mention_key(e["doc_id"], e["surface"])
        mapping[key] = key_to_cluster.get(key, f"singleton::{key}")
    return mapping


def _pairwise_prf1(predicted_clusters: list[list], gold_id_by_key: dict) -> dict:
    """Pairwise precision/recall/F1 (KG-AC-P6): TP = pair in same predicted AND same gold cluster;
    FP = same predicted, different gold; FN = different predicted, same gold. Distinct-mention pairs
    only (i<j), all mentions considered (not just those in a labeled cluster)."""
    all_keys = sorted(gold_id_by_key)
    pred_cluster_by_key = {}
    for ci, cluster in enumerate(predicted_clusters):
        for m in cluster:
            pred_cluster_by_key[m.entity_uid] = ci

    tp = fp = fn = 0
    for i in range(len(all_keys)):
        for j in range(i + 1, len(all_keys)):
            a, b = all_keys[i], all_keys[j]
            same_pred = pred_cluster_by_key[a] == pred_cluster_by_key[b]
            same_gold = gold_id_by_key[a] == gold_id_by_key[b]
            if same_pred and same_gold:
                tp += 1
            elif same_pred and not same_gold:
                fp += 1
            elif not same_pred and same_gold:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
           "tp": tp, "fp": fp, "fn": fn}


def _run_clustering(entities, adjudicate):
    mentions = _build_mentions(entities)
    clusters = cluster_mentions(mentions, fuzzy_floor=0.80, fuzzy_ceiling=0.95, adjudicate=adjudicate)
    return clusters


def _baseline():
    if BASELINES.exists():
        return json.loads(BASELINES.read_text(encoding="utf-8")).get("kg_canonicalization_cluster")
    return None


@pytest.mark.ac("KG-AC-P6")
def test_cluster_quality_gold_oracle_vs_baseline():
    entities, clusters = _load_gold()
    gold_id_by_key = _gold_cluster_id_map(entities, clusters)

    def gold_oracle_adjudicate(a: Mention, b: Mention) -> bool:
        return gold_id_by_key[a.entity_uid] == gold_id_by_key[b.entity_uid]

    predicted = _run_clustering(entities, adjudicate=gold_oracle_adjudicate)
    scores = _pairwise_prf1(predicted, gold_id_by_key)
    print(f"\nKG CANONICALIZATION CLUSTER SCORES (gold-oracle adjudicator): {json.dumps(scores, indent=2)}")

    base = _baseline()
    if base is None:
        # bring-up mode: KG-AC-P6's own floor only; record printed scores in baselines.json
        assert scores["f1"] >= 0.70, scores
    else:
        assert scores["f1"] >= 0.70, scores
        assert scores["f1"] >= base["f1"] - 0.02, f"regression: {scores} vs baseline {base}"


def test_cluster_quality_no_adjudicator_reported_not_gated():
    """Informational only (no AC marker, no assertion floor) -- the honest no-LLM data point."""
    entities, clusters = _load_gold()
    gold_id_by_key = _gold_cluster_id_map(entities, clusters)
    predicted = _run_clustering(entities, adjudicate=None)
    scores = _pairwise_prf1(predicted, gold_id_by_key)
    print(f"\nKG CANONICALIZATION CLUSTER SCORES (NO adjudicator -- realistic no-LLM behavior): "
          f"{json.dumps(scores, indent=2)}")
    assert 0.0 <= scores["f1"] <= 1.0  # sanity only -- this run is reported, never a regression gate
