"""Regression guard — production failure 2026-08-08 (runtime preview, profile `customfibonew`):
`ModuleNotFoundError: No module named 'candidate_pairs'` in the deployed Celery worker, while every
in-process test passed.

**Root cause this locks out.** Celery's `celery.utils.imports.cwd_in_path()` inserts the current
working directory into `sys.path` only for the duration of the `-A <app>` module import, then
REMOVES it again. So a module reachable at app-import time is NOT reachable later, at task runtime.
Any *absolute* import of a top-level worker module (`candidate_pairs`, `core`, `clients`, …) that
executes at task time — either inside a function body, or at the module level of a module that is
itself imported lazily — therefore raises ModuleNotFoundError in the deployed worker.

It passed every test because pytest keeps cwd on `sys.path` permanently (pyproject
`pythonpath = ["."]`), and passed a hand-run `python -c` repro for the same reason (`-c` puts cwd at
`sys.path[0]`). Only a run that *removes* cwd after the app import reproduces the worker.

*Relative* imports inside an already-imported package (`from .llm_graph import …`) are unaffected —
the package is in `sys.modules` with a known `__path__` — which is why only some lazy imports broke.

The rule this enforces: **every absolute import of a top-level worker module must execute at
app-import time**, not lazily at task time.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Reproduces the deployed worker exactly: import the app WITH cwd on sys.path (Celery's
# cwd_in_path), then REMOVE cwd (Celery's cleanup) and only then run the task body.
_SCRIPT = r'''
import os, sys
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

# --- Phase 1: the `-A entity_extraction_worker` app import, inside cwd_in_path() ---
from ontologies import load_pack
from strategies import Chunk, ExtractionConfig, run_pipeline

# --- Phase 2: Celery removes cwd again; the task body runs WITHOUT it ---
sys.path = [p for p in sys.path if p not in ("", cwd)]


class _FakeSentNlp:
    """One sentence spanning the whole chunk — avoids needing the 620MB by-copy spaCy model."""
    class _Doc:
        def __init__(self, text):
            self.sents = [type("S", (), {"start_char": 0, "end_char": len(text)})()]

    def __call__(self, text):
        return self._Doc(text)


class _FakeClient:
    resolved_model = "fake"
    usage = []

    def complete_tool(self, *, tool_name, **_kw):
        if tool_name == "extract_graph":
            return {"entities": [{"type": "Organization", "surface": "Acme Corp", "confidence": 0.9},
                                 {"type": "Person", "surface": "Jane Roe", "confidence": 0.8}],
                    "relations": []}
        return {"labels": [{"pair_index": 0, "relation_type": "employs",
                            "evidence": "Acme Corp employs Jane Roe."}]}


cfg = ExtractionConfig(engine="llm", ontology_pack="fibo_core", relation_strategy="classify")
_ent_rows, edge_rows, _summary, _usage, _blocked = run_pipeline(
    [Chunk("c1", "Acme Corp employs Jane Roe.")], cfg, load_pack("fibo_core"),
    folder_id="f1", llm_client=_FakeClient(), spacy_nlp=_FakeSentNlp(),
)
assert len(edge_rows) == 1, edge_rows
print("RUNTIME_IMPORTS_OK")
'''


@pytest.mark.ac("KG-AC-61")
def test_classify_mode_runs_with_cwd_absent_from_syspath():
    """The deployed-worker condition: cwd is NOT on sys.path when the task body runs."""
    result = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=str(REPO),
                            capture_output=True, text=True)
    assert "RUNTIME_IMPORTS_OK" in result.stdout, (
        "classify-mode run_pipeline failed with cwd off sys.path — a top-level worker module is "
        "being imported lazily at task time instead of at app-import time.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
