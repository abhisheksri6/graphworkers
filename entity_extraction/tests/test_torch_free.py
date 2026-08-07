"""KG-AC-18: the entity_extraction worker's installed dependency set is torch-free.

ADR-0009: the inc-1 layered ensemble (rules/gazetteer + spaCy + LLM-API) carries no transformer,
so no torch. A fresh CPU venv installs clean with pydantic==2.9.2 held. Any FUTURE local transformer
engine must load via raw `onnxruntime` — never `optimum`/`transformers` at runtime (`optimum
[onnxruntime]` still pulls torch, spike F-ONNX-1). This test guards the installed env, so a stray
torch-pulling dep goes red.
"""
import importlib.metadata as md

import pytest


def _installed_names() -> set[str]:
    names = set()
    for dist in md.distributions():
        name = (dist.metadata.get("Name") if dist.metadata else None)
        if name:
            names.add(name.lower())
    return names


@pytest.mark.ac("KG-AC-18")
def test_no_torch_family_in_dependency_set():
    banned = {"torch", "optimum", "transformers", "torchvision", "torchaudio"}
    present = banned & _installed_names()
    assert not present, f"torch-family deps must not be installed (ADR-0009): {sorted(present)}"


@pytest.mark.ac("KG-AC-18")
def test_pydantic_pin_held():
    assert md.version("pydantic") == "2.9.2"


@pytest.mark.ac("KG-AC-18")
def test_spacy_installed_torch_free():
    # en_core_web_lg is a CNN pipeline (not transformer-based) — spaCy installs without torch.
    assert "spacy" in _installed_names()
