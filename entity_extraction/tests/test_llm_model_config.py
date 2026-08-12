"""KG-AC-95: the Bedrock Converse model id is profile-configurable via
``entity_extraction_config.llm_model`` — mirrors KG-AC-87's cap override exactly, on a client
that already accepted an explicit ``model`` override (``build_llm_client(connection_id,
model=None, max_tokens=None)``) with no config surface ever wired to reach it. Found live
2026-08-12: the "AWS LLM Connection" profile's own config has never carried a ``model`` key at
any layer, so every call to date silently ran on ``clients._DEFAULT_MODEL``
(``openai.gpt-oss-120b-1:0``), not a deliberate choice. Model choice belongs on the PROFILE, not
the shared connection (owner decision) — a connection is credentials+region, shared across
profiles that may legitimately want different models."""
import pytest

from clients import BedrockLlmClient, build_llm_client
from strategies.base import ExtractionConfig


def _ok_decrypt(profile, expected_category):
    return "aws_bedrock", {"region": "us-east-1", "model": "amazon.nova-pro-v1:0"}


def _tool_use_response(tool_name, tool_input):
    return {
        "output": {"message": {"content": [{"toolUse": {"name": tool_name, "input": tool_input}}]}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }


# ---- ExtractionConfig ---------------------------------------------------------------------------
@pytest.mark.ac("KG-AC-95")
def test_default_llm_model_is_none():
    assert ExtractionConfig().llm_model is None


@pytest.mark.ac("KG-AC-95")
def test_from_dict_default_llm_model_is_none():
    config = ExtractionConfig.from_dict({"engine": "llm", "ontology_pack": "generic"})
    assert config.llm_model is None


@pytest.mark.ac("KG-AC-95")
def test_from_dict_llm_model_round_trips():
    config = ExtractionConfig.from_dict(
        {"engine": "llm", "ontology_pack": "generic",
         "llm_model": "anthropic.claude-sonnet-4-5-20250929-v1:0"})
    assert config.llm_model == "anthropic.claude-sonnet-4-5-20250929-v1:0"


# ---- BedrockLlmClient / build_llm_client ---------------------------------------------------------
@pytest.mark.ac("KG-AC-95")
def test_build_llm_client_forwards_model_to_the_client():
    client = build_llm_client("c", model="anthropic.claude-sonnet-4-5-20250929-v1:0")
    assert client._model == "anthropic.claude-sonnet-4-5-20250929-v1:0"


@pytest.mark.ac("KG-AC-95")
def test_build_llm_client_unset_model_falls_back_to_the_connection_then_default():
    # None -> BedrockLlmClient._model is None -> _resolved_model falls back to cfg.get("model")
    # (the connection's own config) at call time, then clients._DEFAULT_MODEL -- verified via the
    # DIRECT-construction test below, which observes the model actually invoked.
    client = build_llm_client("c")
    assert client._model is None


@pytest.mark.ac("KG-AC-95")
def test_explicit_model_overrides_the_connections_own_configured_model():
    captured = {}

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None, max_tokens=None):
        captured["model"] = model
        return _tool_use_response("extract_graph", {"entities": [], "relations": []})

    # _ok_decrypt's cfg carries model="amazon.nova-pro-v1:0" -- the explicit override must win.
    client = BedrockLlmClient("c", model="anthropic.claude-sonnet-4-5-20250929-v1:0",
                              decrypt=_ok_decrypt, invoke=invoke)
    client.complete_tool(system_text="s", user_text="u", tool_name="extract_graph",
                         tool_description="d", tool_schema={"type": "object"})
    assert captured["model"] == "anthropic.claude-sonnet-4-5-20250929-v1:0"


@pytest.mark.ac("KG-AC-95")
def test_unset_model_falls_back_to_the_connections_own_configured_model():
    captured = {}

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None, max_tokens=None):
        captured["model"] = model
        return _tool_use_response("extract_graph", {"entities": [], "relations": []})

    client = BedrockLlmClient("c", decrypt=_ok_decrypt, invoke=invoke)  # no model override
    client.complete_tool(system_text="s", user_text="u", tool_name="extract_graph",
                         tool_description="d", tool_schema={"type": "object"})
    assert captured["model"] == "amazon.nova-pro-v1:0"  # from _ok_decrypt's cfg


@pytest.mark.ac("KG-AC-95")
def test_unset_model_and_unset_connection_model_falls_back_to_module_default():
    import clients

    def _no_model_decrypt(profile, expected_category):
        return "aws_bedrock", {"region": "us-east-1"}  # no "model" key -- matches the real
                                                        # "AWS LLM Connection" profile found live

    captured = {}

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None, max_tokens=None):
        captured["model"] = model
        return _tool_use_response("extract_graph", {"entities": [], "relations": []})

    client = BedrockLlmClient("c", decrypt=_no_model_decrypt, invoke=invoke)
    client.complete_tool(system_text="s", user_text="u", tool_name="extract_graph",
                         tool_description="d", tool_schema={"type": "object"})
    assert captured["model"] == clients._DEFAULT_MODEL
