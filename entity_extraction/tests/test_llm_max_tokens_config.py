"""KG-AC-87: the Bedrock Converse ``maxTokens`` cap is profile-configurable via
``entity_extraction_config.llm_max_tokens`` — previously a fixed module constant (`clients._MAX_TOKENS
= 4096`). Found live 2026-08-11: investment_fibo v2.1's 42-property vocabulary + KG-AC-63's
per-occurrence entity emission, over a multi-page runtime-preview sample fed as ONE chunk, hit the
cap on both attempts (a genuine KG-AC-P3 hard failure — `stopReason='max_tokens'`, retried once,
still failed). `llm_max_tokens` unset/None preserves today's default exactly."""
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


# ---- ExtractionConfig -------------------------------------------------------------------------
@pytest.mark.ac("KG-AC-87")
def test_default_llm_max_tokens_is_none():
    assert ExtractionConfig().llm_max_tokens is None


@pytest.mark.ac("KG-AC-87")
def test_from_dict_default_llm_max_tokens_is_none():
    config = ExtractionConfig.from_dict({"engine": "llm", "ontology_pack": "generic"})
    assert config.llm_max_tokens is None


@pytest.mark.ac("KG-AC-87")
def test_from_dict_llm_max_tokens_round_trips():
    config = ExtractionConfig.from_dict(
        {"engine": "llm", "ontology_pack": "generic", "llm_max_tokens": 8000})
    assert config.llm_max_tokens == 8000


# ---- BedrockLlmClient / build_llm_client -------------------------------------------------------
@pytest.mark.ac("KG-AC-87")
def test_build_llm_client_forwards_max_tokens_to_the_client():
    client = build_llm_client("c", max_tokens=8000)
    assert client._max_tokens == 8000


@pytest.mark.ac("KG-AC-87")
def test_build_llm_client_unset_max_tokens_falls_back_to_module_default():
    import clients
    client = build_llm_client("c")  # no max_tokens -> None -> falls back to clients._MAX_TOKENS
    assert client._max_tokens == clients._MAX_TOKENS


@pytest.mark.ac("KG-AC-87")
def test_bedrock_llm_client_direct_construction_with_custom_max_tokens():
    captured = {}

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None, max_tokens=None):
        captured["max_tokens"] = max_tokens
        return _tool_use_response("extract_graph", {"entities": [], "relations": []})

    client = BedrockLlmClient("c", max_tokens=6000, decrypt=_ok_decrypt, invoke=invoke)
    client.complete_tool(system_text="s", user_text="u", tool_name="extract_graph",
                         tool_description="d", tool_schema={"type": "object"})
    assert captured["max_tokens"] == 6000


@pytest.mark.ac("KG-AC-87")
def test_truncation_hint_uses_the_configured_cap_not_the_module_default(caplog):
    import logging

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None, max_tokens=None):
        # simulate a truncated response AT the custom cap (2000), well below the module default
        # (4096) -- the truncation-hint logic must compare against the CONFIGURED cap, not 4096,
        # or a legitimately smaller custom cap would never be flagged as truncated.
        return {
            "output": {"message": {"content": [{"text": "(no tool call)"}]}},
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 100, "outputTokens": 2000},
        }

    client = BedrockLlmClient("c", max_tokens=2000, decrypt=_ok_decrypt, invoke=invoke)
    with caplog.at_level(logging.WARNING, logger="clients"):
        with pytest.raises(Exception):
            client.complete_tool(system_text="s", user_text="u", tool_name="extract_graph",
                                 tool_description="d", tool_schema={"type": "object"})
    messages = [r.getMessage() for r in caplog.records]
    assert any("truncat" in m.lower() and "cap=2000" in m for m in messages)


# ---- capability_schema --------------------------------------------------------------------------
@pytest.mark.ac("KG-AC-87")
def test_capability_schema_declares_llm_max_tokens():
    from capability_schema import CAPABILITY_SCHEMA
    field = CAPABILITY_SCHEMA["input_fields"]["entity_extraction_config.llm_max_tokens"]
    assert field["type"] == "integer"
    assert field["required"] == "optional"
