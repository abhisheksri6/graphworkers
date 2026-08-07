"""KG-AC-34: an LLM hard failure (connect/auth/5xx/invoke error) fails LOUD — never silently
converted to an empty or partial graph. Mirrors classification's LLM-hard-failure behavior."""
import pytest

from clients import BedrockLlmClient, LlmHardFailure, build_llm_client


def _ok_decrypt(profile, expected_category):
    return "aws_bedrock", {"region": "us-east-1", "model": "amazon.nova-pro-v1:0"}


@pytest.mark.ac("KG-AC-34")
def test_invoke_error_raises_hard_failure():
    def boom(cfg, *, model, prompt):
        raise ConnectionError("bedrock 5xx")
    client = BedrockLlmClient("conn-1", decrypt=_ok_decrypt, invoke=boom)
    with pytest.raises(LlmHardFailure):
        client.complete("hi")
    # nothing was recorded as usage on a hard failure
    assert client.usage == []


@pytest.mark.ac("KG-AC-34")
def test_unsupported_connection_type_raises():
    def bad_decrypt(profile, expected_category):
        return "postgres", {}
    client = BedrockLlmClient("conn-1", decrypt=bad_decrypt, invoke=lambda *a, **k: ("x", {}))
    with pytest.raises(LlmHardFailure):
        client.complete("hi")


@pytest.mark.ac("KG-AC-34")
def test_missing_connection_id_is_loud():
    assert build_llm_client(None) is None  # caller (llm engine/relation) treats None as loud failure
    with pytest.raises(LlmHardFailure):
        BedrockLlmClient("").complete("hi")
