"""KG-AC-15: LLM credentials come ONLY via the connection-profile decrypt path
(expected_category='llm'), never ambient env; and the spaCy model loads offline by-copy
(SPACY_MODEL_PATH), never spacy.download. KG-AC-17 (partial): each llm call captures a usage[]
entry (charge_category='llm'); non-LLM engines emit none. The real model load is C12."""
import logging

import pytest
from botocore.exceptions import ClientError

from clients import BedrockLlmClient, LlmHardFailure, LlmOutputError
from strategies import Chunk, ExtractionConfig, SpacyNerStrategy


def _converse_response(text, *, input_tokens, output_tokens, stop_reason="end_turn"):
    """Evolve v6: the injected `invoke` callable now returns a raw Converse API response dict."""
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
    }


def _tool_use_response(tool_name, tool_input, *, input_tokens=10, output_tokens=5):
    return {
        "output": {"message": {"content": [{"toolUse": {"name": tool_name, "input": tool_input}}]}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
    }


def _model_error_exception():
    # the exact shape boto3 raises for bedrock-runtime Converse when the model itself produces
    # invalid tool-use output (a real 2026-08-05 finding — a HARD exception, not a graceful
    # stopReason=malformed_tool_use response).
    return ClientError(
        {"Error": {"Code": "ModelErrorException", "Message": "Model produced invalid sequence as part of ToolUse."}},
        "Converse",
    )


def test_decrypt_uses_llm_category():
    captured = {}

    def decrypt(profile, expected_category):
        captured["profile"] = profile
        captured["category"] = expected_category
        return "aws_bedrock", {"region": "us-east-1", "model": "amazon.nova-pro-v1:0"}

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None):
        return _converse_response("ok", input_tokens=10, output_tokens=5)

    client = BedrockLlmClient("aws-llm-conn", decrypt=decrypt, invoke=invoke)
    client.complete("hi")
    assert captured["profile"] == "aws-llm-conn"
    assert captured["category"] == "llm"  # KG-AC-15: role-gated decrypt, not ambient creds


@pytest.mark.ac("KG-AC-15")
def test_llm_credentials_only_via_decrypt():
    # the client never reads AWS creds from env — the config comes from decrypt's cfg dict.
    seen_cfg = {}

    def decrypt(profile, expected_category):
        return "aws_bedrock", {"aws_access_key_id": "AKIA_FROM_DECRYPT", "region": "eu-west-1", "model": "amazon.nova-pro-v1:0"}

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None):
        seen_cfg.update(cfg)
        return _converse_response("ok", input_tokens=1, output_tokens=1)

    BedrockLlmClient("c", decrypt=decrypt, invoke=invoke).complete("hi")
    assert seen_cfg["aws_access_key_id"] == "AKIA_FROM_DECRYPT"  # from decrypt, not os.environ


@pytest.mark.ac("KG-AC-17")
def test_usage_captured_llm_only():
    def decrypt(profile, expected_category):
        return "aws_bedrock", {"model": "amazon.nova-pro-v1:0"}

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None):
        return _converse_response("ok", input_tokens=120, output_tokens=40)

    client = BedrockLlmClient("c", decrypt=decrypt, invoke=invoke)
    client.complete("p1")
    client.complete("p2")
    assert len(client.usage) == 2
    u = client.usage[0]
    assert u["charge_category"] == "llm"
    # FinOps contract (cost_service.py's frozen usage-block seam): provider_name required,
    # token counts nested under quantity{} — found live 2026-08-05 that this client never
    # conformed, so every KG LLM call was silently unpriced (cost_service._build_event drops
    # any item missing provider_name or a quantity dict).
    assert u["provider_name"] == "aws_bedrock"
    assert u["quantity"] == {"input_tokens": 120, "output_tokens": 40}
    assert u["connection_id"] == "c" and u["model"] == "amazon.nova-pro-v1:0"


@pytest.mark.ac("KG-AC-15")
def test_spacy_requires_by_copy_model_path_no_download(monkeypatch):
    # No SPACY_MODEL_PATH and no injected nlp -> loud error; the code path uses spacy.load(path),
    # never spacy.download (offline, by-copy). The real offline load is verified in C12.
    monkeypatch.delenv("SPACY_MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError):
        SpacyNerStrategy(model_path=None).extract([Chunk("c1", "text")], ExtractionConfig(engine="spacy"), _pack())


@pytest.mark.ac("KG-AC-42")
def test_boot_preflight_fails_loud_when_model_absent():
    # SPACY_MODEL_PATH set but the by-copy model is missing -> the boot preflight raises (fail loud),
    # so the worker aborts startup instead of accepting tasks that each fail per-folder (RCA 2026-08-03).
    from types import SimpleNamespace

    from entity_extraction_worker import preflight_spacy_model

    cfg = SimpleNamespace(spacy_model_path="./models/__does_not_exist__")
    with pytest.raises(RuntimeError):
        preflight_spacy_model(cfg)


@pytest.mark.ac("KG-AC-42")
def test_boot_preflight_noop_when_unset_llm_only():
    # No SPACY_MODEL_PATH -> llm-only worker -> preflight is a no-op (no spaCy asset required); never loads.
    from types import SimpleNamespace

    from entity_extraction_worker import preflight_spacy_model

    assert preflight_spacy_model(SimpleNamespace(spacy_model_path="")) is None


def _pack():
    from ontologies import load_pack
    return load_pack("generic")


# ---- KG-AC-P3 (mechanism gap found live 2026-08-05): Bedrock can reject malformed tool-use as a
# HARD exception (ModelErrorException), not only as a graceful stopReason=malformed_tool_use
# response -- complete_tool's one-retry safety net must cover both delivery mechanisms.
@pytest.mark.ac("KG-AC-P3")
def test_complete_tool_retries_once_on_model_error_exception():
    def decrypt(profile, expected_category):
        return "aws_bedrock", {"model": "amazon.nova-pro-v1:0"}

    calls = {"n": 0}

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _model_error_exception()
        return _tool_use_response("extract_graph", {"entities": [], "relations": []})

    client = BedrockLlmClient("c", decrypt=decrypt, invoke=invoke)
    result = client.complete_tool(system_text="s", user_text="u", tool_name="extract_graph",
                                  tool_description="d", tool_schema={"type": "object"})
    assert result == {"entities": [], "relations": []}
    assert calls["n"] == 2  # exactly one retry


@pytest.mark.ac("KG-AC-P3")
def test_complete_tool_raises_llm_output_error_when_model_error_persists():
    def decrypt(profile, expected_category):
        return "aws_bedrock", {"model": "amazon.nova-pro-v1:0"}

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None):
        raise _model_error_exception()

    client = BedrockLlmClient("c", decrypt=decrypt, invoke=invoke)
    with pytest.raises(LlmOutputError):
        client.complete_tool(system_text="s", user_text="u", tool_name="extract_graph",
                             tool_description="d", tool_schema={"type": "object"})


@pytest.mark.ac("KG-AC-34")
def test_complete_tool_does_not_retry_on_unrelated_hard_failure():
    # a genuine connect/auth/throttling failure is NOT a malformed-tool-use case -- fail loud
    # immediately, no retry (retrying would double the cost/latency of an unrecoverable error).
    def decrypt(profile, expected_category):
        return "aws_bedrock", {"model": "amazon.nova-pro-v1:0"}

    calls = {"n": 0}

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None):
        calls["n"] += 1
        raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "rate limited"}}, "Converse")

    client = BedrockLlmClient("c", decrypt=decrypt, invoke=invoke)
    with pytest.raises(LlmHardFailure):
        client.complete_tool(system_text="s", user_text="u", tool_name="extract_graph",
                             tool_description="d", tool_schema={"type": "object"})
    assert calls["n"] == 1  # no retry for a non-tool-use hard failure


# ---- 2026-08-06 production RCA fixes: token budget + diagnostic logging -------------------------
def test_max_tokens_raised_for_evidence_bearing_relations():
    # KG-AC-46 added a mandatory full-sentence `evidence` field per relation; the original 2048 cap
    # (sized before that addition) risks truncating a dense chunk's tool-call JSON, which manifests
    # as a DETERMINISTIC (temp=0) malformed-tool-use failure on both attempts -- a real production
    # RCA (runtime_entity_extraction_task, 2026-08-06). Bumped to give real headroom.
    import clients
    assert clients._MAX_TOKENS == 4096


@pytest.mark.ac("KG-AC-P3")
def test_complete_tool_logs_diagnostics_on_model_error_exception(caplog):
    def decrypt(profile, expected_category):
        return "aws_bedrock", {"model": "amazon.nova-pro-v1:0"}

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None):
        raise _model_error_exception()

    client = BedrockLlmClient("c", decrypt=decrypt, invoke=invoke)
    with caplog.at_level(logging.WARNING, logger="clients"):
        with pytest.raises(LlmOutputError):
            client.complete_tool(system_text="s", user_text="u", tool_name="extract_graph",
                                 tool_description="d", tool_schema={"type": "object"})
    messages = [r.getMessage() for r in caplog.records]
    # no document content is ever logged (KG-AC-17) -- only the tool name, attempt number, error.
    assert any("extract_graph" in m and "attempt 1" in m and "retry" in m.lower() for m in messages)
    assert any("extract_graph" in m and "both attempts" in m.lower() for m in messages)


@pytest.mark.ac("KG-AC-P3")
def test_complete_tool_logs_truncation_hint_when_output_tokens_hit_cap(caplog):
    def decrypt(profile, expected_category):
        return "aws_bedrock", {"model": "amazon.nova-pro-v1:0"}

    def invoke(cfg, *, model, system_blocks, messages, tool_config=None):
        # graceful stopReason path (no toolUse block) with output_tokens AT the configured cap --
        # the strongest available signal that the response was truncated, not just malformed.
        return {
            "output": {"message": {"content": [{"text": "(no tool call)"}]}},
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 500, "outputTokens": 4096},
        }

    client = BedrockLlmClient("c", decrypt=decrypt, invoke=invoke)
    with caplog.at_level(logging.WARNING, logger="clients"):
        with pytest.raises(LlmOutputError):
            client.complete_tool(system_text="s", user_text="u", tool_name="extract_graph",
                                 tool_description="d", tool_schema={"type": "object"})
    messages = [r.getMessage() for r in caplog.records]
    assert any("truncat" in m.lower() for m in messages)
    assert any("max_tokens" in m for m in messages)  # the stopReason itself, for diagnosis
