"""Prompt-cache capability gate (bug found live 2026-08-08).

`complete_tool` used to attach a `cachePoint` system block for EVERY model. Bedrock's prompt
caching is model-specific: Claude and Nova support it, `openai.gpt-oss-120b-1:0` does not — and
Bedrock rejects the unsupported combination with a **generic `AccessDeniedException`** ("You
invoked an unsupported model or your request did not allow prompt caching"), which reads like a
credentials problem, not a capability one. Effect: switching the profile to any non-caching model
failed 100% of runs with a misleading error.

The cache block is a cost optimisation, never a correctness requirement, so the safe default for
an unrecognised model is to omit it.
"""
import pytest

from clients import supports_prompt_caching


@pytest.mark.ac("KG-AC-66")
@pytest.mark.parametrize("model", [
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
])
def test_caching_families_are_recognised(model):
    assert supports_prompt_caching(model) is True


@pytest.mark.ac("KG-AC-66")
@pytest.mark.parametrize("model", [
    "openai.gpt-oss-120b-1:0",   # the model that exposed the bug
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-large-2407-v1:0",
    "",
    None,
])
def test_unknown_or_non_caching_models_omit_the_cache_block(model):
    # fail SAFE: an unrecognised model must not get a cachePoint, because the failure mode is a
    # hard 100%-of-calls AccessDeniedException, while the cost of skipping the cache is only spend.
    assert supports_prompt_caching(model) is False


@pytest.mark.ac("KG-AC-66")
def test_system_blocks_built_per_capability():
    from clients import build_system_blocks

    cached = build_system_blocks("VOCAB", "amazon.nova-pro-v1:0")
    assert cached == [{"text": "VOCAB"}, {"cachePoint": {"type": "default"}}]

    plain = build_system_blocks("VOCAB", "openai.gpt-oss-120b-1:0")
    assert plain == [{"text": "VOCAB"}]        # no cachePoint -> the call is accepted
