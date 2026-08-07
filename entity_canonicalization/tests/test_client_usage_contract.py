"""KG-AC-15/17 (client side, previously untested): BedrockLlmClient.complete()'s usage[] entry must
match the frozen FinOps worker usage-block contract (specs/finops, cost_service.py's own docstring)
-- charge_category + provider_name + quantity{} -- so cost_service._build_event can price it.

Found live 2026-08-05 (via the entity_extraction runtime-preview showing Cost: '-') that this
client's usage entry never conformed: flat input_tokens/output_tokens, no provider_name, so every
adjudicator LLM call was silently dropped by FinOps capture (fail-open masked it). No prior test
exercised the real client here (test_llm_adjudication.py uses a fake). Fixed in clients.py.
"""
from clients import BedrockLlmClient


def test_usage_entry_matches_finops_contract():
    def decrypt(profile, expected_category):
        return "aws_bedrock", {"model": "amazon.nova-pro-v1:0"}

    def invoke(cfg, *, model, prompt):
        return "yes", {"input_tokens": 20, "output_tokens": 1}

    client = BedrockLlmClient("c", decrypt=decrypt, invoke=invoke)
    client.complete("adjudicate these two mentions")

    assert len(client.usage) == 1
    u = client.usage[0]
    assert u["charge_category"] == "llm"
    assert u["provider_name"] == "aws_bedrock"
    assert u["quantity"] == {"input_tokens": 20, "output_tokens": 1}
    assert u["connection_id"] == "c" and u["model"] == "amazon.nova-pro-v1:0"
