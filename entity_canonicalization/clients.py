"""Connection -> Bedrock LLM client for the entity_extraction worker (copied from the classification
worker's decrypt->Bedrock transport). Credentials come ONLY from the connection-profile decrypt path
(``expected_category='llm'``), never from ambient env (KG-AC-15). Each call captures a FinOps
``usage`` entry (``charge_category='llm'``) — llm calls only (KG-AC-17). A hard connect/auth/invoke
failure raises ``LlmHardFailure`` (fail loud, KG-AC-34) — never a silent empty result.

``decrypt`` and ``invoke`` are injectable so the client is unit-testable without a live endpoint.
The strategies consume ``BedrockLlmClient.complete(prompt) -> str``; the worker reads ``.usage``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

_DEFAULT_MODEL = "amazon.nova-pro-v1:0"
_MAX_TOKENS = 2048


class LlmHardFailure(RuntimeError):
    """A hard LLM failure (connect/auth/5xx/invoke error) — fail loud, never a silent empty graph."""


def _config_value(*keys: str) -> str:
    for key in keys:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, val = line.partition("=")
            if name.strip() in keys and val.strip():
                return val.strip()
    return ""


def _decrypt_connection(profile_ref: str, expected_category: str) -> Tuple[str, Dict[str, Any]]:
    """POST the connection profile name to the decrypt service -> (connection_type, config).
    Fails loud on a missing endpoint or non-200 (never a silent stub); credentials never from env."""
    url = _config_value("PROFILE_DECRYPT_URL")
    if not url:
        raise LlmHardFailure(f"PROFILE_DECRYPT_URL is not set — cannot resolve connection '{profile_ref}'.")
    resp = httpx.post(url, json={"profile_name": profile_ref, "expected_category": expected_category}, timeout=30.0)
    if resp.status_code != 200:
        raise LlmHardFailure(
            f"decrypt connection '{profile_ref}' returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    data = resp.json() or {}
    return str(data.get("connection_type") or ""), (data.get("config") or {})


def _bedrock_invoke(cfg: Dict[str, Any], *, model: str, prompt: str) -> Tuple[str, Dict[str, int]]:
    """Invoke a Bedrock model. Amazon Nova uses ``messages-v1``; Anthropic-on-Bedrock uses
    ``bedrock-2023-05-31``. Returns (text, {input/output tokens})."""
    import boto3  # lazy — only when an AWS LLM connection is actually used

    session = boto3.Session(
        aws_access_key_id=cfg.get("access_key_id") or cfg.get("aws_access_key_id"),
        aws_secret_access_key=cfg.get("secret_access_key") or cfg.get("aws_secret_access_key"),
        aws_session_token=cfg.get("session_token") or None,
        region_name=cfg.get("region") or cfg.get("aws_region") or "us-east-1",
    )
    client = session.client("bedrock-runtime")

    if model.startswith("anthropic."):
        body = {
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": _MAX_TOKENS, "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:  # Amazon Nova family (messages-v1)
        body = {
            "schemaVersion": "messages-v1",
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"temperature": 0.0, "maxTokens": _MAX_TOKENS},
        }

    resp = client.invoke_model(modelId=model, body=json.dumps(body),
                               contentType="application/json", accept="application/json")
    raw = resp["body"].read()
    j = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
    usage = j.get("usage") or {}
    if model.startswith("anthropic."):
        text = j["content"][0]["text"]
        toks = {"input_tokens": usage.get("input_tokens") or 0, "output_tokens": usage.get("output_tokens") or 0}
    else:
        text = j["output"]["message"]["content"][0]["text"]
        toks = {"input_tokens": usage.get("inputTokens") or 0, "output_tokens": usage.get("outputTokens") or 0}
    return text, toks


class BedrockLlmClient:
    """LLM client the llm_ner / llm_relation strategies consume via ``complete(prompt) -> str``."""

    def __init__(self, connection_id: str, *, model: Optional[str] = None,
                 decrypt: Optional[Callable] = None, invoke: Optional[Callable] = None):
        self.connection_id = connection_id
        self._model = model
        self._decrypt = decrypt or _decrypt_connection
        self._invoke = invoke or _bedrock_invoke
        self._cfg: Optional[Dict[str, Any]] = None
        self._resolved_model: Optional[str] = None
        self._provider_name: Optional[str] = None
        self.usage: List[Dict[str, Any]] = []

    @property
    def resolved_model(self) -> Optional[str]:
        """The model id the connection resolved to (set on first complete()); None until resolved."""
        return self._resolved_model

    def _resolve(self) -> None:
        if self._cfg is not None:
            return
        if not self.connection_id:
            raise LlmHardFailure("llm engine/relation mode requires a connection_id")
        ctype, cfg = self._decrypt(self.connection_id, expected_category="llm")  # KG-AC-15
        if ctype not in ("aws_bedrock", "aws_embedding"):
            raise LlmHardFailure(f"connection_type '{ctype}' not supported for LLM (only aws_bedrock in v1)")
        self._cfg = cfg
        self._resolved_model = self._model or cfg.get("model") or _DEFAULT_MODEL
        self._provider_name = ctype

    def complete(self, prompt: str) -> str:
        self._resolve()
        try:
            text, toks = self._invoke(self._cfg, model=self._resolved_model, prompt=prompt)
        except LlmHardFailure:
            raise
        except Exception as exc:  # noqa: BLE001 — any invoke error is a hard failure (KG-AC-34)
            raise LlmHardFailure(f"Bedrock invoke failed: {exc}") from exc
        # The frozen worker usage-block contract (specs/finops, cost_service.py's own docstring):
        # charge_category + provider_name + quantity{} so cost_service._build_event can price it —
        # found live 2026-08-05 that this never conformed (flat input_tokens/output_tokens, no
        # provider_name), so every adjudicator LLM call was silently dropped by FinOps capture.
        self.usage.append({
            "charge_category": "llm",
            "provider_name": self._provider_name,
            "model": self._resolved_model,
            "quantity": {
                "input_tokens": toks.get("input_tokens", 0),
                "output_tokens": toks.get("output_tokens", 0),
            },
            "connection_id": self.connection_id,
        })
        return text


def build_llm_client(connection_id: Optional[str], model: Optional[str] = None) -> Optional[BedrockLlmClient]:
    """Return a client when a connection is configured, else None (the caller decides whether an
    absent connection is fatal — llm engine/relation modes treat None as loud failure)."""
    if not connection_id:
        return None
    return BedrockLlmClient(connection_id, model=model)
