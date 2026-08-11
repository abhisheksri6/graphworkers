"""Connection -> Bedrock LLM client for the entity_extraction worker (evolve v6 — moved from raw
``invoke_model`` + hand-built per-model-family bodies to the **Converse API**, which is
model-family-agnostic: the same request/response shape serves both Claude and Nova, eliminating the
old ``anthropic.*`` vs Nova ``messages-v1`` branch entirely.

Credentials come ONLY from the connection-profile decrypt path (``expected_category='llm'``), never
from ambient env (KG-AC-15). Each call captures a FinOps ``usage`` entry (``charge_category='llm'``)
— llm calls only (KG-AC-17). A hard connect/auth/invoke failure raises ``LlmHardFailure`` (fail
loud, KG-AC-34) — never a silent empty result.

Two invocation modes:
  - ``complete(prompt) -> str`` — plain text completion (no tools). Unchanged signature from v5;
    serves ``coref.py``'s rewrite calls (this repo) and ``entity_canonicalization``'s adjudicator
    (a SEPARATE copy of this file, NOT touched by this evolve).
  - ``complete_tool(...) -> dict`` (NEW) — forces the model to call a single named tool whose
    arguments are typed by a JSON schema (KG-AC-43's mechanism amendment): the API validates the
    shape, not a downstream `json.loads`. The static system vocabulary is marked cacheable
    (`cachePoint`, ONLY for models that support it — see ``supports_prompt_caching``); a
    `stopReason=malformed_tool_use` response is retried exactly once (KG-AC-P3);
    a retry that still fails raises loud (`LlmOutputError`) and propagates — never a silent partial
    graph.

``decrypt`` and ``invoke`` are injectable so the client is unit-testable without a live endpoint.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "openai.gpt-oss-120b-1:0"

# Bedrock prompt caching is MODEL-SPECIFIC. Sending a `cachePoint` block to a model that does not
# support it fails the call outright with a generic `AccessDeniedException` ("You invoked an
# unsupported model or your request did not allow prompt caching") — which reads like a credentials
# problem, not a capability one (found live 2026-08-08 when switching to openai.gpt-oss-120b).
# Caching is a COST optimisation, never a correctness requirement, so an unrecognised model must
# fail SAFE (no cache block): the cost of skipping it is spend, the cost of sending it wrongly is
# 100% of calls failing. Prefixes are matched after stripping any cross-region inference prefix
# (e.g. "us."/"eu."/"apac."). Extend this tuple when a family is CONFIRMED to support caching.
_PROMPT_CACHE_MODEL_PREFIXES = ("anthropic.", "amazon.nova")
# 2026-08-06 production RCA: the original 2048 was sized before KG-AC-46 (evolve v5) added a
# mandatory full-sentence `evidence` field to every relation -- a dense chunk's tool-call JSON can
# exceed that, producing a deterministic (temperature=0) malformed-tool-use failure on BOTH the
# original attempt and the identical-parameter retry. Raised for real headroom.
# KG-AC-87 (evolve v13): this is now only the DEFAULT -- a profile whose document/pack combination
# needs more room (a large `datatype_properties` vocabulary, entity_types count, or a document fed
# as one large chunk, e.g. runtime preview) can raise it via `entity_extraction_config.llm_max_tokens`
# without a code change. Found live 2026-08-11: investment_fibo v2.1's 42-property vocabulary +
# KG-AC-63's per-occurrence entity emission, over a multi-page runtime-preview sample fed as ONE
# chunk, hit this cap on BOTH attempts (a genuine, reproducible KG-AC-P3 hard failure, not a bug).
_MAX_TOKENS = 4096


class LlmHardFailure(RuntimeError):
    """A hard LLM failure (connect/auth/5xx/invoke error) — fail loud, never a silent empty graph."""


class LlmOutputError(RuntimeError):
    """A tool call failed schema validation even after one retry (KG-AC-P3) — fail loud, propagates
    and fails the whole folder task (owner decision 2026-08-05, NOT skip-and-count)."""


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


def _bedrock_converse(cfg: Dict[str, Any], *, model: str, system_blocks: Optional[List[Dict[str, Any]]],
                      messages: List[Dict[str, Any]], tool_config: Optional[Dict[str, Any]] = None,
                      max_tokens: int = _MAX_TOKENS) -> Dict[str, Any]:
    """Invoke Bedrock's Converse API — model-family-agnostic (Claude + Nova use the SAME request
    shape here, unlike the old raw invoke_model bodies). Returns the raw Converse response dict.
    ``max_tokens`` (KG-AC-87): defaults to the module cap, overridable per-call by the caller
    (``BedrockLlmClient`` threads its own resolved value through)."""
    import boto3  # lazy — only when an AWS LLM connection is actually used

    session = boto3.Session(
        aws_access_key_id=cfg.get("access_key_id") or cfg.get("aws_access_key_id"),
        aws_secret_access_key=cfg.get("secret_access_key") or cfg.get("aws_secret_access_key"),
        aws_session_token=cfg.get("session_token") or None,
        region_name=cfg.get("region") or cfg.get("aws_region") or "us-east-1",
    )
    client = session.client("bedrock-runtime")

    kwargs: Dict[str, Any] = {
        "modelId": model,
        "messages": messages,
        "inferenceConfig": {"temperature": 0.0, "maxTokens": max_tokens},
    }
    if system_blocks:
        kwargs["system"] = system_blocks
    if tool_config:
        kwargs["toolConfig"] = tool_config
    return client.converse(**kwargs)


def supports_prompt_caching(model: Optional[str]) -> bool:
    """True iff ``model`` is a family CONFIRMED to accept Bedrock `cachePoint` blocks. Unknown or
    empty -> False (fail safe, see _PROMPT_CACHE_MODEL_PREFIXES)."""
    if not model:
        return False
    bare = model
    for region_prefix in ("us.", "eu.", "apac."):  # cross-region inference profile ids
        if bare.startswith(region_prefix):
            bare = bare[len(region_prefix):]
            break
    return bare.startswith(_PROMPT_CACHE_MODEL_PREFIXES)


def build_system_blocks(system_text: str, model: Optional[str]) -> List[Dict[str, Any]]:
    """Converse `system` blocks: the static vocabulary, plus a cachePoint ONLY when the resolved
    model supports prompt caching (KG-AC-43's caching note is an optimisation, not a contract)."""
    blocks: List[Dict[str, Any]] = [{"text": system_text}]
    if supports_prompt_caching(model):
        blocks.append({"cachePoint": {"type": "default"}})
    return blocks


def _extract_text(resp: Dict[str, Any]) -> str:
    content = resp.get("output", {}).get("message", {}).get("content", []) or []
    for block in content:
        if "text" in block:
            return block["text"]
    return ""


def _extract_tool_input(resp: Dict[str, Any], tool_name: str) -> Optional[Dict[str, Any]]:
    """None means retry-worthy (malformed_tool_use / tool not invoked); a dict means success."""
    if resp.get("stopReason") != "tool_use":
        return None
    content = resp.get("output", {}).get("message", {}).get("content", []) or []
    for block in content:
        tool_use = block.get("toolUse")
        if tool_use and tool_use.get("name") == tool_name:
            return tool_use.get("input") or {}
    return None


def _is_model_tool_use_error(exc: BaseException) -> bool:
    """True when a hard failure IS Bedrock's ModelErrorException for invalid tool-use output — the
    exception-delivered sibling of a graceful stopReason=malformed_tool_use response (found live
    2026-08-05: Bedrock does not always return the latter gracefully). KG-AC-P3's one-retry
    treatment covers both delivery mechanisms; any other hard failure (auth/network/throttling)
    still fails loud immediately (KG-AC-34), no retry."""
    from botocore.exceptions import ClientError  # lazy — only when an AWS LLM connection is used

    cause = exc.__cause__
    if not isinstance(cause, ClientError):
        return False
    return cause.response.get("Error", {}).get("Code") == "ModelErrorException"


def _usage_entry(resp: Dict[str, Any], *, model: str, connection_id: str, provider_name: str) -> Dict[str, Any]:
    """The frozen worker usage-block contract (specs/finops, cost_service.py's own docstring):
    charge_category + provider_name + quantity{} so cost_service._build_event can price it — found
    live 2026-08-05 that this client never conformed (flat input_tokens/output_tokens, no
    provider_name), so every KG LLM call was silently dropped by FinOps capture (fail-open masked
    it). Matches workers/classification/strategies/llm.py's reference shape."""
    usage = resp.get("usage") or {}
    entry = {
        "charge_category": "llm",
        "provider_name": provider_name,
        "model": model,
        "quantity": {
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        },
        "connection_id": connection_id,
    }
    # cache stats, when present, are a useful FinOps signal (prompt caching, evolve v6) -- kept OUTSIDE
    # quantity (informational only): compute_billed prices every nonzero quantity component and a
    # rate card without a cache-token rate would make the WHOLE call unpriced, not just uncached.
    if "cacheReadInputTokens" in usage or "cacheWriteInputTokens" in usage:
        entry["cache_read_tokens"] = usage.get("cacheReadInputTokens", 0)
        entry["cache_write_tokens"] = usage.get("cacheWriteInputTokens", 0)
    return entry


class BedrockLlmClient:
    """LLM client the extraction/coreference strategies consume."""

    def __init__(self, connection_id: str, *, model: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 decrypt: Optional[Callable] = None, invoke: Optional[Callable] = None):
        self.connection_id = connection_id
        self._model = model
        # KG-AC-87: None (unset/not configured on the profile) falls back to the module default,
        # preserving today's behaviour exactly for every existing profile.
        self._max_tokens = max_tokens if max_tokens is not None else _MAX_TOKENS
        self._decrypt = decrypt or _decrypt_connection
        self._invoke = invoke or _bedrock_converse
        self._cfg: Optional[Dict[str, Any]] = None
        self._resolved_model: Optional[str] = None
        self._provider_name: Optional[str] = None
        self.usage: List[Dict[str, Any]] = []

    @property
    def resolved_model(self) -> Optional[str]:
        """The model id the connection resolved to (set on first call); None until resolved."""
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

    def _call(self, *, system_blocks, messages, tool_config=None) -> Dict[str, Any]:
        try:
            return self._invoke(self._cfg, model=self._resolved_model, max_tokens=self._max_tokens,
                                system_blocks=system_blocks, messages=messages, tool_config=tool_config)
        except LlmHardFailure:
            raise
        except Exception as exc:  # noqa: BLE001 — any invoke error is a hard failure (KG-AC-34)
            raise LlmHardFailure(f"Bedrock invoke failed: {exc}") from exc

    def complete(self, prompt: str) -> str:
        """Plain text completion — no tools, no system split. Unchanged signature from v5."""
        self._resolve()
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        resp = self._call(system_blocks=None, messages=messages)
        self.usage.append(_usage_entry(resp, model=self._resolved_model, connection_id=self.connection_id,
                                       provider_name=self._provider_name))
        return _extract_text(resp)

    def complete_tool(self, *, system_text: str, user_text: str, tool_name: str,
                      tool_description: str, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
        """KG-AC-43/P3: forced tool-use. The static ``system_text`` (pack vocabulary) is marked
        cacheable; ``user_text`` (per-chunk content) never is. ONE retry on a schema-invalid tool
        call; a retry that still fails raises ``LlmOutputError`` (propagates, fails the folder)."""
        self._resolve()
        system_blocks = build_system_blocks(system_text, self._resolved_model)
        messages = [{"role": "user", "content": [{"text": user_text}]}]
        tool_config = {
            "tools": [{"toolSpec": {"name": tool_name, "description": tool_description,
                                    "inputSchema": {"json": tool_schema}}}],
            "toolChoice": {"tool": {"name": tool_name}},
        }

        for attempt in (1, 2):
            try:
                resp = self._call(system_blocks=system_blocks, messages=messages, tool_config=tool_config)
            except LlmHardFailure as exc:
                if not _is_model_tool_use_error(exc):
                    raise  # a genuinely hard failure (auth/network/throttling) — no retry
                if attempt == 1:
                    # Diagnostic only — never logs document content (KG-AC-17): tool name + attempt.
                    logger.warning(
                        "tool call '%s' attempt 1 failed with a model tool-use error (%s); retrying once",
                        tool_name, exc,
                    )
                    continue  # KG-AC-P3: exactly one retry — malformed tool-use, delivered as an exception
                logger.exception(
                    "tool call '%s' failed schema validation on both attempts (exception-delivered "
                    "ModelErrorException); no retry remains", tool_name,
                )
                raise LlmOutputError(
                    f"tool call '{tool_name}' failed schema validation after one retry "
                    "(model produced invalid tool-use output both attempts)"
                ) from exc
            self.usage.append(_usage_entry(resp, model=self._resolved_model, connection_id=self.connection_id,
                                           provider_name=self._provider_name))
            result = _extract_tool_input(resp, tool_name)
            if result is not None:
                return result
            stop_reason = resp.get("stopReason")
            output_tokens = (resp.get("usage") or {}).get("outputTokens")
            truncated_hint = (
                " — output_tokens reached the maxTokens cap, response likely TRUNCATED"
                if isinstance(output_tokens, int) and output_tokens >= self._max_tokens else ""
            )
            if attempt == 1:
                logger.warning(
                    "tool call '%s' attempt 1 returned stopReason=%r (no valid tool_use), "
                    "output_tokens=%s (cap=%s)%s; retrying once",
                    tool_name, stop_reason, output_tokens, self._max_tokens, truncated_hint,
                )
                continue  # KG-AC-P3: exactly one retry on malformed_tool_use / tool not invoked
            logger.error(
                "tool call '%s' failed on both attempts, stopReason=%r, output_tokens=%s (cap=%s)%s",
                tool_name, stop_reason, output_tokens, self._max_tokens, truncated_hint,
            )
        raise LlmOutputError(
            f"tool call '{tool_name}' failed schema validation after one retry "
            f"(stopReason={resp.get('stopReason')!r})"
        )


def build_llm_client(connection_id: Optional[str], model: Optional[str] = None,
                     max_tokens: Optional[int] = None) -> Optional[BedrockLlmClient]:
    """Return a client when a connection is configured, else None (the caller decides whether an
    absent connection is fatal — llm engine/relation modes treat None as loud failure). ``max_tokens``
    (KG-AC-87): None uses the client's own default (today's behaviour, unchanged)."""
    if not connection_id:
        return None
    return BedrockLlmClient(connection_id, model=model, max_tokens=max_tokens)
