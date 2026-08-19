"""Gemini adapter."""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from mehullm.llm.gemini.schema import needs_parameters, sanitize
from mehullm.llm.types import (
    LLMEvent,
    Msg,
    ProviderError,
    StreamEnd,
    TextDelta,
    ToolCall,
    ToolCallReady,
    ToolDef,
    Usage,
    UsageEvent,
)

DEFAULT_MODEL = "gemini-3.5-flash"


def _quota_details(exc: Exception) -> tuple[str, float | None, int | None]:
    """(window, retry_after_s, limit) from a 429 body. window is "day"|"minute"|""."""
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return "", None, None
    window, retry, limit = "", None, None
    for d in (details.get("error") or {}).get("details") or []:
        kind = str(d.get("@type", ""))
        if kind.endswith("QuotaFailure"):
            for v in d.get("violations") or []:
                qid = str(v.get("quotaId", ""))
                if "PerDay" in qid:
                    window = "day"
                elif "PerMinute" in qid and window != "day":
                    window = "minute"
                with contextlib.suppress(TypeError, ValueError):
                    limit = int(v.get("quotaValue"))
        elif kind.endswith("RetryInfo"):
            raw = str(d.get("retryDelay", "")).rstrip("s")
            with contextlib.suppress(ValueError):
                retry = float(raw)
    return window, retry, limit


def _parts_of(candidate: Any) -> list[Any]:
    content = getattr(candidate, "content", None)
    return (getattr(content, "parts", None) or []) if content else []


def _usage(um: Any) -> Usage:
    return Usage(
        input_tokens=getattr(um, "prompt_token_count", 0) or 0,
        output_tokens=getattr(um, "candidates_token_count", 0) or 0,
        extra={"thoughts": getattr(um, "thoughts_token_count", 0) or 0},
    )


def _classify(exc: Exception) -> ProviderError:
    s = f"{type(exc).__name__}: {exc}".lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)

    if code == 429 or "resource_exhausted" in s:
        # A 429 is BOTH windows. Google says which in quotaId; the string does
        # not, and it always contains the word "quota" either way.
        window, retry, limit = _quota_details(exc)
        kind = "quota" if window == "day" else "rate_limit"
        return ProviderError(kind, str(exc), provider="gemini", retry_after=retry, limit=limit)
    if code in (500, 502, 503, 504) or "unavailable" in s or "overloaded" in s:
        return ProviderError("overloaded", str(exc), provider="gemini")
    if "deadline" in s or "timeout" in s:
        return ProviderError("timeout", str(exc), provider="gemini")
    if code == 404 or "no longer available" in s or "not_found" in s:
        return ProviderError("model_unavailable", str(exc), provider="gemini")
    if code == 400 or "invalid" in s:
        return ProviderError("bad_request", str(exc), provider="gemini")
    if code in (401, 403) or "api key" in s or "permission" in s:
        return ProviderError("auth", str(exc), provider="gemini")
    return ProviderError("unknown", str(exc), provider="gemini")


class GeminiClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from google import genai  # imported lazily: optional dep

        # Per-instance and model-qualified: quota is per-model, and the router
        # keys every lookup and demotion on client.name.
        self.name = f"gemini:{model}"

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self._turn = 0
        self._cid = uuid.uuid4().hex[:6]
        # Raw functionCall Parts keyed by call id. Gemini 400s if a signature is
        # not echoed back, and rebuilding a Part loses it.
        self._parts: dict[str, Any] = {}

    def _to_contents(self, messages: list[Msg]) -> list[Any]:
        from google.genai import types

        out = []
        for m in messages:
            if m.role == "system":
                continue  # carried in system_instruction
            if m.role == "tool":
                out.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=m.tool_name or "tool",
                                response={"result": m.text or ""},
                            )
                        ],
                    )
                )
                continue

            parts = []
            if m.text:
                parts.append(types.Part.from_text(text=m.text))
            unsigned = []
            for tc in m.tool_calls:
                if (raw := self._parts.get(tc.id)) is not None:
                    parts.append(raw)
                else:
                    # Foreign or restored call: an unsigned Part is a guaranteed 400.
                    unsigned.append(f"{tc.name}({json.dumps(tc.args, default=str)})")
            if unsigned:
                parts.append(
                    types.Part.from_text(text="[previously called: " + "; ".join(unsigned) + "]")
                )
            if not parts:
                continue
            out.append(
                types.Content(role="model" if m.role == "assistant" else "user", parts=parts)
            )
        return out

    def _to_tools(self, tools: list[ToolDef]) -> list[Any]:
        from google.genai import types

        if not tools:
            return []
        decls = []
        for t in tools:
            kwargs: dict[str, Any] = {"name": t.name, "description": t.description}
            # Zero-arg tools must OMIT parameters -- an empty object is rejected.
            if needs_parameters(t.parameters):
                kwargs["parameters"] = sanitize(t.parameters)
            decls.append(types.FunctionDeclaration(**kwargs))
        return [types.Tool(function_declarations=decls)]

    async def stream(
        self,
        *,
        system: str,
        messages: list[Msg],
        tools: list[ToolDef],
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMEvent]:
        from google.genai import types

        self._turn += 1
        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
            # We drive the loop; the SDK must not call tools behind our back or
            # the guardrail interceptor is bypassed.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tools=self._to_tools(tools) or None,
        )

        idx = 0
        finish = "stop"
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model, contents=self._to_contents(messages), config=cfg
            )
            async for chunk in stream:
                for cand in getattr(chunk, "candidates", None) or []:
                    for part in _parts_of(cand):
                        text = getattr(part, "text", None)
                        if text:
                            yield TextDelta(text)
                        fc = getattr(part, "function_call", None)
                        if fc:
                            # Synthesised at ingest: canonical history always has an id.
                            call_id = f"call_{self._cid}_{self._turn}_{idx}"
                            # Keep the WHOLE part, signature included, whether or
                            # not one is visible on it. See _parts in __init__.
                            self._parts[call_id] = part
                            yield ToolCallReady(
                                ToolCall(
                                    id=call_id,
                                    name=fc.name,
                                    args=dict(fc.args or {}),
                                )
                            )
                            idx += 1
                    if getattr(cand, "finish_reason", None):
                        finish = str(cand.finish_reason)

                um = getattr(chunk, "usage_metadata", None)
                if um:
                    yield UsageEvent(_usage(um), self.name, self.model)
        except Exception as e:
            raise _classify(e) from e

        yield StreamEnd("tool_use" if idx else finish)
