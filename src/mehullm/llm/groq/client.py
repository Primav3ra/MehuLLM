"""Groq adapter (rate-limit fallback).

Uses the `openai` SDK pointed at Groq's OpenAI-compatible endpoint rather than
the `groq` package: identical wire format, one less dependency, and it makes any
other OpenAI-compatible endpoint (Cerebras, a local vLLM) a drop-in third
fallback for free.

THE BINDING CONSTRAINT IS TPM, NOT RPM. Groq's free tier allows 30 requests/min
but only 6-12K tokens/min, and 40 MCP tool schemas cost 3-5K tokens *per
request*. Tool allowlisting is what makes failover work at all.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

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

BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _classify(exc: Exception) -> ProviderError:
    status = getattr(exc, "status_code", None)
    s = f"{type(exc).__name__}: {exc}".lower()
    retry = None
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            retry = float(resp.headers.get("retry-after", "")) or None
        except (TypeError, ValueError):
            retry = None
    if status == 429 or "rate limit" in s:
        kind = "quota" if "day" in s or "tpd" in s or "rpd" in s else "rate_limit"
        return ProviderError(kind, str(exc), retry_after=retry, provider="groq")
    if status in (500, 502, 503, 504) or "overloaded" in s:
        return ProviderError("overloaded", str(exc), provider="groq")
    if "timeout" in s:
        return ProviderError("timeout", str(exc), provider="groq")
    if status == 400:
        return ProviderError("bad_request", str(exc), provider="groq")
    if status in (401, 403):
        return ProviderError("auth", str(exc), provider="groq")
    return ProviderError("unknown", str(exc), provider="groq")


def _to_openai(messages: list[Msg]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id or "",
                    "content": m.text or "",
                }
            )
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.text or ""}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        else:
            out.append({"role": m.role, "content": m.text or ""})
    return out


def _repair_json(s: str) -> dict[str, Any]:
    """Best-effort recovery of truncated tool arguments.

    Real and not theoretical: hitting the TPM ceiling mid-generation cuts the
    argument string off, leaving invalid JSON. Closing the open braces recovers
    the complete keys, which beats dropping the whole call.
    """
    s = (s or "").strip()
    if not s:
        return {}
    for candidate in (s, s + "}", s + '"}', s + '"}}'):
        try:
            v = json.loads(candidate)
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            continue
    return {}


class GroqClient:
    name = "groq"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from openai import AsyncOpenAI  # lazy: optional dep

        self._client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)
        self.model = model

    async def stream(
        self,
        *,
        system: str,
        messages: list[Msg],
        tools: list[ToolDef],
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMEvent]:
        payload_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    # Groq accepts full JSON Schema -- no sanitiser needed here.
                    "parameters": t.parameters or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]

        msgs = ([{"role": "system", "content": system}] if system else []) + _to_openai(messages)

        # index -> partial call. Groq streams tool calls FRAGMENTED: chunk 1
        # carries id+name with empty arguments, chunks 2..N carry only argument
        # substrings that are not individually valid JSON.
        acc: dict[int, dict[str, Any]] = {}
        finish = "stop"

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=msgs,
                tools=payload_tools or None,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    u = chunk.usage
                    yield UsageEvent(
                        Usage(
                            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                            output_tokens=getattr(u, "completion_tokens", 0) or 0,
                        ),
                        self.name,
                        self.model,
                    )
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)

                if delta is not None and getattr(delta, "content", None):
                    yield TextDelta(delta.content)

                for tc in (getattr(delta, "tool_calls", None) or []) if delta else []:
                    slot = acc.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                    if getattr(tc, "id", None):
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            slot["args"] += fn.arguments

                if getattr(choice, "finish_reason", None):
                    finish = choice.finish_reason
                    if finish == "tool_calls":
                        for i in sorted(acc):
                            s = acc[i]
                            try:
                                args = json.loads(s["args"] or "{}")
                            except json.JSONDecodeError:
                                args = _repair_json(s["args"])
                            yield ToolCallReady(
                                ToolCall(
                                    id=s["id"] or f"call_{i}",
                                    name=s["name"],
                                    args=args if isinstance(args, dict) else {},
                                )
                            )
                        acc.clear()
        except Exception as e:  # noqa: BLE001
            raise _classify(e) from e

        yield StreamEnd(finish)
