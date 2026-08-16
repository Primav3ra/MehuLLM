"""Gemini adapter (primary brain).

Uses `generate_content_stream`, NOT the Interactions API. Interactions is
Google's recommended agentic surface, but every one of its selling points is
hostile to this design:

  * server-side state (`previous_interaction_id`) cannot transfer to Groq on
    failover -- and failover is the router's entire reason to exist;
  * managed tool orchestration would bypass the guardrail interceptor, which
    must gate every single tool call;
  * `store=true` is the default and persists conversations on Google's servers,
    against the project's privacy posture.

Kept under ~250 lines behind the LLMClient protocol so it stays swappable if
that calculus changes.
"""

from __future__ import annotations

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


def _classify(exc: Exception) -> ProviderError:
    s = f"{type(exc).__name__}: {exc}".lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429 or "429" in s or "resource_exhausted" in s or "rate limit" in s:
        return ProviderError("rate_limit", str(exc), provider="gemini")
    if "quota" in s or "exceeded your current quota" in s:
        return ProviderError("quota", str(exc), provider="gemini")
    if code in (500, 502, 503, 504) or "unavailable" in s or "overloaded" in s:
        return ProviderError("overloaded", str(exc), provider="gemini")
    if "deadline" in s or "timeout" in s:
        return ProviderError("timeout", str(exc), provider="gemini")
    if code == 400 or "invalid" in s:
        # Never fails over -- this is a sanitiser bug and must surface.
        return ProviderError("bad_request", str(exc), provider="gemini")
    if code in (401, 403) or "api key" in s or "permission" in s:
        return ProviderError("auth", str(exc), provider="gemini")
    return ProviderError("unknown", str(exc), provider="gemini")


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from google import genai  # imported lazily: optional dep

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self._turn = 0


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
            for tc in m.tool_calls:
                parts.append(types.Part.from_function_call(name=tc.name, args=tc.args))
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
                    content = getattr(cand, "content", None)
                    for part in (getattr(content, "parts", None) or []) if content else []:
                        text = getattr(part, "text", None)
                        if text:
                            yield TextDelta(text)
                        fc = getattr(part, "function_call", None)
                        if fc:
                            # Gemini gives no call id. Synthesise one NOW so the
                            # canonical history always has it -- without this a
                            # mid-conversation failover to Groq cannot build a
                            # legal tool message and the result degrades to text.
                            yield ToolCallReady(
                                ToolCall(
                                    id=f"call_{self._turn}_{idx}",
                                    name=fc.name,
                                    args=dict(fc.args or {}),
                                )
                            )
                            idx += 1
                    if getattr(cand, "finish_reason", None):
                        finish = str(cand.finish_reason)

                um = getattr(chunk, "usage_metadata", None)
                if um:
                    yield UsageEvent(
                        Usage(
                            input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
                            extra={"thoughts": getattr(um, "thoughts_token_count", 0) or 0},
                        ),
                        self.name,
                        self.model,
                    )
        except Exception as e:  # noqa: BLE001 -- normalised into ProviderError
            raise _classify(e) from e

        yield StreamEnd("tool_use" if idx else finish)
