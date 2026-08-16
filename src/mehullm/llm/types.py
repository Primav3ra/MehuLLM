"""Provider-agnostic LLM interface.

One protocol, two adapters (Gemini primary, Groq fallback). The agent loop never
sees a provider name.

The abstraction leaks in exactly three places, each contained deliberately:

1. TOOL SCHEMA DIALECT. Gemini accepts an OpenAPI subset and rejects `$ref`,
   `anyOf`, `additionalProperties` and `format` -- all of which appear in
   basically every MCP tool schema. Contained in llm/gemini/schema.py.

2. TOOL CALL IDs. Gemini's function_call parts carry no ID; Groq requires
   `tool_call_id`. Contained by synthesising IDs at ingest, so canonical history
   always has one and either renderer can use it.

3. PROVIDER-OPAQUE BLOBS (Gemini thought signatures). Cannot cross providers.
   Contained in Msg.provider_opaque, keyed by provider; each renderer drops
   anything not its own.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "ToolDef", "ToolCall", "Msg", "Usage",
    "TextDelta", "ToolCallReady", "UsageEvent", "StreamEnd", "LLMEvent",
    "ProviderError", "ErrorKind", "LLMClient",
]

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # neutral JSON Schema; adapters sanitise


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str  # synthesised for Gemini -- see module docstring
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class Msg:
    role: Role
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None   # role == "tool"
    tool_name: str | None = None      # role == "tool"
    is_error: bool = False
    provider_opaque: dict[str, Any] = field(default_factory=dict)  # {"gemini": {...}}


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    extra: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallReady:
    """Emitted only once a call is fully accumulated.

    Groq streams tool arguments as fragmented partial-JSON across chunks, so
    adapters buffer and emit whole calls. The loop never sees a partial call.
    """

    call: ToolCall


@dataclass(frozen=True, slots=True)
class UsageEvent:
    usage: Usage
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class StreamEnd:
    finish_reason: str


LLMEvent = TextDelta | ToolCallReady | UsageEvent | StreamEnd

ErrorKind = Literal[
    "rate_limit", "quota", "overloaded", "timeout", "bad_request", "auth", "unknown"
]


class ProviderError(Exception):
    def __init__(
        self,
        kind: ErrorKind,
        message: str = "",
        *,
        retry_after: float | None = None,
        provider: str = "",
    ):
        super().__init__(message or kind)
        self.kind = kind
        self.retry_after = retry_after
        self.provider = provider

    @property
    def should_failover(self) -> bool:
        """A 400 must NEVER fail over -- it is a schema-sanitiser bug you need
        to see, and retrying it on the other provider just hides it."""
        return self.kind in {"rate_limit", "quota", "overloaded", "timeout"}


@runtime_checkable
class LLMClient(Protocol):
    name: str
    model: str

    def stream(
        self,
        *,
        system: str,
        messages: list[Msg],
        tools: list[ToolDef],
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMEvent]: ...
