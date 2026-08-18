"""Provider-neutral message and event types."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "ErrorKind",
    "LLMClient",
    "LLMEvent",
    "Msg",
    "ProviderError",
    "StreamEnd",
    "TextDelta",
    "ToolCall",
    "ToolCallReady",
    "ToolDef",
    "Usage",
    "UsageEvent",
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
    tool_call_id: str | None = None  # role == "tool"
    tool_name: str | None = None  # role == "tool"
    is_error: bool = False


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
    """Emitted only once a call is fully accumulated."""

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
    "rate_limit",
    "quota",
    "overloaded",
    "timeout",
    "bad_request",
    "auth",
    "model_unavailable",
    "unknown",
]


class ProviderError(Exception):
    def __init__(
        self,
        kind: ErrorKind,
        message: str = "",
        *,
        retry_after: float | None = None,
        provider: str = "",
        limit: int | None = None,
    ):
        super().__init__(message or kind)
        self.kind = kind
        self.retry_after = retry_after
        self.provider = provider
        self.limit = limit  # the real ceiling, straight from quotaValue

    @property
    def should_failover(self) -> bool:
        """bad_request never fails over -- it is a sanitiser bug and must surface."""
        return self.kind in {
            "rate_limit",
            "quota",
            "overloaded",
            "timeout",
            "model_unavailable",
        }


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
