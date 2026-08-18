"""SSE event taxonomy — the single source of truth for the frontend contract."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EventType = Literal[
    "run_start",
    "status",
    "text_delta",
    "tool_start",
    "confirmation_request",
    "confirmation_resolved",
    "tool_result",
    "guardrail_blocked",
    "provider_switch",
    "voice_start",
    "voice_delta",
    "voice_end",
    "usage",
    "error",
    "done",
]


@dataclass
class Event:
    type: EventType
    seq: int = 0
    run_id: str = ""
    trace_id: str = ""
    ts: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        payload = {
            "type": self.type,
            "seq": self.seq,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "ts": self.ts,
            **self.data,
        }
        return f"id: {self.seq}\nevent: {self.type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("data")
        return {**d, **self.data}


def _e(t: EventType, **data: Any) -> Event:
    return Event(type=t, data=data)


def run_start(conversation_id: str, provider: str, model: str, tool_count: int) -> Event:
    return _e(
        "run_start",
        conversation_id=conversation_id,
        provider=provider,
        model=model,
        tool_count=tool_count,
    )


def status(stage: str, detail: str = "") -> Event:
    return _e("status", stage=stage, detail=detail)


def text_delta(text: str, step: int = 0) -> Event:
    return _e("text_delta", text=text, step=step)


def tool_start(tool_use_id: str, tool: str, server: str, risk: str, preview: str) -> Event:
    return _e(
        "tool_start",
        tool_use_id=tool_use_id,
        tool=tool,
        server=server,
        risk=risk,
        arguments_preview=preview,
    )


def confirmation_request(
    interaction_id: str,
    tool: str,
    server: str,
    risk: str,
    rule: str,
    summary: str,
    arguments: dict,
    timeout_s: int,
    expires_at: float,
) -> Event:
    return _e(
        "confirmation_request",
        interaction_id=interaction_id,
        tool=tool,
        server=server,
        risk=risk,
        rule=rule,
        summary=summary,
        arguments=arguments,
        sensitive=True,
        options=["approve", "deny"],
        timeout_s=timeout_s,
        expires_at=expires_at,
    )


def confirmation_resolved(interaction_id: str, decision: str, by: str) -> Event:
    return _e("confirmation_resolved", interaction_id=interaction_id, decision=decision, by=by)


def tool_result(
    tool_use_id: str,
    tool: str,
    ok: bool,
    duration_ms: int,
    preview: str,
    bytes_: int,
    truncated: bool,
) -> Event:
    return _e(
        "tool_result",
        tool_use_id=tool_use_id,
        tool=tool,
        ok=ok,
        duration_ms=duration_ms,
        preview=preview,
        bytes=bytes_,
        truncated=truncated,
    )


def guardrail_blocked(rule: str, category: str, message: str, tool: str | None = None) -> Event:
    return _e("guardrail_blocked", rule=rule, category=category, message=message, tool=tool)


def provider_switch(from_provider: str, to_provider: str, reason: str) -> Event:
    return _e("provider_switch", **{"from": from_provider, "to": to_provider, "reason": reason})


def voice_start(model: str) -> Event:
    return _e("voice_start", model=model)


def voice_delta(text: str) -> Event:
    return _e("voice_delta", text=text)


def voice_end(invariants_ok: bool, fell_back: bool, duration_ms: int) -> Event:
    return _e(
        "voice_end", invariants_ok=invariants_ok, fell_back=fell_back, duration_ms=duration_ms
    )


def usage(input_tokens: int, output_tokens: int, step: int) -> Event:
    return _e("usage", input_tokens=input_tokens, output_tokens=output_tokens, step=step)


def error(code: str, message: str, retriable: bool = False) -> Event:
    return _e("error", code=code, message=message, retriable=retriable)


def done(status_: str, final_text: str, steps: int, tool_calls: int, total_ms: int) -> Event:
    return _e(
        "done",
        status=status_,
        final_text=final_text,
        steps=steps,
        tool_calls=tool_calls,
        total_ms=total_ms,
    )
