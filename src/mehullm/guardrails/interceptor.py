"""The single choke point for every tool call."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from mehullm.agent import events as ev
from mehullm.agent.run_manager import Decision, Run
from mehullm.guardrails.killswitch import Killed, KillSwitch, RateLimited, RateLimiter
from mehullm.guardrails.policy import Policy
from mehullm.llm.types import ToolCall
from mehullm.mcp.hub import MCPHub
from mehullm.mcp.registry import Registry
from mehullm.obs import get_logger
from mehullm.persistence import tracing

log = get_logger(__name__)

PREVIEW = 400


@dataclass
class ToolOutcome:
    tool_call_id: str
    tool_name: str
    text: str
    is_error: bool


@dataclass
class _LocalResult:
    """Shapes a local-tool return like an MCPHub ToolResult so the rest of the interceptor is identical for both."""

    text: str
    is_error: bool = False
    bytes_: int = 0
    truncated: bool = False

    def __post_init__(self) -> None:
        self.bytes_ = len(self.text)


class Interceptor:
    def __init__(
        self,
        *,
        registry: Registry,
        hub: MCPHub,
        policy: Policy,
        killswitch: KillSwitch,
        limiter: RateLimiter | None = None,
        confirm_timeout_s: int = 120,
        local_tools: Any = None,
    ):
        self.registry = registry
        self.hub = hub
        self.policy = policy
        self.killswitch = killswitch
        self.limiter = limiter or RateLimiter()
        self.confirm_timeout_s = confirm_timeout_s
        # Native memory tools. Routed through THIS SAME path, not around it --
        # there is exactly one way to call a tool.
        self.local_tools = local_tools

    async def execute(self, run: Run, call: ToolCall) -> ToolOutcome:
        with tracing.span("tool", "tool_call", tool=call.name) as sp:
            return await self._execute(run, call, sp)

    async def _execute(self, run: Run, call: ToolCall, sp: Any) -> ToolOutcome:
        started = time.monotonic()

        def err(msg: str) -> ToolOutcome:
            return ToolOutcome(call.id, call.name, msg, True)

        try:
            self.killswitch.check()
        except Killed as e:
            await run.emit(ev.guardrail_blocked("kill", "kill_switch", str(e), call.name))
            return err(str(e))

        ref = self.registry.resolve(call.name)
        if ref is None:
            # Never split the model's string -- always dict-lookup.
            known = sorted(d.name for d in self.registry.tool_defs())
            sp.status, sp.error = "error", f"unknown tool {call.name}"
            log.warning(
                "tool.unknown",
                tool=call.name,
                registered=len(known),
                servers=sorted({k.split("__")[0] for k in known}),
            )
            return err(f"Unknown tool {call.name!r}. Use one of the tools provided.")

        verdict = self.policy.evaluate(
            call.name, read_only_hint=ref.read_only_hint, provenance=run.provenance
        )

        if verdict.action == "deny":
            await run.emit(
                ev.guardrail_blocked(verdict.rule_id, "policy", verdict.reason, call.name)
            )
            return err(
                f"Blocked by policy: {verdict.reason or verdict.rule_id}. "
                "Explain this to the user and offer an alternative."
            )

        try:
            self.limiter.acquire(call.name, ref.server_id)
        except RateLimited as e:
            await run.emit(ev.guardrail_blocked("ratelimit", "rate_limit", str(e), call.name))
            return err(f"{e}. Try again later or use a different approach.")

        # Rehydrate BEFORE showing the confirmation card: the whole point of
        # human review is seeing the real values that will actually be sent.
        args = run.vault.rehydrate(call.args)

        if (
            verdict.action == "confirm"
            and _key(ref.server_id, ref.tool_name) not in run.session_grants
        ):
            with tracing.span("confirm", "guardrail", tool=call.name) as csp:
                decision = await self._confirm(run, call, ref, verdict, args)
                csp.attrs.update(decision=decision.approved, by=decision.by)
            if not decision.approved:
                await run.emit(ev.confirmation_resolved(decision.reason or "", "deny", decision.by))
                return err(
                    f"The user declined this action. {decision.reason} "
                    "Do not attempt it via another tool."
                )

        await run.emit(
            ev.tool_start(call.id, call.name, ref.server_id, verdict.risk, _preview(call.args))
        )

        with tracing.span("dispatch", "tool_call", server=ref.server_id) as dsp:
            if ref.server_id == "local" and self.local_tools is not None:
                text_out, is_err = await self.local_tools.call(call.name, args)
                result = _LocalResult(text_out, is_err)
            else:
                result = await self.hub.call(ref, args)
            dsp.attrs["is_error"] = result.is_error

        text, secrets = run.vault.scrub_secrets(result.text)
        if secrets:
            await run.emit(
                ev.guardrail_blocked(
                    "secret-scan",
                    "secret_scan",
                    f"redacted {', '.join(secrets)} from tool output",
                    call.name,
                )
            )
        text = run.vault.redact(text)
        run.provenance.add(ref.server_id)

        ms = int((time.monotonic() - started) * 1000)
        sp.output_text = text[:PREVIEW]
        sp.attrs.update(server=ref.server_id, risk=verdict.risk, ok=not result.is_error)
        await run.emit(
            ev.tool_result(
                call.id,
                call.name,
                not result.is_error,
                ms,
                text[:PREVIEW],
                result.bytes_,
                result.truncated,
            )
        )
        return ToolOutcome(call.id, call.name, text, result.is_error)

    async def _confirm(self, run: Run, call: ToolCall, ref, verdict, args: dict) -> Decision:
        iid, fut = run.new_interaction()
        expires = time.time() + self.confirm_timeout_s
        await run.emit(
            ev.confirmation_request(
                interaction_id=iid,
                tool=call.name,
                server=ref.server_id,
                risk=verdict.risk,
                rule=verdict.rule_id,
                summary=_summarize(call.name, args),
                arguments=args,
                timeout_s=self.confirm_timeout_s,
                expires_at=expires,
            )
        )
        try:
            decision: Decision = await asyncio.wait_for(fut, self.confirm_timeout_s)
        except TimeoutError:
            # Timeout is a DENY, not an error. The model handles this gracefully
            # because the system prompt says denial is expected.
            decision = Decision(False, "No response within the time limit.", by="timeout")
        finally:
            run.pending.pop(iid, None)

        await run.emit(
            ev.confirmation_resolved(iid, "approve" if decision.approved else "deny", decision.by)
        )
        return decision


def _key(server: str, tool: str) -> str:
    return f"{server}:{tool}"


def _preview(args: Any, n: int = 160) -> str:
    import json

    try:
        s = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(args)
    return s[:n] + ("…" if len(s) > n else "")


def _summarize(tool: str, args: dict) -> str:
    """Human-readable one-liner for the confirmation card."""
    for key in ("to", "recipient", "email", "url", "repo", "path", "query", "title", "subject"):
        if key in args and isinstance(args[key], str):
            return f"{tool} — {key}: {args[key][:120]}"
    return f"{tool} — {_preview(args, 120)}"
