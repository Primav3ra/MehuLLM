"""The orchestration loop."""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass
from typing import Any

from mehullm.agent import events as ev
from mehullm.agent.prompt import system_prompt, wrap_untrusted
from mehullm.agent.run_manager import Run
from mehullm.guardrails.interceptor import Interceptor
from mehullm.guardrails.killswitch import Killed, KillSwitch
from mehullm.llm.router import LLMRouter
from mehullm.llm.types import (
    Msg,
    ProviderError,
    StreamEnd,
    TextDelta,
    ToolCall,
    ToolCallReady,
    UsageEvent,
)
from mehullm.mcp.registry import Registry
from mehullm.obs import bind_run, get_logger
from mehullm.persistence import tracing

log = get_logger(__name__)


@dataclass
class LoopConfig:
    max_steps: int = 12
    max_turn_seconds: int = 180
    max_tool_calls_per_turn: int = 25
    max_tokens: int = 4096
    voice_enabled: bool = True


class AgentLoop:
    def __init__(
        self,
        *,
        router: LLMRouter,
        registry: Registry,
        interceptor: Interceptor,
        killswitch: KillSwitch,
        voice=None,
        config: LoopConfig | None = None,
    ):
        self.router = router
        self.registry = registry
        self.interceptor = interceptor
        self.killswitch = killswitch
        self.voice = voice
        self.cfg = config or LoopConfig()

    async def run_turn(self, run: Run, user_message: str, memory_facts: str = "") -> None:
        tracing.bind_trace(run.trace_id)
        bind_run(trace_id=run.trace_id, run_id=run.id)
        with tracing.span("turn", "turn") as turn_span:
            await self._run_turn(run, user_message, memory_facts, turn_span)

    async def _run_turn(
        self, run: Run, user_message: str, memory_facts: str, turn_span: Any
    ) -> None:
        started = time.monotonic()
        tools = self.registry.tool_defs()
        system = system_prompt(memory_facts)
        tool_calls_total = 0
        step = 0
        final_text = ""

        # Ingress: redact PII before anything reaches the hosted model.
        redacted = run.vault.redact(user_message)
        run.history.append(Msg(role="user", text=redacted))

        provider = self.router.pick()
        turn_span.attrs.update(
            tools=len(tools),
            servers=sorted({d.name.split("__")[0] for d in tools}),
            schema_tokens=self.registry.estimated_schema_tokens(),
            voice=type(self.voice).__name__ if self.voice else None,
        )
        turn_span.input_text = user_message
        log.info(
            "turn.start",
            tools=len(tools),
            servers=sorted({d.name.split("__")[0] for d in tools}),
            provider=provider.client.name,
            voice=bool(self.voice),
        )
        await run.emit(
            ev.run_start(
                run.conversation_id, provider.client.name, provider.client.model, len(tools)
            )
        )

        try:
            for step in range(1, self.cfg.max_steps + 1):
                self.killswitch.check()
                if time.monotonic() - started > self.cfg.max_turn_seconds:
                    await run.emit(
                        ev.guardrail_blocked(
                            "wall-clock", "policy", f"turn exceeded {self.cfg.max_turn_seconds}s"
                        )
                    )
                    break

                await run.emit(ev.status("thinking"))

                text_parts: list[str] = []
                calls: list[ToolCall] = []
                async for event in self.router.stream(
                    system=system,
                    messages=run.history,
                    tools=tools,
                    max_tokens=self.cfg.max_tokens,
                ):
                    if isinstance(event, TextDelta):
                        text_parts.append(event.text)
                        await run.emit(ev.text_delta(event.text, step))
                    elif isinstance(event, ToolCallReady):
                        calls.append(event.call)
                    elif isinstance(event, UsageEvent):
                        await run.emit(
                            ev.usage(event.usage.input_tokens, event.usage.output_tokens, step)
                        )
                    elif isinstance(event, StreamEnd):
                        break

                for sw in self.router.drain_switches():
                    await run.emit(ev.provider_switch(sw.from_provider, sw.to_provider, sw.reason))

                assistant_text = "".join(text_parts)
                run.history.append(
                    Msg(role="assistant", text=assistant_text or None, tool_calls=calls)
                )

                if not calls:
                    final_text = assistant_text
                    break

                tool_calls_total += len(calls)
                if tool_calls_total > self.cfg.max_tool_calls_per_turn:
                    await run.emit(
                        ev.guardrail_blocked(
                            "tool-budget",
                            "policy",
                            f"exceeded {self.cfg.max_tool_calls_per_turn} tool calls this turn",
                        )
                    )
                    break

                await run.emit(ev.status("calling_tools", f"{len(calls)} call(s)"))

                # Parallel across calls. Confirmations therefore arrive as a
                # STACK of cards, not a modal -- the UI must handle that.
                outcomes = await asyncio.gather(*(self.interceptor.execute(run, c) for c in calls))
                # Every call gets a result, in ONE batch, in call order.
                for out in outcomes:
                    run.history.append(
                        Msg(
                            role="tool",
                            tool_call_id=out.tool_call_id,
                            tool_name=out.tool_name,
                            text=wrap_untrusted(out.text, out.tool_name),
                            is_error=out.is_error,
                        )
                    )
            else:
                await run.emit(
                    ev.guardrail_blocked(
                        "step-cap", "policy", f"hit the {self.cfg.max_steps}-step limit"
                    )
                )

            final_text = await self._voice(run, final_text)
            run.status = "ok"

        except Killed as e:
            await run.emit(ev.error("killed", str(e)))
            run.status = "killed"
        except asyncio.CancelledError:
            await run.emit(ev.error("cancelled", "run cancelled"))
            run.status = "cancelled"
            raise
        except ProviderError as e:
            run.error = f"{e.kind}: {e}"
            log.warning("turn.provider_error", kind=e.kind, error=str(e))
            await run.emit(ev.error(e.kind, str(e), retriable=e.should_failover))
            run.status = "error"
        except Exception as e:
            run.error = traceback.format_exc()[-2000:]
            log.error("turn.failed", error=str(e), exc_info=True)
            await run.emit(ev.error("internal", f"{type(e).__name__}: {e}"))
            run.status = "error"
        finally:
            run.final_text = final_text
            run.finished = True
            # done is ALWAYS last and ALWAYS exactly one -- the client has a
            # single teardown path.
            await run.emit(
                ev.done(
                    run.status,
                    final_text,
                    step,
                    tool_calls_total,
                    int((time.monotonic() - started) * 1000),
                )
            )

    async def _voice(self, run: Run, draft: str) -> str:
        """Voice only the final turn, and never let it change facts."""
        if not draft:
            return draft

        skip = (
            "disabled"
            if not self.cfg.voice_enabled
            else "no_model"
            if self.voice is None
            else "too_short"
            if len(draft) < 40
            else None
        )
        if skip:
            with tracing.span("voice", "voice_rewrite", skipped=skip) as sp:
                sp.status = "skipped"
            log.info("voice.skip", reason=skip, chars=len(draft))
            await run.emit(ev.voice_delta(draft))
            return draft

        started = time.monotonic()
        await run.emit(ev.voice_start(getattr(self.voice, "model", "voice")))
        with tracing.span("voice", "voice_rewrite") as sp:
            sp.model = getattr(self.voice, "model", "")
            sp.input_text = draft
            try:
                voiced, ok = await self.voice.rewrite(draft)
            except Exception as e:
                sp.status, sp.error = "error", f"{type(e).__name__}: {e}"
                log.warning("voice.failed", error=str(e), exc_info=True)
                await run.emit(ev.voice_delta(draft))
                # invariants_ok=False: nothing ran, so nothing was preserved.
                await run.emit(ev.voice_end(False, True, int((time.monotonic() - started) * 1000)))
                return draft
            sp.output_text = voiced
            sp.attrs["invariants_ok"] = ok

        text = voiced if ok else draft
        await run.emit(ev.voice_delta(text))
        await run.emit(ev.voice_end(ok, not ok, int((time.monotonic() - started) * 1000)))
        return text
