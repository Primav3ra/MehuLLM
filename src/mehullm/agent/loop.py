"""The orchestration loop.

Hand-rolled deliberately. The confirmation gate must suspend mid-flight and
resume from a different HTTP request minutes later, which needs the run to be a
durable object rather than a callback -- and the loop is the artifact this
project is actually about.

Correctness rules baked in below:

* ALL tool results go back in ONE turn. Splitting them across turns trains the
  model to stop making parallel calls (true on both providers).
* EVERY tool call gets a matching result, including denied and errored ones.
  A missing result is a 400.
* Hard step cap. A runaway loop burns a free-tier daily quota in ninety seconds.
* Only the FINAL turn is voiced. Voicing in-loop narration would double latency
  on every step for no benefit.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

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
        await run.emit(
            ev.run_start(run.conversation_id, provider.client.name,
                         provider.client.model, len(tools))
        )

        try:
            for step in range(1, self.cfg.max_steps + 1):
                self.killswitch.check()
                if time.monotonic() - started > self.cfg.max_turn_seconds:
                    await run.emit(ev.guardrail_blocked(
                        "wall-clock", "policy",
                        f"turn exceeded {self.cfg.max_turn_seconds}s"))
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
                        await run.emit(ev.usage(
                            event.usage.input_tokens, event.usage.output_tokens, step))
                    elif isinstance(event, StreamEnd):
                        break

                for sw in self.router.drain_switches():
                    await run.emit(ev.provider_switch(
                        sw.from_provider, sw.to_provider, sw.reason))

                assistant_text = "".join(text_parts)
                run.history.append(
                    Msg(role="assistant", text=assistant_text or None, tool_calls=calls)
                )

                if not calls:
                    final_text = assistant_text
                    break

                tool_calls_total += len(calls)
                if tool_calls_total > self.cfg.max_tool_calls_per_turn:
                    await run.emit(ev.guardrail_blocked(
                        "tool-budget", "policy",
                        f"exceeded {self.cfg.max_tool_calls_per_turn} tool calls this turn"))
                    break

                await run.emit(ev.status("calling_tools", f"{len(calls)} call(s)"))

                # Parallel across calls. Confirmations therefore arrive as a
                # STACK of cards, not a modal -- the UI must handle that.
                outcomes = await asyncio.gather(
                    *(self.interceptor.execute(run, c) for c in calls)
                )
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
                await run.emit(ev.guardrail_blocked(
                    "step-cap", "policy", f"hit the {self.cfg.max_steps}-step limit"))

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
            await run.emit(ev.error(e.kind, str(e), retriable=e.should_failover))
            run.status = "error"
        except Exception as e:  # noqa: BLE001
            await run.emit(ev.error("internal", f"{type(e).__name__}: {e}"))
            run.status = "error"
        finally:
            run.final_text = final_text
            run.finished = True
            # done is ALWAYS last and ALWAYS exactly one -- the client has a
            # single teardown path.
            await run.emit(ev.done(
                run.status, final_text, step, tool_calls_total,
                int((time.monotonic() - started) * 1000),
            ))

    async def _voice(self, run: Run, draft: str) -> str:
        """Voice only the final turn, and never let it change facts."""
        if not draft:
            return draft
        if not self.cfg.voice_enabled or self.voice is None or len(draft) < 40:
            # Still emitted as voice_delta so the client keeps one render path.
            await run.emit(ev.voice_delta(draft))
            return draft

        started = time.monotonic()
        await run.emit(ev.voice_start(getattr(self.voice, "model", "voice")))
        try:
            voiced, ok = await self.voice.rewrite(draft)
        except Exception:  # noqa: BLE001 -- voice is best-effort, never fatal
            await run.emit(ev.voice_delta(draft))
            await run.emit(ev.voice_end(True, True, int((time.monotonic() - started) * 1000)))
            return draft

        text = voiced if ok else draft
        await run.emit(ev.voice_delta(text))
        await run.emit(
            ev.voice_end(ok, not ok, int((time.monotonic() - started) * 1000))
        )
        return text
