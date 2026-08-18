"""Provider router over the Gemini model ladder."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from mehullm.llm.quota import Limits, QuotaStore, day_pt
from mehullm.llm.types import (
    LLMClient,
    LLMEvent,
    Msg,
    ProviderError,
    StreamEnd,
    ToolDef,
    UsageEvent,
)
from mehullm.obs import get_logger
from mehullm.persistence import tracing

log = get_logger(__name__)

RATE_LIMIT_DEMOTE_S = 15 * 60
TRANSIENT_DEMOTE_S = 5 * 60
SHORT_RETRY_MAX_S = 8.0

# Retried in place before any failover. `rate_limit` is included because a.
_RETRY_IN_PLACE = frozenset({"overloaded", "timeout", "rate_limit"})
_TRANSIENT_RETRIES = 2
_RETRY_BASE_S = 1.5


def _seconds_to_pt_midnight() -> float:
    """Quota resets at midnight Pacific, not local midnight."""
    import datetime as _dt

    from mehullm.llm.quota import PT

    now = _dt.datetime.now(PT)
    tomorrow = (now + _dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (tomorrow - now).total_seconds()


@dataclass
class Provider:
    client: LLMClient
    limits: Limits
    priority: int = 0


@dataclass
class RouterEvent:
    """Emitted when the active provider changes, so the UI can show it."""

    from_provider: str
    to_provider: str
    reason: str


@dataclass
class LLMRouter:
    providers: list[Provider]
    quota: QuotaStore
    _active: str | None = None
    _switch_log: list[RouterEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.providers.sort(key=lambda p: p.priority)

    def _by_name(self, name: str) -> Provider | None:
        return next((p for p in self.providers if p.client.name == name), None)

    def available(self) -> list[Provider]:
        return [p for p in self.providers if self.quota.is_available(p.client.name, p.limits)]

    def pick(self) -> Provider:
        """Sticky: keep the current provider while it is still usable."""
        if self._active:
            cur = self._by_name(self._active)
            if cur and self.quota.is_available(cur.client.name, cur.limits):
                return cur
        for p in self.providers:
            if self.quota.is_available(p.client.name, p.limits):
                self._note_switch(p.client.name, "preflight")
                return p
        # Everything is exhausted. Surface it, then let the real error come from
        # the API rather than inventing a synthetic one.
        log.warning("providers.all_exhausted", providers=[p.client.name for p in self.providers])
        return self.providers[0]

    def _note_switch(self, to: str, reason: str) -> None:
        if self._active and self._active != to:
            self._switch_log.append(RouterEvent(self._active, to, reason))
        self._active = to

    def drain_switches(self) -> list[RouterEvent]:
        out, self._switch_log = self._switch_log, []
        return out

    def _demote(self, provider: str, err: ProviderError) -> None:
        if err.kind in ("model_unavailable", "quota"):
            until = time.time() + _seconds_to_pt_midnight()
            if err.kind == "quota":
                # Only a PerDay 429 teaches us a daily ceiling.
                self.quota.note_daily_limit(provider, err.limit)
        elif err.kind == "rate_limit":
            # Per-minute. Honour Google's own retryDelay; never touch the daily
            # ceiling, or one 16s blip caps the model for the rest of the day.
            until = time.time() + min(err.retry_after or 60.0, RATE_LIMIT_DEMOTE_S)
        else:
            until = time.time() + TRANSIENT_DEMOTE_S
        log.info(
            "provider.demoted",
            provider=provider,
            kind=err.kind,
            seconds=round(until - time.time()),
            retry_after=err.retry_after,
            reported_limit=err.limit,
        )
        self.quota.demote(provider, until, f"{err.kind}: {err}")

    async def stream(
        self,
        *,
        system: str,
        messages: list[Msg],
        tools: list[ToolDef],
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMEvent]:
        """Stream from the best available provider, failing over on the way."""
        tried: set[str] = set()
        last: ProviderError | None = None

        for _ in range(len(self.providers)):
            provider = self.pick()
            name = provider.client.name
            if name in tried:
                remaining = [p for p in self.providers if p.client.name not in tried]
                if not remaining:
                    break
                provider = remaining[0]
                name = provider.client.name
            tried.add(name)
            self._note_switch(name, "start")

            buffered: list[LLMEvent] = []
            usage_in = usage_out = 0
            t_start = time.monotonic()
            # Accounting MUST happen in a finally. This is an async generator and.
            recorded = False
            try:
                # Transient failures get ONE in-place retry before failover.
                for attempt in range(_TRANSIENT_RETRIES + 1):
                    t0 = time.monotonic()
                    try:
                        async for ev in provider.client.stream(
                            system=system,
                            messages=messages,
                            tools=tools,
                            max_tokens=max_tokens,
                        ):
                            if isinstance(ev, UsageEvent):
                                usage_in += ev.usage.input_tokens
                                usage_out += ev.usage.output_tokens
                            buffered.append(ev)
                            yield ev
                            if isinstance(ev, StreamEnd):
                                break
                        break
                    except ProviderError as e:
                        tracing.record_span(
                            "llm",
                            "llm_call",
                            int((time.monotonic() - t0) * 1000),
                            provider=name,
                            model=provider.client.model,
                            status="error",
                            error=f"{e.kind}: {e}",
                            attempt=attempt,
                        )
                        retriable = e.kind in _RETRY_IN_PLACE and not buffered
                        if attempt >= _TRANSIENT_RETRIES or not retriable:
                            raise
                        self.quota.record(name, provider.client.model, ok=False, error_kind=e.kind)
                        delay = e.retry_after or _RETRY_BASE_S * (2**attempt)
                        # Jitter so concurrent runs do not retry in lockstep.
                        await asyncio.sleep(min(delay, 8.0) * (0.5 + random.random()))
                        self._note_switch(name, f"retry:{e.kind}")
            except ProviderError as e:
                last = e
                self.quota.record(name, provider.client.model, ok=False, error_kind=e.kind)
                recorded = True
                if not e.should_failover:
                    raise  # bad_request / auth: surface it, do not mask it
                self._demote(name, e)
                if buffered:
                    # Partial output already went to the client. Signal the
                    # restart rather than silently splicing two models together.
                    yield StreamEnd("provider_switch")
                continue
            finally:
                # Runs on normal completion AND on generator close, which is the
                # only reason a successful request gets counted at all.
                tracing.record_span(
                    "llm",
                    "llm_call",
                    int((time.monotonic() - t_start) * 1000),
                    provider=name,
                    model=provider.client.model,
                    tokens_in=usage_in,
                    tokens_out=usage_out,
                )
                if not recorded:
                    self.quota.record(
                        name,
                        provider.client.model,
                        input_tokens=usage_in,
                        output_tokens=usage_out,
                    )
            return

        raise last or ProviderError("quota", "all providers exhausted")

    def status(self) -> dict:
        return {
            "active": self._active,
            "day_pt": day_pt(),
            "providers": [
                {
                    "name": p.client.name,
                    "model": p.client.model,
                    "available": self.quota.is_available(p.client.name, p.limits),
                    "limits": {"rpm": p.limits.rpm, "rpd": p.limits.rpd},
                    **self.quota.snapshot().get(p.client.name, {}),
                }
                for p in self.providers
            ],
        }
