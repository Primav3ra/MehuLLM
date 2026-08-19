"""Runs decoupled from HTTP connections."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from mehullm.agent.events import Event
from mehullm.guardrails.redaction import Vault

BUFFER = 2000
SUBSCRIBER_QUEUE = 512


@dataclass
class Decision:
    approved: bool
    reason: str = ""
    by: str = "user"


@dataclass
class Run:
    id: str
    conversation_id: str
    trace_id: str
    task: asyncio.Task | None = None
    buffer: deque[Event] = field(default_factory=lambda: deque(maxlen=BUFFER))
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    pending: dict[str, asyncio.Future] = field(default_factory=dict)
    session_grants: set[str] = field(default_factory=set)
    provenance: set[str] = field(default_factory=set)
    vault: Vault = field(default_factory=Vault)
    history: list[Any] = field(default_factory=list)
    seq: int = 0
    started: float = field(default_factory=time.time)
    finished: bool = False
    status: str = "running"
    final_text: str = ""
    error: str = ""
    tool_memo: dict[str, str] = field(default_factory=dict)

    async def emit(self, ev: Event) -> None:
        self.seq += 1
        ev.seq, ev.run_id, ev.trace_id = self.seq, self.id, self.trace_id
        self.buffer.append(ev)
        for q in list(self.subscribers):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                # Bounded, drop-oldest. A slow browser must never apply
                # backpressure to the agent loop.
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(ev)

    async def subscribe(self, after_seq: int = 0) -> AsyncIterator[Event]:
        """Replay from the ring buffer, then attach live. No gap between the two."""
        q: asyncio.Queue = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE)
        self.subscribers.add(q)
        try:
            replayed = 0
            for ev in list(self.buffer):
                if ev.seq > after_seq:
                    replayed = ev.seq
                    yield ev
            if self.finished and not self.pending:
                return
            while True:
                ev = await q.get()
                if ev.seq <= replayed:
                    continue  # already sent during replay
                yield ev
                if ev.type == "done":
                    return
        finally:
            self.subscribers.discard(q)

    def new_interaction(self) -> tuple[str, asyncio.Future]:
        iid = f"cnf_{uuid.uuid4().hex[:8]}"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending[iid] = fut
        return iid, fut

    def resolve(self, interaction_id: str, decision: Decision) -> bool:
        """Returns False if unknown or already resolved -- double-clicks and retries must 409, not crash."""
        fut = self.pending.get(interaction_id)
        if fut is None or fut.done():
            return False
        fut.set_result(decision)
        return True

    def cancel(self) -> None:
        for fut in self.pending.values():
            if not fut.done():
                fut.set_result(Decision(False, "run cancelled", by="system"))
        if self.task and not self.task.done():
            self.task.cancel()


class RunManager:
    def __init__(self, max_runs: int = 50):
        self._runs: dict[str, Run] = {}
        self._max = max_runs

    def create(self, conversation_id: str) -> Run:
        run = Run(
            id=f"run_{uuid.uuid4().hex[:10]}",
            conversation_id=conversation_id,
            trace_id=f"tr_{uuid.uuid4().hex[:12]}",
        )
        self._runs[run.id] = run
        self._evict()
        return run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def active(self) -> list[Run]:
        return [r for r in self._runs.values() if not r.finished]

    def _evict(self) -> None:
        if len(self._runs) <= self._max:
            return
        finished = sorted((r for r in self._runs.values() if r.finished), key=lambda r: r.started)
        for r in finished[: len(self._runs) - self._max]:
            self._runs.pop(r.id, None)

    def cancel_all(
        self,
    ) -> int:
        n = 0
        for r in self.active():
            r.cancel()
            n += 1
        return n
