"""Multi-server MCP supervisor.

Owns connection lifecycle, per-server concurrency, health/restart, and dispatch.
The official `mcp` SDK handles the wire (JSON-RPC framing, protocol-era
negotiation, transport headers); everything above that is here.

Two constraints that shaped this:

* STDIO NEEDS A SEMAPHORE OF 1. stdio is a single newline-delimited duplex
  channel. Concurrent calls are legal at protocol level but many community
  servers are single-threaded and will interleave or deadlock. Serialise per
  stdio server; parallelise across servers.

* WINDOWS. `npx`/`uvx` are .cmd shims that CreateProcess cannot exec directly;
  there is no SIGTERM; orphaned node processes survive a crashed backend and eat
  RAM this machine does not have. All three are handled below.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Literal

from mehullm.mcp.registry import Registry, ToolRef

Transport = Literal["stdio", "http"]
State = Literal["idle", "ready", "degraded", "failed", "disabled"]

IS_WINDOWS = sys.platform == "win32"


@dataclass
class ServerSpec:
    id: str
    transport: Transport = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    allow: set[str] | None = None
    timeout_s: float = 30.0
    lazy: bool = False
    idle_kill_s: float = 600.0
    max_restarts: int = 3

    def resolved_command(self) -> str | None:
        """Resolve .cmd/.bat shims on Windows.

        `npx` on Windows is `npx.cmd`; CreateProcess cannot execute it directly
        and you get a bare FileNotFoundError that looks like the tool is missing.
        """
        if not self.command:
            return None
        if IS_WINDOWS:
            for cand in (self.command, f"{self.command}.cmd", f"{self.command}.exe"):
                found = shutil.which(cand)
                if found:
                    return found
        return shutil.which(self.command) or self.command

    def expanded_env(self) -> dict[str, str]:
        return {k: os.path.expandvars(v) for k, v in self.env.items()}

    def expanded_headers(self) -> dict[str, str]:
        return {k: os.path.expandvars(v) for k, v in self.headers.items()}


@dataclass
class ToolResult:
    text: str
    is_error: bool = False
    bytes_: int = 0
    truncated: bool = False


MAX_RESULT_CHARS = 32_000


class ServerConn:
    def __init__(self, spec: ServerSpec):
        self.spec = spec
        self.state: State = "idle"
        self.error: str | None = None
        self.protocol_version: str | None = None
        self.last_used = 0.0
        self.restarts = 0
        self.session: Any = None
        self._stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()
        # stdio is ONE duplex channel -> serialise. http can fan out.
        self._sem = asyncio.Semaphore(1 if spec.transport == "stdio" else 4)

    async def ensure(self) -> None:
        if self.state == "ready":
            return
        if self.state == "disabled":
            raise RuntimeError(f"server {self.spec.id} is disabled")
        async with self._lock:
            if self.state == "ready":
                return
            await self._connect()

    async def _connect(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        try:
            if self.spec.transport == "http":
                from mcp.client.streamable_http import streamablehttp_client

                read, write, *_ = await self._stack.enter_async_context(
                    streamablehttp_client(self.spec.url, headers=self.spec.expanded_headers())
                )
            else:
                cmd = self.spec.resolved_command()
                if not cmd:
                    raise RuntimeError(f"command {self.spec.command!r} not found on PATH")
                params = StdioServerParameters(
                    command=cmd,
                    args=self.spec.args,
                    env={**os.environ, **self.spec.expanded_env()},
                    cwd=self.spec.cwd,
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))

            self.session = await self._stack.enter_async_context(ClientSession(read, write))
            init = await self.session.initialize()
            self.protocol_version = getattr(init, "protocolVersion", None) or str(
                getattr(init, "protocol_version", "") or ""
            )
            self.state = "ready"
            self.error = None
            self.last_used = time.monotonic()
        except Exception as e:  # noqa: BLE001
            await self.aclose()
            self.state = "failed"
            self.error = f"{type(e).__name__}: {e}"
            raise

    async def list_tools(self) -> list[Any]:
        await self.ensure()
        res = await self.session.list_tools()
        return list(getattr(res, "tools", None) or [])

    async def call(self, tool: str, args: dict, timeout: float | None = None) -> ToolResult:
        await self.ensure()
        async with self._sem:
            self.last_used = time.monotonic()
            try:
                res = await asyncio.wait_for(
                    self.session.call_tool(tool, args), timeout or self.spec.timeout_s
                )
            except TimeoutError:
                self.state = "degraded"
                return ToolResult(
                    f"Tool timed out after {timeout or self.spec.timeout_s:.0f}s. "
                    "Do not retry the same call; try a narrower query.",
                    is_error=True,
                )
            except Exception as e:  # noqa: BLE001
                self.state = "degraded"
                self.error = str(e)
                return ToolResult(f"Tool error: {e}", is_error=True)

        return self._flatten(res)

    @staticmethod
    def _flatten(res: Any) -> ToolResult:
        parts: list[str] = []
        for c in getattr(res, "content", None) or []:
            t = getattr(c, "text", None)
            parts.append(t if t is not None else str(c))
        text = "\n".join(parts) if parts else str(getattr(res, "structuredContent", "") or "")
        n = len(text)
        truncated = n > MAX_RESULT_CHARS
        if truncated:
            # One list_messages on a busy chat returns 400 KB and blows both the
            # context window and the TPM ceiling in a single call.
            text = text[:MAX_RESULT_CHARS] + f"\n[truncated; {n - MAX_RESULT_CHARS} chars omitted — narrow your query]"
        return ToolResult(text, is_error=bool(getattr(res, "isError", False)), bytes_=n, truncated=truncated)

    async def aclose(self) -> None:
        if self._stack is not None:
            # Shutdown must never raise -- a failure here would mask the real
            # error that triggered the shutdown in the first place.
            with contextlib.suppress(Exception):
                await self._stack.aclose()
            self._stack = None
        self.session = None
        if self.state != "disabled":
            self.state = "idle"


class MCPHub:
    def __init__(self, specs: list[ServerSpec], registry: Registry | None = None):
        self.specs = {s.id: s for s in specs}
        self.conns = {s.id: ServerConn(s) for s in specs}
        self.registry = registry or Registry()

    async def start(self) -> dict[str, str]:
        """Connect non-lazy servers and register their tools.

        A server that fails to start does NOT abort startup -- its tools are
        simply absent and the agent is told the capability is unavailable.
        """
        report: dict[str, str] = {}
        for sid, conn in self.conns.items():
            if conn.spec.lazy:
                report[sid] = "lazy"
                continue
            try:
                tools = await conn.list_tools()
                n = self.registry.ingest(sid, tools, conn.spec.allow)
                report[sid] = f"ready ({n} tools, protocol {conn.protocol_version or '?'})"
            except Exception as e:  # noqa: BLE001
                report[sid] = f"failed: {e}"
        return report

    async def ensure_registered(self, server_id: str) -> None:
        """Spawn a lazy server on first use and register its tools."""
        conn = self.conns[server_id]
        if conn.state != "ready":
            tools = await conn.list_tools()
            self.registry.ingest(server_id, tools, conn.spec.allow)

    async def call(self, ref: ToolRef, args: dict, timeout: float | None = None) -> ToolResult:
        conn = self.conns.get(ref.server_id)
        if conn is None:
            return ToolResult(f"Unknown server {ref.server_id!r}.", is_error=True)
        if conn.state == "disabled":
            return ToolResult(
                f"Server '{ref.server_id}' is disabled. Use another approach or tell the user.",
                is_error=True,
            )
        try:
            await self.ensure_registered(ref.server_id)
        except Exception as e:  # noqa: BLE001
            self.registry.drop_server(ref.server_id)
            return ToolResult(
                f"Server '{ref.server_id}' is unavailable this turn ({e}). "
                "Use another approach or tell the user.",
                is_error=True,
            )
        return await conn.call(ref.tool_name, args, timeout)

    async def reap_idle(self) -> list[str]:
        """Shut down idle stdio servers. Each is a full node/python process and
        this machine has ~0.4 GB of free RAM."""
        killed = []
        now = time.monotonic()
        for sid, conn in self.conns.items():
            if (
                conn.state == "ready"
                and conn.spec.transport == "stdio"
                and conn.spec.idle_kill_s
                and now - conn.last_used > conn.spec.idle_kill_s
            ):
                await conn.aclose()
                self.registry.drop_server(sid)
                killed.append(sid)
        return killed

    async def set_enabled(self, server_id: str, enabled: bool) -> None:
        conn = self.conns[server_id]
        if enabled:
            conn.state = "idle"
        else:
            await conn.aclose()
            conn.state = "disabled"
            self.registry.drop_server(server_id)

    def status(self) -> list[dict]:
        return [
            {
                "id": sid,
                "transport": c.spec.transport,
                "state": c.state,
                "protocol_version": c.protocol_version,
                "last_error": c.error,
                "tools": sum(1 for r in self.registry.describe() if r["server"] == sid),
            }
            for sid, c in self.conns.items()
        ]

    async def aclose(self) -> None:
        for conn in self.conns.values():
            await conn.aclose()
        if IS_WINDOWS:
            self._reap_orphans()

    def _reap_orphans(self) -> None:
        """Windows has no SIGTERM and stdio children can outlive us.

        Best-effort: the process tree is killed via taskkill /T. Without this a
        crashed backend leaves node processes holding hundreds of MB.
        """
        for conn in self.conns.values():
            proc = getattr(conn.session, "_process", None)
            pid = getattr(proc, "pid", None)
            if pid:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
