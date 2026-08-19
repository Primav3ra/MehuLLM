"""Multi-server MCP supervisor."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Literal

from mehullm.mcp.registry import SEP, Registry, ToolRef
from mehullm.obs import get_logger

Transport = Literal["stdio", "http"]
State = Literal["idle", "ready", "degraded", "failed", "disabled"]

log = get_logger(__name__)

IS_WINDOWS = sys.platform == "win32"

_PLACEHOLDER = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*")


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    """Read .env directly."""
    from mehullm.settings import ROOT

    out: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def expand(value: str) -> str:
    """Resolve ${VAR} from the real environment first, then .env."""
    return Template(value).safe_substitute({**_dotenv(), **os.environ})


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
    # "auto" negotiates the 2026-07-28 handshake (server/discover) and falls back
    # to "legacy" (initialize) if that fails. Pin a value only to skip a probe.
    connect_mode: str = "auto"

    def resolved_command(self) -> str | None:
        """Resolve .cmd/.bat shims on Windows."""
        if not self.command:
            return None
        if IS_WINDOWS:
            for cand in (self.command, f"{self.command}.cmd", f"{self.command}.exe"):
                found = shutil.which(cand)
                if found:
                    return found
        return shutil.which(self.command) or self.command

    def expanded_env(self) -> dict[str, str]:
        return {k: expand(v) for k, v in self.env.items()}

    def expanded_headers(self) -> dict[str, str]:
        return {k: expand(v) for k, v in self.headers.items()}

    def unresolved(self) -> list[str]:
        """Placeholders that expanded to nothing -- surfaced in the start report so a missing token reads as a missing to."""
        out = []
        for src in (self.env, self.headers):
            for k, v in src.items():
                if _PLACEHOLDER.search(expand(v)):
                    out.append(f"{k}={v}")
        return out


def load_server_specs(path: str | Path) -> list[ServerSpec]:
    """Unknown keys are dropped: servers.yaml is hand-edited."""
    import yaml

    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    fields = ServerSpec.__dataclass_fields__
    specs = []
    for raw in doc.get("servers", []):
        kw = {k: v for k, v in raw.items() if k in fields}
        if isinstance(tools := raw.get("tools"), dict) and tools.get("allow"):
            kw["allow"] = set(tools["allow"])
        specs.append(ServerSpec(**kw))
    return specs


@dataclass
class ToolResult:
    text: str
    is_error: bool = False
    bytes_: int = 0
    truncated: bool = False


MAX_RESULT_CHARS = 32_000


def describe_exc(e: BaseException, depth: int = 0) -> str:
    """Flatten ExceptionGroups."""
    if isinstance(e, BaseExceptionGroup) and depth < 4:
        inner = "; ".join(describe_exc(x, depth + 1) for x in e.exceptions)
        return f"{type(e).__name__}[{inner}]"
    text = str(e).strip() or repr(e)
    return f"{type(e).__name__}: {text[:300]}"


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
        """mcp SDK 2.0."""
        from mcp import Client

        # Servers span protocol eras: "auto" tries the modern handshake, "legacy"
        # the older initialize call.
        modes = ["auto", "legacy"] if self.spec.connect_mode == "auto" else [self.spec.connect_mode]
        last: Exception | None = None

        for mode in modes:
            self._stack = AsyncExitStack()
            try:
                transport = await self._build_transport()
                # Client enters the transport itself and performs the handshake.
                self.session = await self._stack.enter_async_context(Client(transport, mode=mode))
                self.protocol_version = str(getattr(self.session, "protocol_version", "") or "")
                self.state = "ready"
                self.error = None
                self.last_used = time.monotonic()
                return
            except Exception as e:
                last = e
                # The transport generator is single-use, so a retry must rebuild
                # it from scratch -- hence the stack is torn down here.
                await self.aclose()

        self.state = "failed"
        self.error = describe_exc(last) if last else "connect failed"
        raise last or RuntimeError(self.error)

    async def _build_transport(self) -> Any:
        from mcp.client.stdio import StdioServerParameters, stdio_client

        assert self._stack is not None
        if self.spec.transport == "http":
            import httpx2
            from mcp.client.streamable_http import streamable_http_client

            # v2 dropped the headers= kwarg; auth rides on the httpx2 client.
            http = await self._stack.enter_async_context(
                httpx2.AsyncClient(headers=self.spec.expanded_headers())
            )
            return streamable_http_client(self.spec.url, http_client=http)

        # resolved_command() stays: the SDK has get_windows_executable_command.
        cmd = self.spec.resolved_command()
        if not cmd:
            raise RuntimeError(f"command {self.spec.command!r} not found on PATH")
        return stdio_client(
            StdioServerParameters(
                command=cmd,
                args=self.spec.args,
                env={**os.environ, **self.spec.expanded_env()},
                cwd=self.spec.cwd,
            )
        )

    async def list_tools(self) -> list[Any]:
        """Follows next_cursor: v2's tools/list is paginated, and a server with more tools than one page would silently r."""
        await self.ensure()
        out: list[Any] = []
        cursor: str | None = None
        for _ in range(10):  # bounded: a server looping cursors must not hang us
            res = await self.session.list_tools(cursor=cursor)
            out.extend(getattr(res, "tools", None) or [])
            cursor = getattr(res, "next_cursor", None)
            if not cursor:
                break
        return out

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
            except Exception as e:
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
            text = (
                text[:MAX_RESULT_CHARS]
                + f"\n[truncated; {n - MAX_RESULT_CHARS} chars omitted — narrow your query]"
            )
        return ToolResult(
            text, is_error=bool(getattr(res, "isError", False)), bytes_=n, truncated=truncated
        )

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
        # An empty Registry is falsy, so `registry or Registry()` would discard it.
        self.registry = Registry() if registry is None else registry

    async def start(self) -> dict[str, str]:
        """Connect non-lazy servers and register their tools."""
        report: dict[str, str] = {}
        for sid, conn in self.conns.items():
            if conn.spec.lazy:
                report[sid] = "lazy"
                continue
            if missing := conn.spec.unresolved():
                report[sid] = f"failed: unresolved config placeholders {missing}"
                continue
            try:
                # Bounded: otherwise one slow remote server stalls the whole startup.
                tools = await asyncio.wait_for(conn.list_tools(), timeout=conn.spec.timeout_s)
                n = self.registry.ingest(sid, tools, conn.spec.allow)
                report[sid] = f"ready ({n} tools, protocol {conn.protocol_version or '?'})"
            except (Exception, TimeoutError) as e:
                report[sid] = f"failed: {describe_exc(e)}"
        return report

    async def register_all(self) -> dict[str, str]:
        """start() PLUS schema harvest for lazy servers."""
        report = await self.start()
        for sid, conn in self.conns.items():
            if not conn.spec.lazy or conn.state == "disabled":
                continue
            if missing := conn.spec.unresolved():
                report[sid] = f"skipped: unresolved placeholders {missing}"
                continue
            try:
                await asyncio.wait_for(self.ensure_registered(sid), timeout=conn.spec.timeout_s)
                n = sum(1 for d in self.registry.tool_defs() if d.name.startswith(f"{sid}{SEP}"))
                report[sid] = f"ready ({n} tools, protocol {conn.protocol_version or '?'})"
            except (Exception, TimeoutError) as e:
                self.registry.drop_server(sid)
                report[sid] = f"failed: {describe_exc(e)}"
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
            # Bounded: a lazy server's first connect spawns a process and
            # handshakes, either of which can hang indefinitely.
            await asyncio.wait_for(
                self.ensure_registered(ref.server_id), timeout=conn.spec.timeout_s
            )
        except (Exception, TimeoutError) as e:
            self.registry.drop_server(ref.server_id)
            return ToolResult(
                f"Server '{ref.server_id}' is unavailable this turn "
                f"({describe_exc(e)}). Use another approach or tell the user.",
                is_error=True,
            )
        return await conn.call(ref.tool_name, args, timeout)

    async def reap_idle(self) -> list[str]:
        """Free idle stdio processes, KEEPING their tool schemas registered."""
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
                killed.append(sid)
        if killed:
            log.info("mcp.reaped", servers=killed)
        return killed

    async def reaper(self, every_s: float = 60.0) -> None:
        """Background loop. idle_kill_s was configured but never scheduled."""
        while True:
            await asyncio.sleep(every_s)
            with contextlib.suppress(Exception):
                await self.reap_idle()

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
        """Windows has no SIGTERM and stdio children can outlive us."""
        for conn in self.conns.values():
            proc = getattr(conn.session, "_process", None)
            pid = getattr(proc, "pid", None)
            if pid:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
