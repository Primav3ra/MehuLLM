"""Assemble a real agent for the eval harness.

Everything is the production object except the MCP hub, which is wrapped so a
scenario's `inject:` can substitute a tool result. Injection replaces only the
transport's return value -- policy, provenance escalation, secret scanning and
truncation still run, which is the point of the injection scenarios.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mehullm.agent.loop import AgentLoop, LoopConfig
from mehullm.eval.bank import Scenario

CONFIG_DIR = Path("config")


class _InjectingHub:
    """Delegates to the real hub, except for the one tool a scenario stubs."""

    def __init__(self, inner: Any, inject: dict[str, Any] | None):
        self._inner = inner
        self._inject = inject or {}

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    async def call(self, ref: Any, args: dict[str, Any], timeout: float | None = None):
        name = getattr(ref, "namespaced", None) or getattr(ref, "name", "")
        want = self._inject.get("tool")
        if want and _glob(want, name):
            if "error" in self._inject:
                raise RuntimeError(self._inject["error"])
            if "result_bytes" in self._inject:
                # Oversized-result scenario: the 32 KB truncation in the
                # interceptor is what is under test, so generate the bulk here
                # rather than storing 400 KB of filler in a YAML file.
                n = int(self._inject["result_bytes"])
                return "Subject: mail\n" * (n // 14)
            return self._inject.get("result", "")
        return await self._inner.call(ref, args, timeout=timeout)


def _glob(pattern: str, name: str) -> bool:
    import re
    if "*" not in pattern:
        return pattern == name
    rx = "^" + ".*".join(re.escape(p) for p in pattern.split("*")) + "$"
    return re.match(rx, name) is not None


def load_server_specs(path: str | Path) -> list[Any]:
    """Parse servers.yaml into ServerSpec objects.

    Unknown keys are dropped rather than raising: servers.yaml is hand-edited,
    and a stray comment-turned-key should not stop an eval run.
    """
    import yaml

    from mehullm.mcp.hub import ServerSpec

    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    specs = []
    fields = ServerSpec.__dataclass_fields__
    for raw in doc.get("servers", []):
        kw = {k: v for k, v in raw.items() if k in fields}
        if isinstance(tools := raw.get("tools"), dict) and "allow" in tools:
            kw["allow"] = set(tools["allow"])
        specs.append(ServerSpec(**kw))
    return specs


def build_loop_factory(*, voice: bool = True) -> Callable[[Scenario, Path | None], AgentLoop]:
    """Return a factory that builds one AgentLoop per scenario.

    Raises ImportError with an actionable message when no brain is configured --
    the bank still validates offline, so a missing key should not read as a
    broken harness.
    """
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")):
        raise ImportError(
            "no GEMINI_API_KEY or GROQ_API_KEY in the environment. "
            "Copy .env.example to .env and add a free key, "
            "or use `mehullm-eval validate` to check the bank offline."
        )

    from mehullm.guardrails.interceptor import Interceptor
    from mehullm.guardrails.killswitch import KillSwitch
    from mehullm.guardrails.policy import Policy
    from mehullm.llm.quota import Limits, QuotaStore
    from mehullm.llm.router import LLMRouter, Provider
    from mehullm.mcp.hub import MCPHub
    from mehullm.mcp.registry import Registry

    providers: list[Provider] = []
    if key := os.getenv("GEMINI_API_KEY"):
        from mehullm.llm.gemini.client import GeminiClient
        providers.append(Provider(
            client=GeminiClient(key, os.getenv("GEMINI_MODEL", "gemini-2.5-flash")),
            # Gemini's free-tier RPM/RPD are no longer published (§4), so these
            # are conservative placeholders; the quota store learns the real
            # ceiling from 429s.
            limits=Limits(rpm=10, rpd=200), priority=0))
    if key := os.getenv("GROQ_API_KEY"):
        from mehullm.llm.groq.client import GroqClient
        providers.append(Provider(
            client=GroqClient(key, os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")),
            limits=Limits(rpm=30, rpd=1000), priority=1))

    router = LLMRouter(providers=providers,
                       quota=QuotaStore("data/derived/quota.db"))
    registry = Registry()
    policy = Policy.load(CONFIG_DIR / "policy.yaml")
    killswitch = KillSwitch("data/derived/runtime.db")
    hub = MCPHub(specs=load_server_specs(CONFIG_DIR / "servers.yaml"),
                 registry=registry)

    def factory(scenario: Scenario, db: Path | None) -> AgentLoop:
        return AgentLoop(
            router=router,
            registry=registry,
            interceptor=Interceptor(
                registry=registry,
                hub=_InjectingHub(hub, scenario.inject),
                policy=policy,
                killswitch=killswitch,
                # Scenarios must not sit for two minutes waiting on a human who
                # is not there; the harness auto-denies, and this bounds the
                # window in which a card could be missed.
                confirm_timeout_s=10,
            ),
            killswitch=killswitch,
            voice=None,
            config=LoopConfig(voice_enabled=voice, max_turn_seconds=120),
        )

    return factory
