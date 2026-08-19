"""Assemble a real agent for the eval harness."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mehullm.agent.loop import AgentLoop, LoopConfig
from mehullm.eval.bank import Scenario
from mehullm.mcp.hub import MAX_RESULT_CHARS, ToolResult, load_server_specs
from mehullm.mcp.registry import glob_match as _glob

CONFIG_DIR = Path("config")


class _InjectingHub:
    """Delegates to the real hub, except for the one tool a scenario stubs."""

    def __init__(self, inner: Any, inject: dict[str, Any] | None):
        self._inner = inner
        self._inject = inject or {}

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    async def call(self, ref: Any, args: dict[str, Any], timeout: float | None = None):
        """Injected results must be ToolResult, like the real hub returns.

        A bare str made the interceptor raise "'str' object has no attribute
        is_error", and raising for inject.error killed the turn -- so every
        injection scenario failed on contact instead of exercising the guardrail.
        """
        name = getattr(ref, "namespaced", None) or getattr(ref, "name", "")
        want = self._inject.get("tool")
        if not (want and _glob(want, name)):
            return await self._inner.call(ref, args, timeout=timeout)

        if "error" in self._inject:
            msg = str(self._inject["error"])
            return ToolResult(msg, is_error=True, bytes_=len(msg))
        if "result_bytes" in self._inject:
            n = int(self._inject["result_bytes"])
            full = "Subject: mail\n" * (n // 14)
            return ToolResult(full[:MAX_RESULT_CHARS], bytes_=len(full),
                              truncated=len(full) > MAX_RESULT_CHARS)
        out = str(self._inject.get("result", ""))
        return ToolResult(out, bytes_=len(out))


def build_loop_factory(*, voice: bool = True) -> Callable[[Scenario, Path | None], AgentLoop]:
    """Return a factory that builds one AgentLoop per scenario."""
    # `settings`, NOT os.getenv: keys live in .env, which only pydantic-settings.
    from mehullm.settings import settings

    if not settings.gemini_api_key:
        raise ImportError(
            "no GEMINI_API_KEY in .env. "
            "Copy .env.example to .env and add a free key, "
            "or use `mehullm-eval validate` to check the bank offline."
        )

    from mehullm.guardrails.interceptor import Interceptor
    from mehullm.guardrails.killswitch import KillSwitch, RateLimiter
    from mehullm.guardrails.policy import Policy
    from mehullm.llm.providers import build_providers
    from mehullm.llm.quota import QuotaStore
    from mehullm.llm.router import LLMRouter
    from mehullm.mcp.hub import MCPHub
    from mehullm.mcp.registry import Registry
    from mehullm.obs import configure_logging
    from mehullm.persistence import tracing
    from mehullm.persistence.tracing import TraceStore

    # The SHARED builder. This function used to construct its own providers with.
    configure_logging(settings.log_level, settings.log_file)
    tracing.set_store(TraceStore(settings.mehullm_db))
    providers = build_providers()

    router = LLMRouter(providers=providers, quota=QuotaStore(settings.mehullm_db))
    registry = Registry()
    policy = Policy.load(CONFIG_DIR / "policy.yaml")
    killswitch = KillSwitch(settings.mehullm_db)
    hub = MCPHub(specs=load_server_specs(CONFIG_DIR / "servers.yaml"), registry=registry)

    # THE POINT OF THE EVAL. loop.py:190 short-circuits on `self.voice is None`,.
    rewriter = None
    if voice:
        from mehullm.voice.rewrite import voice_if_available

        rewriter = voice_if_available(settings.ollama_host, settings.ollama_voice_model)
        if rewriter is None:
            raise ImportError(
                f"voice requested but {settings.ollama_voice_model!r} is not installed "
                "in ollama. Install it, or pass --no-voice to measure the raw baseline."
            )

    async def warmup() -> dict[str, str]:
        """Connect servers and harvest tool schemas BEFORE the first scenario."""
        return await hub.register_all()

    def factory(scenario: Scenario, db: Path | None) -> AgentLoop:
        # Local tools were never registered here, so local__remember,.
        from mehullm.memory.store import MemoryStore
        from mehullm.tools import local as local_tools_mod

        local_tools = local_tools_mod.register(
            registry, MemoryStore(str(db) if db else settings.memory_db)
        )
        return AgentLoop(
            router=router,
            registry=registry,
            interceptor=Interceptor(
                registry=registry,
                hub=_InjectingHub(hub, scenario.inject),
                policy=policy,
                killswitch=killswitch,
                limiter=RateLimiter(),
                local_tools=local_tools,
                # Scenarios must not sit for two minutes waiting on a human who.
                confirm_timeout_s=10,
            ),
            killswitch=killswitch,
            voice=rewriter,
            config=LoopConfig(voice_enabled=voice, max_turn_seconds=120),
        )

    # Attached rather than returned so build_loop_factory keeps its sync
    # signature and its existing callers.
    factory.warmup = warmup  # type: ignore[attr-defined]
    return factory
