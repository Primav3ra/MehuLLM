"""FastAPI backend."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from mehullm.agent.loop import AgentLoop, LoopConfig
from mehullm.agent.run_manager import Decision, RunManager
from mehullm.guardrails.interceptor import Interceptor
from mehullm.guardrails.killswitch import KillSwitch, RateLimiter
from mehullm.guardrails.policy import Policy
from mehullm.llm.providers import build_providers
from mehullm.llm.quota import QuotaStore
from mehullm.llm.router import LLMRouter
from mehullm.mcp.hub import MCPHub, load_server_specs
from mehullm.mcp.registry import Registry
from mehullm.memory.conversation import ConversationStore
from mehullm.memory.retrieve import render_memory_block, search_facts
from mehullm.obs import bind_run, configure_logging, redacted_len
from mehullm.persistence import tracing
from mehullm.persistence.tracing import TraceStore
from mehullm.settings import settings
from mehullm.voice.rewrite import VoiceRewriter

log = structlog.get_logger(__name__)

app = FastAPI(title="MehuLLM", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Trace-Id"],
)

STATE: dict[str, Any] = {}


def auth(authorization: str = Header(default="")) -> None:
    token = settings.mehullm_api_token
    if not token:
        return  # unset in dev; loopback only
    if authorization.removeprefix("Bearer ").strip() != token:
        raise HTTPException(401, "bad token")


def _voice_if_available() -> VoiceRewriter | None:
    from mehullm.voice.rewrite import voice_if_available

    rewriter = voice_if_available(settings.ollama_host, settings.ollama_voice_model)
    if rewriter is None:
        log.info(
            "voice unavailable; serving brain output unstyled", model=settings.ollama_voice_model
        )
    return rewriter


@app.on_event("startup")
async def startup() -> None:
    configure_logging(settings.log_level, settings.log_file)
    registry = Registry()
    specs = load_server_specs(settings.servers_config)
    for s in specs:
        log.info(
            "server.config",
            server=s.id,
            transport=s.transport,
            secrets={k: redacted_len(v) for k, v in s.expanded_env().items()},
            unresolved=s.unresolved() or None,
        )
    hub = MCPHub(specs, registry)
    report = await hub.register_all()
    STATE["mcp_report"] = report
    STATE["reaper"] = asyncio.create_task(hub.reaper())
    log.info("mcp.ready", **report)
    log.info(
        "tools.registered",
        n=len(registry),
        by_server={
            sid: sorted(
                d.name.split("__", 1)[1]
                for d in registry.tool_defs()
                if d.name.startswith(f"{sid}__")
            )
            for sid in sorted({d.name.split("__")[0] for d in registry.tool_defs()})
        },
        schema_tokens=registry.estimated_schema_tokens(),
    )

    quota = QuotaStore(settings.mehullm_db)
    providers = build_providers()
    log.info("provider ladder", providers=[p.client.name for p in providers])

    # Local tools must be registered BEFORE the loop is built, or the model is
    # never told memory_search/remember/now exist.
    from mehullm.memory.store import MemoryStore
    from mehullm.tools import local as local_tools_mod

    memory = MemoryStore(settings.memory_db)
    local_tools = local_tools_mod.register(registry, memory)
    convos = ConversationStore(memory)
    traces = TraceStore(settings.mehullm_db)
    tracing.set_store(traces)

    ks = KillSwitch(settings.mehullm_db)
    ks.load()
    policy = (
        Policy.load(settings.policy_config) if Path(settings.policy_config).exists() else Policy()
    )

    STATE.update(
        registry=registry,
        hub=hub,
        quota=quota,
        killswitch=ks,
        policy=policy,
        memory=memory,
        convos=convos,
        traces=traces,
        runs=RunManager(),
        router=LLMRouter(providers, quota) if providers else None,
        loop=None,
    )
    if providers:
        STATE["loop"] = AgentLoop(
            router=STATE["router"],
            registry=registry,
            interceptor=Interceptor(
                registry=registry,
                hub=hub,
                policy=policy,
                killswitch=ks,
                limiter=RateLimiter(),
                local_tools=local_tools,
                confirm_timeout_s=settings.mehullm_confirm_timeout_s,
            ),
            killswitch=ks,
            # Only if the LoRA has actually been built. Otherwise every turn
            # pays an Ollama round-trip that can only fail back to the draft.
            voice=_voice_if_available(),
            config=LoopConfig(
                max_turn_seconds=settings.mehullm_max_turn_seconds,
                max_tool_calls_per_turn=settings.mehullm_max_tool_calls_per_turn,
            ),
        )


@app.on_event("shutdown")
async def shutdown() -> None:
    hub: MCPHub | None = STATE.get("hub")
    if hub:
        await hub.aclose()


class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"


class ConfirmRequest(BaseModel):
    interaction_id: str
    decision: str  # "approve" | "deny"
    reason: str = ""
    remember: str = ""  # "session" only -- never persistent


async def _stream(run, after_seq: int = 0) -> AsyncIterator[dict]:
    async for event in run.subscribe(after_seq):
        yield {
            "id": str(event.seq),
            "event": event.type,
            "data": event.to_sse().split("data: ", 1)[1].rstrip("\n"),
        }


@app.post("/api/chat", dependencies=[Depends(auth)])
async def chat(req: ChatRequest):
    loop: AgentLoop | None = STATE.get("loop")
    if loop is None:
        raise HTTPException(503, "No LLM provider configured. Set GEMINI_API_KEY in .env")
    runs: RunManager = STATE["runs"]
    run = runs.create(req.conversation_id)
    tracing.bind_trace(run.trace_id)
    bind_run(trace_id=run.trace_id, run_id=run.id)

    # Retrieve BEFORE the turn starts: the facts go into the system prompt, so
    # they have to be resolved up front rather than fetched mid-stream.
    memory_block = ""
    if (memory := STATE.get("memory")) is not None:
        try:
            with tracing.span("memory", "retrieval", k=settings.memory_facts_k) as sp:
                hits = search_facts(memory, req.message, k=settings.memory_facts_k)
                sp.attrs["hits"] = [h.id for h in hits]
            memory_block = render_memory_block(hits) if hits else ""
        except Exception as e:  # memory is an enhancement, never a hard dependency
            log.warning("memory retrieval failed", error=str(e))

    # Short-term memory: replay recent turns so follow-ups resolve.
    convos: ConversationStore | None = STATE.get("convos")
    if convos is not None:
        run.history.extend(convos.history(req.conversation_id))

    traces: TraceStore | None = STATE.get("traces")
    if traces is not None:
        traces.start_trace(run.trace_id, req.conversation_id, req.message)

    async def _turn_and_persist() -> None:
        try:
            await loop.run_turn(run, req.message, memory_facts=memory_block)
        finally:
            if traces is not None:
                traces.end_trace(
                    run.trace_id,
                    status=run.status,
                    final_output=run.final_text,
                    error=run.error,
                )
            # Persisted even on error: a failed turn the user can still see is
            # better than a hole in the conversation.
            if convos is not None:
                convos.append(req.conversation_id, "user", req.message, run.trace_id)
                convos.append(req.conversation_id, "assistant", run.final_text, run.trace_id)

    # The loop is owned by the RUN, not by this request.
    run.task = asyncio.create_task(_turn_and_persist())
    return EventSourceResponse(_stream(run), ping=15, headers={"X-Trace-Id": run.trace_id})


@app.get("/api/chat/{run_id}/events", dependencies=[Depends(auth)])
async def chat_events(run_id: str, after_seq: int = 0):
    run = STATE["runs"].get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    return EventSourceResponse(_stream(run, after_seq), ping=15)


@app.post("/api/chat/{run_id}/confirm", dependencies=[Depends(auth)])
async def confirm(run_id: str, req: ConfirmRequest):
    run = STATE["runs"].get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    approved = req.decision == "approve"
    if approved and req.remember == "session":
        # SESSION ONLY. A click in a chat UI must never write a persistent
        # grant -- those go through policy.yaml, a file you edit deliberately.
        run.session_grants.add(req.interaction_id)
    ok = run.resolve(req.interaction_id, Decision(approved, req.reason))
    if not ok:
        raise HTTPException(409, "no such pending confirmation")
    return {"ok": True}


@app.post("/api/chat/{run_id}/cancel", dependencies=[Depends(auth)])
async def cancel(run_id: str):
    run = STATE["runs"].get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    run.cancel()
    return {"ok": True}


@app.get("/api/traces", dependencies=[Depends(auth)])
async def traces_list(limit: int = 20):
    traces: TraceStore | None = STATE.get("traces")
    return {"traces": traces.recent(limit) if traces else []}


@app.get("/api/traces/{trace_id}", dependencies=[Depends(auth)])
async def trace_detail(trace_id: str):
    traces: TraceStore | None = STATE.get("traces")
    if traces is None:
        raise HTTPException(503, "tracing unavailable")
    data = traces.get(trace_id)
    if not data.get("trace"):
        raise HTTPException(404, "unknown trace")
    return data


@app.get("/api/conversations", dependencies=[Depends(auth)])
async def conversations(limit: int = 50):
    convos: ConversationStore | None = STATE.get("convos")
    return {"conversations": convos.conversations(limit) if convos else []}


@app.get("/api/conversations/{conversation_id}", dependencies=[Depends(auth)])
async def conversation(conversation_id: str, limit: int = 100):
    convos: ConversationStore | None = STATE.get("convos")
    if convos is None:
        raise HTTPException(503, "conversation store unavailable")
    return {"conversation_id": conversation_id, "turns": convos.recent(conversation_id, limit)}


@app.get("/api/tools", dependencies=[Depends(auth)])
async def tools():
    reg: Registry = STATE["registry"]
    return {
        "count": len(reg),
        "estimated_schema_tokens": reg.estimated_schema_tokens(),
        "tools": [
            {**t, "tier": STATE["policy"].tier_for(t["name"], t["read_only_hint"])}
            for t in reg.describe()
        ],
    }


@app.get("/api/servers", dependencies=[Depends(auth)])
async def servers():
    return {"servers": STATE["hub"].status(), "startup": STATE.get("mcp_report", {})}


@app.post("/api/servers/{server_id}/{action}", dependencies=[Depends(auth)])
async def server_action(server_id: str, action: str):
    hub: MCPHub = STATE["hub"]
    if server_id not in hub.conns:
        raise HTTPException(404, "unknown server")
    if action == "disable":
        await hub.set_enabled(server_id, False)
    elif action == "enable":
        await hub.set_enabled(server_id, True)
    elif action == "reconnect":
        await hub.conns[server_id].aclose()
        with contextlib.suppress(Exception):
            await hub.ensure_registered(server_id)
    else:
        raise HTTPException(400, "action must be enable|disable|reconnect")
    return {"ok": True, "state": hub.conns[server_id].state}


@app.post("/api/admin/kill", dependencies=[Depends(auth)])
async def kill(reason: str = "manual"):
    ks: KillSwitch = STATE["killswitch"]
    ks.engage(reason)
    n = STATE["runs"].cancel_all()
    return {"ok": True, "cancelled": n}


@app.post("/api/admin/resume", dependencies=[Depends(auth)])
async def resume():
    STATE["killswitch"].release()
    return {"ok": True}


@app.get("/api/admin/status", dependencies=[Depends(auth)])
async def status():
    router: LLMRouter | None = STATE.get("router")
    return {
        "killed": STATE["killswitch"].engaged,
        "active_runs": len(STATE["runs"].active()),
        "llm": router.status() if router else None,
        "servers": STATE["hub"].status(),
    }


@app.get("/api/health")
async def health():
    import shutil

    # The import lives inside the guard on purpose: health is what you call.
    free_ram = None
    with contextlib.suppress(Exception):
        import psutil  # type: ignore[import-not-found]

        free_ram = round(psutil.virtual_memory().available / 1e9, 2)
    return {
        "ok": True,
        "llm_configured": STATE.get("router") is not None,
        "tools": len(STATE["registry"]) if STATE.get("registry") else 0,
        "free_ram_gb": free_ram,
        "free_disk_gb": round(shutil.disk_usage(".").free / 1e9, 1),
    }
