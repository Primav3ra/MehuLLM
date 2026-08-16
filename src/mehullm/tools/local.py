"""Native (non-MCP) tools, registered alongside MCP tools.

These are how the agent actually reaches memory. Deliberately native rather than
an MCP server: they need in-process access to the store and the run's vault, and
routing them through a subprocess would buy nothing on a machine this tight.

They still go through the SAME guardrail interceptor as every MCP tool -- there
is no second path to a tool call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from mehullm.llm.types import ToolDef
from mehullm.memory.embed import embed_passages
from mehullm.memory.retrieve import render_memory_block, search_facts, search_style
from mehullm.memory.store import MemoryStore

TOOL_DEFS: list[ToolDef] = [
    ToolDef(
        name="local__memory_search",
        description=(
            "[local] Search Mehul's personal memory: facts about his life, past "
            "conversations, and people he knows. Call this whenever the answer "
            "depends on something personal to him rather than general knowledge "
            "or something you can look up on the web."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "k": {"type": "integer", "description": "How many results (default 8)."},
            },
            "required": ["query"],
        },
    ),
    ToolDef(
        name="local__remember",
        description=(
            "[local] Store a durable fact about Mehul for future conversations. "
            "Use only for things that stay true (where he lives, what he is "
            "working on, preferences) -- not for transient state."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "A standalone sentence, e.g. 'Mehul prefers dark mode.'",
                },
                "predicate": {
                    "type": "string",
                    "description": "snake_case relation, e.g. prefers, lives_in, works_at.",
                },
                "object": {"type": "string", "description": "The value of the relation."},
            },
            "required": ["text", "predicate", "object"],
        },
    ),
    ToolDef(
        name="local__now",
        description="[local] Current date and time. Use before any relative-date reasoning.",
        parameters={"type": "object", "properties": {}},
    ),
]

SINGLE_VALUED = {"lives_in", "works_at", "studies_at", "current_role", "current_project"}


@dataclass
class LocalTools:
    store: MemoryStore

    async def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Returns (text, is_error) -- same shape the MCP hub returns."""
        try:
            if name == "local__memory_search":
                return self._search(args), False
            if name == "local__remember":
                return self._remember(args), False
            if name == "local__now":
                return time.strftime("%A, %d %B %Y, %H:%M %Z"), False
        except Exception as e:  # noqa: BLE001 -- surfaces as a tool error, never kills the turn
            return f"Local tool error: {type(e).__name__}: {e}", True
        return f"Unknown local tool {name!r}.", True

    def _search(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "No query given."
        k = int(args.get("k", 8) or 8)

        facts = search_facts(self.store, query, k=k)
        style = search_style(self.store, query, k=3)

        parts = []
        if facts:
            parts.append("FACTS:\n" + render_memory_block(facts))
        if style:
            # Past conversation excerpts. Labelled separately so the model does
            # not confuse "something he once said" with "an established fact".
            parts.append(
                "PAST MESSAGES (context, not necessarily still true):\n"
                + "\n".join(f"- {h.text[:200]}" for h in style)
            )
        return "\n\n".join(parts) if parts else "Nothing in memory matches that."

    def _remember(self, args: dict) -> str:
        text = str(args.get("text", "")).strip()
        pred = str(args.get("predicate", "")).strip().lower().replace(" ", "_")
        obj = str(args.get("object", "")).strip()
        if not (text and pred and obj):
            return "Need text, predicate and object."

        vec = embed_passages([text])[0]
        near = self.store.nearest_fact(vec)
        if near and near[1] < 0.08:
            self.store.merge_fact(near[0], [], 0.8)
            return f"Already knew that (merged into F{near[0]})."

        fid = self.store.add_fact(
            subject="Mehul",
            predicate=pred,
            object_=obj,
            text=text,
            embedding=vec,
            single_valued=pred in SINGLE_VALUED,
            confidence=0.85,          # user-asserted beats extracted
            observed_at=int(time.time()),
        )
        return f"Remembered as [F{fid}]."


def register(registry, store: MemoryStore) -> LocalTools:
    """Add local tools to the shared registry so they sit beside MCP tools."""
    from mehullm.mcp.registry import ToolRef

    tools = LocalTools(store)
    for td in TOOL_DEFS:
        short = td.name.split("__", 1)[1]
        registry._refs[td.name] = ToolRef(  # noqa: SLF001 -- registry is ours
            server_id="local",
            tool_name=short,
            namespaced=td.name,
            annotations={"readOnlyHint": short in ("memory_search", "now")},
        )
        registry._defs[td.name] = td  # noqa: SLF001
    return tools
